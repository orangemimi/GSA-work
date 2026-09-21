"""Reusable CMIP6 discovery, download, validation, and preprocessing helpers.

This module deliberately contains no Case A scientific choices.  A notebook or
script supplies the models, experiments, variables, requested periods, and
output directories.  The helpers then:

* query ESGF search endpoints with retry, fallback, and pagination;
* consolidate mirrored file records;
* resolve one member/grid shared by required time-varying variables;
* validate or download cached files safely through ``.part`` files;
* validate monthly time coverage and coordinate compatibility; and
* convert monthly CMIP6 fluxes to annual totals without changing their sign.

The module does not classify climate regions, sample grid cells, construct
relationship pairs, or invoke the relationship classifier.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SEARCH_ENDPOINTS = (
    "https://esgf-node.llnl.gov/esg-search/search/",
    "https://esgf-data.dkrz.de/esg-search/search/",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search/",
)

DATE_RANGE_RE = re.compile(
    r"_(?P<start>\d{4})(?:\d{2,10})?-(?P<end>\d{4})(?:\d{2,10})?\.nc$"
)


@dataclass
class FileRecord:
    """One logical CMIP6 file, potentially available from several mirrors."""

    source_id: str
    experiment_id: str
    variable_id: str
    table_id: str
    member_id: str
    grid_label: str
    title: str
    version: str = ""
    start_year: int | None = None
    end_year: int | None = None
    size: int = 0
    checksum: str = ""
    checksum_type: str = ""
    candidate_urls: list[str] = field(default_factory=list)
    dataset_id: str = ""
    local_path: str = ""
    status: str = "planned"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_fixed(self) -> bool:
        return self.table_id == "fx" or self.start_year is None


@dataclass
class ValidationResult:
    ok: bool
    status: str
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ResolvedBundle:
    source_id: str
    experiment_id: str
    member_id: str
    grid_label: str
    requested_start_year: int
    requested_end_year: int
    records_by_alias: dict[str, list[FileRecord]]
    optional_missing: list[str] = field(default_factory=list)
    status: str = "ready"
    messages: list[str] = field(default_factory=list)


@dataclass
class DownloadPlan:
    """Resolved member/grid plus the inventory used to produce it."""

    bundle: ResolvedBundle
    fixed_records: dict[str, FileRecord | None]
    discovered_time: dict[str, list[FileRecord]]
    discovered_fixed: dict[str, list[FileRecord]]
    available_period: tuple[int, int] | None = None
    available_variables: list[str] = field(default_factory=list)
    missing_variables: list[str] = field(default_factory=list)
    status: str = "ready"
    messages: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.bundle.source_id,
            "experiment": self.bundle.experiment_id,
            "selected_member": self.bundle.member_id,
            "selected_grid": self.bundle.grid_label,
            "requested_period": [
                self.bundle.requested_start_year,
                self.bundle.requested_end_year,
            ],
            "available_period": (
                list(self.available_period) if self.available_period else None
            ),
            "available_variables": list(self.available_variables),
            "missing_variables": list(self.missing_variables),
            "status": self.status,
            "messages": list(self.messages),
        }


def _scalar(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def file_years(title: str) -> tuple[int, int] | None:
    """Extract a CMIP-style filename year range; fixed fields return ``None``."""

    match = DATE_RANGE_RE.search(title)
    if match is None:
        return None
    return int(match.group("start")), int(match.group("end"))


def httpserver_urls(doc: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    for encoded in _as_list(doc.get("url")):
        parts = str(encoded).split("|")
        if len(parts) >= 3 and parts[-1] == "HTTPServer":
            url = parts[0]
            if url.startswith(("https://", "http://")):
                urls.append(url)
    return urls


def url_priority(url: str) -> tuple[int, int, str]:
    """Prefer HTTPS and established European mirrors, deterministically."""

    preferred_hosts = ("dkrz.de", "ipsl.fr", "ceda.ac.uk")
    host_rank = 0 if any(host in url for host in preferred_hosts) else 1
    scheme_rank = 0 if url.startswith("https://") else 1
    return host_rank, scheme_rank, url


def _url_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


_HOST_PROBE_CACHE: dict[str, float] = {}
_HOST_PROBE_BAD: set[str] = set()
_HOST_PROBE_LOCK = threading.Lock()


def mark_host_failed(url: str) -> None:
    """Stop preferring a host after 404 / hard failure."""
    host = _url_host(url)
    with _HOST_PROBE_LOCK:
        _HOST_PROBE_BAD.add(host)
        _HOST_PROBE_CACHE.pop(host, None)


def _probe_url(url: str, nbytes: int, timeout: int) -> int:
    """Read up to ``nbytes`` from ``url`` and return how many bytes arrived."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CaseA-CMIP6-pipeline/1.0",
            "Range": f"bytes=0-{nbytes - 1}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunk = response.read(nbytes)
    return len(chunk)


def pick_fastest_url(
    urls: Sequence[str],
    *,
    probe_bytes: int = 2 * 1024 * 1024,
    max_candidates: int = 3,
    timeout: int = 10,
) -> list[str]:
    """Order mirrors by a short range-request probe, then by ``url_priority``.

    Host timings are cached for the process so a 17-variable model does not
    probe DKRZ/CEDA/LLNL on every file.  Failed probes are skipped; if every
    probe fails the original priority order is returned.
    """

    ordered = sorted({url for url in urls if url}, key=url_priority)
    with _HOST_PROBE_LOCK:
        bad = set(_HOST_PROBE_BAD)
    preferred = [url for url in ordered if _url_host(url) not in bad]
    fallback = [url for url in ordered if _url_host(url) in bad]
    ordered = preferred + fallback
    if len(ordered) <= 1:
        return ordered

    scored: list[tuple[float, str]] = []
    for url in ordered[:max_candidates]:
        if _url_host(url) in bad:
            continue
        host = _url_host(url)
        with _HOST_PROBE_LOCK:
            cached = _HOST_PROBE_CACHE.get(host)
        if cached is not None:
            scored.append((cached, url))
            continue
        try:
            started = time.perf_counter()
            nbytes = _probe_url(url, probe_bytes, timeout)
            elapsed = time.perf_counter() - started
            if nbytes <= 0 or elapsed <= 0:
                continue
            seconds_per_byte = elapsed / nbytes
            with _HOST_PROBE_LOCK:
                _HOST_PROBE_CACHE[host] = seconds_per_byte
            scored.append((seconds_per_byte, url))
        except Exception:
            continue

    if not scored:
        return ordered
    scored.sort(key=lambda item: item[0])
    ranked = [url for _, url in scored]
    ranked_set = set(ranked)
    return ranked + [url for url in ordered if url not in ranked_set]


def _request_json(url: str, *, timeout: int, retries: int, backoff: float) -> dict:
    errors: list[str] = []
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "CaseA-CMIP6-pipeline/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
            if attempt < retries:
                time.sleep(min(backoff * (2**attempt), 20.0))
    raise RuntimeError("ESGF query failed: " + " | ".join(errors))


def query_esgf(
    facets: Mapping[str, Any],
    *,
    endpoints: Sequence[str] = DEFAULT_SEARCH_ENDPOINTS,
    timeout: int = 120,
    retries: int = 2,
    backoff: float = 1.0,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Query ESGF with endpoint fallback and pagination.

    ``facets`` should contain CMIP6 facets such as ``source_id``,
    ``experiment_id``, and ``variable_id``.  File-level records are requested.
    The first endpoint that successfully yields a complete paginated response
    is used; a failed endpoint does not silently produce an empty inventory.
    """

    base_params: dict[str, Any] = {
        "project": "CMIP6",
        "type": "File",
        "format": "application/solr+json",
        "latest": "true",
        **{key: value for key, value in facets.items() if value is not None},
    }
    endpoint_errors: list[str] = []
    successful_empty_response = False

    for endpoint in endpoints:
        docs: list[dict[str, Any]] = []
        offset = 0
        try:
            while True:
                params = {**base_params, "limit": page_size, "offset": offset}
                url = endpoint + "?" + urllib.parse.urlencode(params, doseq=True)
                payload = _request_json(
                    url, timeout=timeout, retries=retries, backoff=backoff
                )
                response = payload.get("response", {})
                page = list(response.get("docs", []))
                num_found = int(response.get("numFound", len(page)))
                docs.extend(page)
                offset += len(page)
                if not page or offset >= num_found:
                    break
            if docs:
                return docs
            successful_empty_response = True
        except Exception as exc:
            endpoint_errors.append(f"{endpoint}: {exc}")

    if successful_empty_response:
        return []
    raise RuntimeError(
        "All ESGF search endpoints failed:\n" + "\n".join(endpoint_errors)
    )


def _doc_member(doc: Mapping[str, Any]) -> str:
    return str(_scalar(doc.get("variant_label"), _scalar(doc.get("member_id"), "")))


def _doc_version(doc: Mapping[str, Any]) -> str:
    value = _scalar(doc.get("version"), "")
    return str(value)


def _version_key(version: str) -> tuple[int, str]:
    digits = "".join(ch for ch in version if ch.isdigit())
    return (int(digits) if digits else -1, version)


def records_from_docs(
    docs: Iterable[Mapping[str, Any]],
    *,
    requested_start_year: int | None = None,
    requested_end_year: int | None = None,
    fixed: bool = False,
) -> list[FileRecord]:
    """Consolidate mirrored ESGF docs into logical file records."""

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for doc in docs:
        title = str(doc.get("title", ""))
        if not title.endswith(".nc"):
            continue
        years = file_years(title)
        if not fixed:
            if years is None:
                continue
            if requested_start_year is not None and years[1] < requested_start_year:
                continue
            if requested_end_year is not None and years[0] > requested_end_year:
                continue
        urls = httpserver_urls(doc)
        if not urls:
            continue
        key = (
            title,
            _doc_member(doc),
            str(_scalar(doc.get("grid_label"), "")),
            str(_scalar(doc.get("table_id"), "")),
            _doc_version(doc),
        )
        grouped.setdefault(key, []).append(doc)

    # A title may be returned for more than one dataset version.  Retain only
    # the newest version for each title/member/grid/table combination.
    newest: dict[tuple[str, str, str, str], tuple[str, list[Mapping[str, Any]]]] = {}
    for (title, member, grid, table, version), copies in grouped.items():
        key = (title, member, grid, table)
        old = newest.get(key)
        if old is None or _version_key(version) > _version_key(old[0]):
            newest[key] = (version, copies)

    records: list[FileRecord] = []
    for (title, member, grid, table), (version, copies) in sorted(newest.items()):
        reference = max(copies, key=lambda item: int(item.get("size", 0) or 0))
        years = file_years(title)
        urls = sorted(
            {url for doc in copies for url in httpserver_urls(doc)}, key=url_priority
        )
        record = FileRecord(
            source_id=str(_scalar(reference.get("source_id"), "")),
            experiment_id=str(_scalar(reference.get("experiment_id"), "")),
            variable_id=str(_scalar(reference.get("variable_id"), "")),
            table_id=table,
            member_id=member,
            grid_label=grid,
            title=title,
            version=version,
            start_year=None if years is None else years[0],
            end_year=None if years is None else years[1],
            size=int(reference.get("size", 0) or 0),
            checksum=str(_scalar(reference.get("checksum"), "") or ""),
            checksum_type=str(
                _scalar(reference.get("checksum_type"), "") or ""
            ).upper(),
            candidate_urls=urls,
            dataset_id=str(_scalar(reference.get("dataset_id"), "")),
        )
        if not record.checksum or not record.checksum_type:
            record.warnings.append("ESGF checksum unavailable")
        records.append(record)
    return records


def discover_file_records(
    *,
    source_id: str,
    experiment_id: str | None,
    variable_id: str,
    allowed_tables: Sequence[str],
    requested_start_year: int | None = None,
    requested_end_year: int | None = None,
    fixed: bool = False,
    member_id: str | None = None,
    endpoints: Sequence[str] = DEFAULT_SEARCH_ENDPOINTS,
    activity_id: str = "CMIP",
) -> list[FileRecord]:
    """Discover one variable across its allowed CMIP6 tables.

    Pass ``member_id`` whenever it is known.  File-level ESGF listings
    otherwise include every ensemble member and can be huge.
    """

    records: list[FileRecord] = []
    for table_id in allowed_tables:
        facets: dict[str, Any] = {
            "activity_id": activity_id,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "variable_id": variable_id,
            "table_id": table_id,
        }
        if member_id:
            facets["variant_label"] = member_id
        docs = query_esgf(
            facets,
            endpoints=endpoints,
        )
        records.extend(
            records_from_docs(
                docs,
                requested_start_year=requested_start_year,
                requested_end_year=requested_end_year,
                fixed=fixed,
            )
        )
    return records


def coverage_intervals(records: Iterable[FileRecord]) -> list[tuple[int, int]]:
    intervals = sorted(
        (record.start_year, record.end_year)
        for record in records
        if record.start_year is not None and record.end_year is not None
    )
    if not intervals:
        return []
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for start, end in intervals[1:]:
        assert start is not None and end is not None
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], int(end))
        else:
            merged.append([int(start), int(end)])
    return [(start, end) for start, end in merged]


def covers_period(records: Iterable[FileRecord], start_year: int, end_year: int) -> bool:
    return any(
        start <= start_year and end >= end_year
        for start, end in coverage_intervals(records)
    )


def _preference_rank(value: str, preferred: str) -> tuple[int, str]:
    return (0 if value == preferred else 1, value)


def resolve_compatible_bundle(
    *,
    source_id: str,
    experiment_id: str,
    variable_specs: Mapping[str, Mapping[str, Any]],
    records_by_alias: Mapping[str, Sequence[FileRecord]],
    requested_start_year: int,
    requested_end_year: int,
    preferred_member: str = "r1i1p1f1",
    preferred_grid: str = "gn",
) -> ResolvedBundle:
    """Resolve a member/grid bundle shared by all required time variables."""

    required_aliases = [
        alias for alias, spec in variable_specs.items() if bool(spec.get("required", True))
    ]
    optional_aliases = [alias for alias in variable_specs if alias not in required_aliases]
    if not required_aliases:
        raise ValueError("At least one time-varying variable must be required")

    available_by_alias: dict[str, set[tuple[str, str]]] = {}
    for alias in required_aliases:
        records = list(records_by_alias.get(alias, []))
        combos = {(r.member_id, r.grid_label) for r in records if r.member_id and r.grid_label}
        valid_combos = {
            combo
            for combo in combos
            if covers_period(
                [r for r in records if (r.member_id, r.grid_label) == combo],
                requested_start_year,
                requested_end_year,
            )
        }
        available_by_alias[alias] = valid_combos

    missing_required = [alias for alias, combos in available_by_alias.items() if not combos]
    if missing_required:
        return ResolvedBundle(
            source_id=source_id,
            experiment_id=experiment_id,
            member_id="",
            grid_label="",
            requested_start_year=requested_start_year,
            requested_end_year=requested_end_year,
            records_by_alias={},
            status="missing_required_variable",
            messages=["No period-complete candidates for: " + ", ".join(missing_required)],
        )

    common_combos = set.intersection(*(available_by_alias[a] for a in required_aliases))
    if not common_combos:
        return ResolvedBundle(
            source_id=source_id,
            experiment_id=experiment_id,
            member_id="",
            grid_label="",
            requested_start_year=requested_start_year,
            requested_end_year=requested_end_year,
            records_by_alias={},
            status="no_common_member_grid",
            messages=["Required variables have no common period-complete member/grid"],
        )

    selected_member, selected_grid = sorted(
        common_combos,
        key=lambda combo: (
            _preference_rank(combo[0], preferred_member),
            _preference_rank(combo[1], preferred_grid),
        ),
    )[0]

    selected: dict[str, list[FileRecord]] = {}
    optional_missing: list[str] = []
    for alias in variable_specs:
        matching = [
            record
            for record in records_by_alias.get(alias, [])
            if record.member_id == selected_member
            and record.grid_label == selected_grid
            and (
                record.is_fixed
                or (record.end_year is not None and record.end_year >= requested_start_year)
                and (record.start_year is not None and record.start_year <= requested_end_year)
            )
        ]
        if alias in required_aliases:
            selected[alias] = matching
        elif covers_period(matching, requested_start_year, requested_end_year):
            selected[alias] = matching
        else:
            optional_missing.append(alias)

    return ResolvedBundle(
        source_id=source_id,
        experiment_id=experiment_id,
        member_id=selected_member,
        grid_label=selected_grid,
        requested_start_year=requested_start_year,
        requested_end_year=requested_end_year,
        records_by_alias=selected,
        optional_missing=optional_missing,
    )


def _available_period(records: Iterable[FileRecord]) -> tuple[int, int] | None:
    intervals = coverage_intervals(records)
    if not intervals:
        return None
    return intervals[0][0], intervals[-1][1]


def build_download_plan(
    *,
    source_id: str,
    experiment_id: str,
    start_year: int,
    end_year: int,
    time_variables: Mapping[str, Mapping[str, Any]],
    fixed_fields: Mapping[str, Mapping[str, Any]] | None = None,
    preferred_member: str = "r1i1p1f1",
    preferred_grid: str = "gn",
    activity_id: str = "CMIP",
    endpoints: Sequence[str] = DEFAULT_SEARCH_ENDPOINTS,
) -> DownloadPlan:
    """Discover files and resolve one compatible member/grid bundle.

    This function only queries ESGF and builds a plan.  It does not download.
    Required time variables without a common period-complete member/grid are
    reported and skipped; conditions are never silently relaxed.
    """

    fixed_fields = dict(fixed_fields or {})
    discovered_time: dict[str, list[FileRecord]] = {}
    for alias, spec in time_variables.items():
        discovered_time[alias] = discover_file_records(
            source_id=source_id,
            experiment_id=experiment_id,
            variable_id=str(spec["variable_id"]),
            allowed_tables=list(spec.get("allowed_tables", [])),
            requested_start_year=start_year,
            requested_end_year=end_year,
            fixed=False,
            endpoints=endpoints,
            activity_id=activity_id,
        )

    bundle = resolve_compatible_bundle(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_specs=time_variables,
        records_by_alias=discovered_time,
        requested_start_year=start_year,
        requested_end_year=end_year,
        preferred_member=preferred_member,
        preferred_grid=preferred_grid,
    )

    discovered_fixed: dict[str, list[FileRecord]] = {}
    fixed_records: dict[str, FileRecord | None] = {}
    messages = list(bundle.messages)
    status = bundle.status

    for alias, spec in fixed_fields.items():
        records = discover_file_records(
            source_id=source_id,
            experiment_id=experiment_id,
            variable_id=str(spec["variable_id"]),
            allowed_tables=list(spec.get("allowed_tables", [])),
            fixed=True,
            endpoints=endpoints,
            activity_id=activity_id,
        )
        discovered_fixed[alias] = records
        selected = None
        if bundle.grid_label:
            selected = select_fixed_record(
                records,
                selected_grid=bundle.grid_label,
                preferred_member=bundle.member_id or preferred_member,
            )
        fixed_records[alias] = selected
        if selected is None:
            label = f"{alias} ({spec.get('variable_id', '')})"
            if bool(spec.get("required", True)):
                if status == "ready":
                    status = "missing_required_fixed_field"
                messages.append(f"Required fixed field missing for selected grid: {label}")
            else:
                messages.append(f"Optional fixed field unavailable: {label}")

    available_variables = [
        alias for alias, records in bundle.records_by_alias.items() if records
    ]
    available_variables.extend(
        alias for alias, record in fixed_records.items() if record is not None
    )
    missing_variables = list(bundle.optional_missing)
    for alias, spec in time_variables.items():
        if alias not in bundle.records_by_alias and alias not in missing_variables:
            missing_variables.append(alias)
    for alias, record in fixed_records.items():
        if record is None:
            missing_variables.append(alias)

    selected_time = [
        record
        for records in bundle.records_by_alias.values()
        for record in records
        if not record.is_fixed
    ]
    available_period = _available_period(selected_time)

    bundle.status = status
    bundle.messages = messages
    return DownloadPlan(
        bundle=bundle,
        fixed_records=fixed_records,
        discovered_time=discovered_time,
        discovered_fixed=discovered_fixed,
        available_period=available_period,
        available_variables=available_variables,
        missing_variables=missing_variables,
        status=status,
        messages=messages,
    )


def select_fixed_record(
    records: Sequence[FileRecord],
    *,
    selected_grid: str,
    preferred_member: str,
) -> FileRecord | None:
    """Select a fixed field compatible with the chosen analysis grid."""

    candidates = [record for record in records if record.grid_label == selected_grid]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda record: (
            _preference_rank(record.member_id, preferred_member),
            tuple(-x if isinstance(x, int) else x for x in _version_key(record.version)),
            record.title,
        ),
    )[0]


def digest(path: Path, algorithm: str) -> str:
    normalized = algorithm.lower().replace("-", "")
    if not normalized:
        raise ValueError("Checksum algorithm is empty")
    hasher = hashlib.new(normalized)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_checksum_and_size(path: Path, record: FileRecord) -> ValidationResult:
    messages: list[str] = []
    warns: list[str] = []
    if not path.exists():
        return ValidationResult(False, "missing", [f"Missing file: {path}"])
    actual_size = path.stat().st_size
    if actual_size <= 0:
        return ValidationResult(False, "empty", [f"Empty file: {path}"])

    if record.checksum and record.checksum_type:
        try:
            actual = digest(path, record.checksum_type)
        except (ValueError, OSError) as exc:
            return ValidationResult(False, "checksum_error", [str(exc)])
        if actual.lower() != record.checksum.lower():
            return ValidationResult(
                False,
                "checksum_mismatch",
                [f"Expected {record.checksum}, found {actual}"],
            )
        messages.append(f"{record.checksum_type} checksum verified")
        if record.size and actual_size != record.size:
            warns.append(
                f"Size differs from ESGF listing ({actual_size} vs {record.size}); "
                "checksum matched"
            )
        return ValidationResult(True, "file_integrity_ok", messages, warns)

    if record.size:
        if actual_size != record.size:
            return ValidationResult(
                False,
                "size_mismatch",
                [f"Expected {record.size} bytes, found {actual_size}"],
            )
        warns.append("Checksum unavailable; used size and NetCDF validation")
    else:
        warns.append("Checksum and expected size unavailable; used NetCDF validation")
    return ValidationResult(True, "file_integrity_ok", messages, warns)


def validate_local_netcdf(
    path: Path,
    record: FileRecord,
    *,
    requested_start_year: int | None = None,
    requested_end_year: int | None = None,
    require_monthly_coverage: bool = False,
) -> ValidationResult:
    """Validate file integrity, metadata, variable, and optional monthly coverage."""

    integrity = _validate_checksum_and_size(path, record)
    if not integrity.ok:
        return integrity

    messages = list(integrity.messages)
    warns = list(integrity.warnings)
    try:
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            if record.variable_id not in dataset.variables:
                return ValidationResult(
                    False,
                    "variable_missing",
                    [f"{record.variable_id!r} is absent from {path.name}"],
                    warns,
                )
            attrs = dataset.attrs
            expected_attrs = {
                "source_id": record.source_id,
                "experiment_id": record.experiment_id,
                "grid_label": record.grid_label,
            }
            for key, expected in expected_attrs.items():
                actual = str(attrs.get(key, ""))
                if expected and actual and actual != expected:
                    return ValidationResult(
                        False,
                        "metadata_mismatch",
                        [f"{key}: expected {expected}, found {actual}"],
                        warns,
                    )

            if require_monthly_coverage:
                if "time" not in dataset.coords:
                    return ValidationResult(False, "time_missing", ["time coordinate absent"], warns)
                if requested_start_year is None or requested_end_year is None:
                    raise ValueError("Requested years are required for monthly validation")
                years = np.asarray(dataset["time"].dt.year.values, dtype=int)
                months = np.asarray(dataset["time"].dt.month.values, dtype=int)
                observed = {
                    (int(year), int(month))
                    for year, month in zip(years, months)
                    if requested_start_year <= year <= requested_end_year
                }
                expected = {
                    (year, month)
                    for year in range(requested_start_year, requested_end_year + 1)
                    for month in range(1, 13)
                }
                if observed != expected:
                    missing = sorted(expected - observed)
                    extra = sorted(observed - expected)
                    return ValidationResult(
                        False,
                        "monthly_coverage_incomplete",
                        [
                            f"missing months={missing[:12]}{'...' if len(missing) > 12 else ''}",
                            f"unexpected months={extra[:12]}{'...' if len(extra) > 12 else ''}",
                        ],
                        warns,
                    )
        messages.append("NetCDF structure and metadata verified")
        return ValidationResult(True, "complete", messages, warns)
    except Exception as exc:
        return ValidationResult(False, "netcdf_invalid", [str(exc)], warns)


def local_cache_status(
    records: Sequence[FileRecord],
    target_dir: Path,
    *,
    requested_start_year: int | None = None,
    requested_end_year: int | None = None,
    require_monthly_coverage: bool = False,
) -> ValidationResult:
    """Classify a local file set as complete, missing, partial, or invalid."""

    if not records:
        return ValidationResult(False, "missing", ["No file records to validate"])

    messages: list[str] = []
    warns: list[str] = []
    statuses: list[str] = []
    part_names: list[str] = []
    for record in records:
        target = target_dir / record.title
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            part_names.append(record.title)
        result = validate_local_netcdf(
            target,
            record,
            requested_start_year=requested_start_year,
            requested_end_year=requested_end_year,
            require_monthly_coverage=require_monthly_coverage and not record.is_fixed,
        )
        statuses.append(result.status)
        messages.extend(result.messages)
        warns.extend(result.warnings)

    if part_names:
        return ValidationResult(
            False,
            "partial",
            [f".part file present: {name}" for name in part_names] + messages,
            warns,
        )
    if all(status == "complete" for status in statuses):
        return ValidationResult(True, "complete", messages, warns)
    if all(status == "missing" for status in statuses):
        return ValidationResult(False, "missing", messages, warns)
    if any(status == "monthly_coverage_incomplete" for status in statuses):
        return ValidationResult(False, "period_incomplete", messages, warns)
    if any(status == "missing" for status in statuses):
        return ValidationResult(False, "partial", messages, warns)
    return ValidationResult(False, "invalid", messages, warns)


def _download_stream(
    url: str,
    partial: Path,
    *,
    timeout: int,
) -> None:
    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "CaseA-CMIP6-pipeline/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        append = resume_at > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)


def download_record(
    record: FileRecord,
    target_dir: Path,
    *,
    timeout: int = 180,
    retries_per_mirror: int = 1,
) -> Path:
    """Validate a cached file or download it safely from ESGF mirrors.

    Checksum verification still runs twice for a fresh download: once on the
    ``.part`` file and again inside ``validate_local_netcdf`` after rename.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / record.title
    partial = target.with_suffix(target.suffix + ".part")
    record.local_path = str(target)

    if target.exists():
        validation = validate_local_netcdf(target, record)
        record.warnings.extend(validation.warnings)
        if validation.ok:
            record.status = "existing-verified"
            size = target.stat().st_size
            print(
                f"    {record.variable_id}: cache {human_size(size)} "
                f"existing-verified ({record.title})"
            )
            return target
        record.warnings.extend(validation.messages)

    errors: list[str] = []
    for url in pick_fastest_url(record.candidate_urls):
        for attempt in range(retries_per_mirror + 1):
            try:
                started = time.perf_counter()
                _download_stream(url, partial, timeout=timeout)
                elapsed = max(time.perf_counter() - started, 1e-6)
                integrity = _validate_checksum_and_size(partial, record)
                if not integrity.ok:
                    if partial.exists():
                        partial.unlink()
                    raise RuntimeError("; ".join(integrity.messages))
                partial.replace(target)
                validation = validate_local_netcdf(target, record)
                record.warnings.extend(validation.warnings)
                if not validation.ok:
                    target.unlink(missing_ok=True)
                    raise RuntimeError("; ".join(validation.messages))
                record.status = "downloaded-verified"
                size = target.stat().st_size
                speed_mb = size / elapsed / 1e6
                print(
                    f"    {record.variable_id}: {human_size(size)} from "
                    f"{_url_host(url)} in {elapsed:.0f}s ({speed_mb:.2f} MB/s) "
                    f"downloaded-verified"
                )
                return target
            except Exception as exc:
                host = _url_host(url)
                kind = type(exc).__name__
                msg = str(exc)
                if "404" in msg or "Not Found" in msg:
                    mark_host_failed(url)
                errors.append(f"{host} {kind}")
                print(
                    f"    {record.variable_id}: failed {host} "
                    f"attempt {attempt + 1}: {kind}"
                )
                if attempt < retries_per_mirror:
                    time.sleep(min(2**attempt, 10))

    record.status = "failed"
    hosts = ", ".join(errors[:6])
    raise RuntimeError(f"{record.title}: all mirrors failed ({hosts})")


def ensure_records(
    records_by_alias: Mapping[str, Sequence[FileRecord]],
    raw_root: Path,
) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {}
    for alias, records in records_by_alias.items():
        alias_paths = [
            download_record(record, raw_root / record.variable_id) for record in records
        ]
        paths[alias] = alias_paths
    return paths


validate_local_file = validate_local_netcdf
download_file = download_record
ensure_downloaded = ensure_records


def write_manifest(
    path: Path,
    *,
    bundle: ResolvedBundle,
    fixed_records: Mapping[str, FileRecord | None],
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "source_id": bundle.source_id,
        "experiment_id": bundle.experiment_id,
        "member_id": bundle.member_id,
        "grid_label": bundle.grid_label,
        "requested_period": [
            bundle.requested_start_year,
            bundle.requested_end_year,
        ],
        "status": bundle.status,
        "messages": bundle.messages,
        "optional_missing": bundle.optional_missing,
        "time_variables": {
            alias: [asdict(record) for record in records]
            for alias, records in bundle.records_by_alias.items()
        },
        "fixed_fields": {
            alias: None if record is None else asdict(record)
            for alias, record in fixed_records.items()
        },
        "extra": dict(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_manifest(
    path: Path,
) -> tuple[ResolvedBundle, dict[str, FileRecord | None], dict[str, Any]]:
    """Reconstruct a resolved bundle and fixed records from a local manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    period = payload.get("requested_period", [None, None])
    if len(period) != 2 or period[0] is None or period[1] is None:
        raise ValueError(f"Invalid requested_period in {path}")

    record_fields = set(FileRecord.__dataclass_fields__)

    def make_record(data: Mapping[str, Any]) -> FileRecord:
        return FileRecord(**{key: value for key, value in data.items() if key in record_fields})

    time_variables = {
        alias: [make_record(record) for record in records]
        for alias, records in payload.get("time_variables", {}).items()
    }
    fixed_fields = {
        alias: None if record is None else make_record(record)
        for alias, record in payload.get("fixed_fields", {}).items()
    }
    bundle = ResolvedBundle(
        source_id=str(payload.get("source_id", "")),
        experiment_id=str(payload.get("experiment_id", "")),
        member_id=str(payload.get("member_id", "")),
        grid_label=str(payload.get("grid_label", "")),
        requested_start_year=int(period[0]),
        requested_end_year=int(period[1]),
        records_by_alias=time_variables,
        optional_missing=list(payload.get("optional_missing", [])),
        status=str(payload.get("status", "ready")),
        messages=list(payload.get("messages", [])),
    )
    return bundle, fixed_fields, dict(payload.get("extra", {}))


def processing_signature(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def open_time_series(
    paths: Sequence[Path],
    variable_id: str,
    *,
    start_year: int,
    end_year: int,
):
    """Open, subset, load, and concatenate a monthly variable without dask."""

    import xarray as xr

    pieces = []
    for path in sorted(paths):
        with xr.open_dataset(path) as dataset:
            if variable_id not in dataset:
                raise KeyError(f"{variable_id!r} absent from {path}")
            piece = dataset[variable_id].sel(
                time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
            ).load()
            if piece.sizes.get("time", 0):
                pieces.append(piece)
    if not pieces:
        raise ValueError(f"No {variable_id} values in {start_year}-{end_year}")
    combined = xr.concat(pieces, dim="time").sortby("time")
    _, unique_indices = np.unique(combined["time"].values, return_index=True)
    combined = combined.isel(time=np.sort(unique_indices))
    expected_months = {
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    }
    observed_months = {
        (int(year), int(month))
        for year, month in zip(
            combined["time"].dt.year.values,
            combined["time"].dt.month.values,
        )
    }
    if observed_months != expected_months:
        missing = sorted(expected_months - observed_months)
        extra = sorted(observed_months - expected_months)
        raise ValueError(
            f"{variable_id}: incomplete monthly coverage; "
            f"missing={missing[:12]}, unexpected={extra[:12]}"
        )
    return combined


def _normalized_units(units: str) -> str:
    return (
        units.lower()
        .replace("**", "^")
        .replace(" ", "")
        .replace("^-", "-")
        .replace("^", "")
    )


def _time_calendar(data_array) -> str:
    time = data_array["time"]
    calendar = getattr(getattr(time, "dt", None), "calendar", None)
    if calendar:
        return str(calendar)
    encoding = getattr(time, "encoding", {}) or {}
    if encoding.get("calendar"):
        return str(encoding["calendar"])
    attrs = getattr(time, "attrs", {}) or {}
    if attrs.get("calendar"):
        return str(attrs["calendar"])
    return "unknown"


def monthly_flux_to_annual_total(data_array):
    """Integrate a signed monthly water flux to annual mm totals.

    Units are read from metadata, not assumed.  Supported conventions:

    * rate per second (``kg m-2 s-1``, ``mm s-1``) — multiply by
      calendar-aware month length in seconds;
    * rate per day (``kg m-2 day-1``, ``mm day-1``) — multiply by
      calendar-aware ``days_in_month``;
    * already a monthly depth (``kg m-2``, ``mm``) — sum months.

    ``days_in_month`` comes from the CF time coordinate, so ``360_day``,
    ``noleap`` / ``365_day``, and Gregorian calendars are each honoured.
    The sign is preserved (negative ``evspsbl`` is not taken as absolute).
    """

    units = str(data_array.attrs.get("units", "")).strip()
    if not units:
        raise ValueError("Water flux has no units attribute")
    normalized = _normalized_units(units)
    days = data_array["time"].dt.days_in_month.astype("float64")
    values = data_array.astype("float64")
    calendar = _time_calendar(data_array)

    per_second = {
        "kgm-2s-1",
        "kg/m2/s",
        "kgm-2/s",
        "mms-1",
        "mm/s",
    }
    per_day = {
        "kgm-2day-1",
        "kgm-2d-1",
        "kg/m2/day",
        "kg/m2/d",
        "mmday-1",
        "mmd-1",
        "mm/day",
        "mm/d",
    }
    monthly_depth = {
        "kgm-2",
        "kg/m2",
        "mm",
    }

    if normalized in per_second:
        monthly_total = values * days * 86400.0
        how = "rate per second × calendar-aware month length in seconds"
    elif normalized in per_day:
        monthly_total = values * days
        how = "rate per day × calendar-aware days_in_month"
    elif normalized in monthly_depth:
        monthly_total = values
        how = "monthly depth summed over the year"
    else:
        raise ValueError(
            "Unrecognised water-flux units "
            f"{units!r} (normalized={normalized!r}); "
            "expected kg m-2 s-1, mm/day, or monthly mm / kg m-2"
        )

    annual = monthly_total.groupby("time.year").sum("time", skipna=False)
    annual.attrs = dict(data_array.attrs)
    annual.attrs.update(
        {
            "units": "mm yr-1",
            "processing": (
                f"{how}; calendar={calendar}; "
                "native CF time coordinate, not a Gregorian assumption"
            ),
            "calendar": calendar,
            "input_units": units,
            "sign_policy": "original CMIP6 sign preserved",
        }
    )
    return annual


def monthly_state_to_annual_mean(data_array):
    """Calendar-day-weighted annual mean, keeping the original physical units.

    ``days_in_month`` is taken from the CF time coordinate (360_day, noleap,
    Gregorian, …).  ``tas`` remains K unless a later step converts to °C.
    """

    days = data_array["time"].dt.days_in_month.astype("float64")
    calendar = _time_calendar(data_array)
    numerator = (data_array.astype("float64") * days).groupby("time.year").sum(
        "time", skipna=False
    )
    denominator = days.groupby("time.year").sum("time")
    annual = numerator / denominator
    annual.attrs = dict(data_array.attrs)
    annual.attrs["processing"] = (
        "calendar-day-weighted annual mean; "
        f"calendar={calendar}; original units retained"
    )
    annual.attrs["calendar"] = calendar
    return annual


def normalize_fraction_to_percent(data_array, *, name: str):
    """Normalize a fixed surface fraction to percent while preserving metadata."""

    values = data_array.astype("float64")
    units = str(values.attrs.get("units", "")).strip().lower()
    finite = np.asarray(values.values)[np.isfinite(values.values)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if units in {"%", "percent", "percentage"} or maximum > 1.5:
        percent = values
    elif units in {"1", "", "fraction"} and minimum >= -1e-8 and maximum <= 1.0 + 1e-8:
        percent = values * 100.0
    else:
        raise ValueError(
            f"Cannot infer percent/fraction convention for {name}: "
            f"units={units!r}, range=({minimum}, {maximum})"
        )
    finite_percent = np.asarray(percent.values)[np.isfinite(percent.values)]
    if finite_percent.min() < -1e-6 or finite_percent.max() > 100.0 + 1e-6:
        raise ValueError(f"{name} falls outside 0-100% after normalization")
    percent = percent.clip(0.0, 100.0)
    percent.attrs = dict(data_array.attrs)
    percent.attrs["units"] = "%"
    percent.attrs["processing"] = "normalized to percent"
    return percent


def normalize_longitude(data_array, *, convention: str = "0_360"):
    """Normalize and sort a longitude coordinate without spatial interpolation.

    CMIP6 fixed and time-varying fields can carry the same native grid with
    different longitude conventions (for example, -178.125..180 versus
    0..358.125).  Converting coordinate labels before an exact-grid check is
    not regridding and does not alter data values.
    """

    if "lon" not in data_array.coords:
        raise ValueError("DataArray lacks a lon coordinate")
    lon = data_array["lon"].astype("float64")
    if convention == "0_360":
        normalized = np.mod(lon, 360.0)
    elif convention == "-180_180":
        normalized = np.mod(lon + 180.0, 360.0) - 180.0
    else:
        raise ValueError("convention must be '0_360' or '-180_180'")
    result = data_array.assign_coords(lon=normalized).sortby("lon")
    if np.unique(result["lon"].values).size != result["lon"].size:
        raise ValueError("Longitude normalization produced duplicate coordinates")
    return result


def assert_same_horizontal_grid(
    arrays: Mapping[str, Any],
    *,
    atol: float = 1e-10,
    rtol: float = 0.0,
) -> None:
    """Require numerically compatible latitude/longitude coordinates.

    CMIP6 variables on the same native grid can encode nominally identical
    coordinates with tiny floating-point differences across tables.  Shape is
    therefore checked exactly, while coordinate values are compared with a
    small absolute tolerance.  This function only validates; use
    :func:`harmonize_horizontal_grid` to also replace compatible coordinate
    labels with one canonical reference.
    """

    reference_name: str | None = None
    reference = None
    for name, array in arrays.items():
        if "lat" not in array.coords or "lon" not in array.coords:
            raise ValueError(f"{name} lacks lat/lon coordinates")
        if reference is None:
            reference_name, reference = name, array
            continue
        for coordinate, label in (("lat", "Latitude"), ("lon", "Longitude")):
            reference_values = np.asarray(reference[coordinate].values)
            values = np.asarray(array[coordinate].values)
            if reference_values.shape != values.shape:
                raise ValueError(
                    f"{label} shape mismatch: {reference_name} "
                    f"{reference_values.shape} vs {name} {values.shape}"
                )
            if not np.allclose(
                reference_values,
                values,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            ):
                max_difference = float(np.max(np.abs(reference_values - values)))
                raise ValueError(
                    f"{label} mismatch: {reference_name} vs {name}; "
                    f"max_abs_difference={max_difference:.6g}, "
                    f"atol={atol:.6g}, rtol={rtol:.6g}"
                )


def harmonize_horizontal_grid(
    arrays: Mapping[str, Any],
    *,
    reference_name: str | None = None,
    atol: float = 1e-10,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Validate a shared grid and assign one canonical lat/lon coordinate.

    No interpolation or regridding is performed.  Only coordinate labels that
    have already passed the numerical compatibility check are replaced.
    """

    if not arrays:
        raise ValueError("At least one DataArray is required")
    if reference_name is None:
        reference_name = next(iter(arrays))
    if reference_name not in arrays:
        raise KeyError(f"Reference array {reference_name!r} is not present")

    ordered = {reference_name: arrays[reference_name]}
    ordered.update({name: array for name, array in arrays.items() if name != reference_name})
    assert_same_horizontal_grid(ordered, atol=atol, rtol=rtol)

    reference = arrays[reference_name]
    return {
        name: array.assign_coords(lat=reference["lat"], lon=reference["lon"])
        for name, array in arrays.items()
    }


def human_size(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


__all__ = [
    "DEFAULT_SEARCH_ENDPOINTS",
    "DownloadPlan",
    "FileRecord",
    "ResolvedBundle",
    "ValidationResult",
    "assert_same_horizontal_grid",
    "harmonize_horizontal_grid",
    "build_download_plan",
    "covers_period",
    "digest",
    "discover_file_records",
    "download_file",
    "download_record",
    "ensure_downloaded",
    "ensure_records",
    "file_years",
    "human_size",
    "local_cache_status",
    "monthly_flux_to_annual_total",
    "monthly_state_to_annual_mean",
    "normalize_fraction_to_percent",
    "normalize_longitude",
    "open_time_series",
    "pick_fastest_url",
    "processing_signature",
    "query_esgf",
    "read_manifest",
    "records_from_docs",
    "resolve_compatible_bundle",
    "select_fixed_record",
    "validate_local_file",
    "validate_local_netcdf",
    "write_manifest",
]
