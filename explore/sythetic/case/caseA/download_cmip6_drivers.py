#!/usr/bin/env python3
"""Download CMIP6 drivers used by Case A from the official ESGF index.

The selection is intentionally fixed to match the existing Case A water-cycle
data: MPI-ESM1-2-LR, historical, r1i1p1f1, native grid (gn), monthly output.
Only files whose time ranges overlap 1985--2014 are downloaded.

Examples
--------
List matching files without downloading them:
    python download_cmip6_drivers.py --list-only

Download all configured variables and verify their ESGF checksums:
    python download_cmip6_drivers.py

Download a subset:
    python download_cmip6_drivers.py --variables tas rsds rsus rlds rlus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ESGF_SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search/"
SELECTION = {
    "project": "CMIP6",
    "activity_id": "CMIP",
    "institution_id": "MPI-M",
    "source_id": "MPI-ESM1-2-LR",
    "experiment_id": "historical",
    "variant_label": "r1i1p1f1",
    "grid_label": "gn",
}

# CMIP6 monthly tables. snw is a land variable in LImon for this model.
VARIABLE_TABLES = {
    "tas": "Amon",
    "rsds": "Amon",
    "rsus": "Amon",
    "rlds": "Amon",
    "rlus": "Amon",
    "mrso": "Lmon",
    "mrsos": "Lmon",
    "snw": "LImon",
}

DATE_RANGE_RE = re.compile(r"_(\d{4})(?:\d{2,10})?-(\d{4})(?:\d{2,10})?\.nc$")


@dataclass
class FileRecord:
    variable: str
    table: str
    title: str
    start_year: int
    end_year: int
    size: int
    checksum: str
    checksum_type: str
    version: str
    candidate_urls: list[str]
    local_path: str = ""
    status: str = "planned"


def scalar(value, default=""):
    """Return the first item for ESGF fields that may be scalar or list."""
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def file_years(title: str) -> tuple[int, int] | None:
    match = DATE_RANGE_RE.search(title)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def httpserver_urls(doc: dict) -> list[str]:
    urls = []
    for encoded in doc.get("url", []):
        parts = encoded.split("|")
        if len(parts) >= 3 and parts[-1] == "HTTPServer":
            url = parts[0]
            if url.startswith(("https://", "http://")):
                urls.append(url)
    return urls


def url_priority(url: str) -> tuple[int, str]:
    """Prefer HTTPS, then well-established European CMIP mirrors."""
    host_rank = 0 if any(host in url for host in ("dkrz.de", "ipsl.fr", "ceda.ac.uk")) else 1
    scheme_rank = 0 if url.startswith("https://") else 1
    return host_rank + scheme_rank, url


def query_esgf(variable: str, table: str, timeout: int = 180) -> list[dict]:
    params = {
        **SELECTION,
        "variable_id": variable,
        "table_id": table,
        "type": "File",
        "format": "application/solr+json",
        "limit": 1000,
        "latest": "true",
    }
    url = ESGF_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "CaseA-CMIP6-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["response"]["docs"]


def build_records(variable: str, table: str, start_year: int, end_year: int) -> list[FileRecord]:
    docs = query_esgf(variable, table)
    grouped: dict[str, list[dict]] = {}
    for doc in docs:
        title = doc.get("title", "")
        years = file_years(title)
        if not years or years[1] < start_year or years[0] > end_year:
            continue
        if not httpserver_urls(doc):
            continue
        grouped.setdefault(title, []).append(doc)

    records = []
    for title, copies in sorted(grouped.items()):
        years = file_years(title)
        assert years is not None
        urls = sorted({url for doc in copies for url in httpserver_urls(doc)}, key=url_priority)
        reference = max(copies, key=lambda doc: int(doc.get("version", 0) or 0))
        records.append(
            FileRecord(
                variable=variable,
                table=table,
                title=title,
                start_year=years[0],
                end_year=years[1],
                size=int(reference.get("size", 0) or 0),
                checksum=str(scalar(reference.get("checksum"))),
                checksum_type=str(scalar(reference.get("checksum_type"))).upper(),
                version=str(reference.get("version", "")),
                candidate_urls=urls,
            )
        )
    return records


def digest(path: Path, algorithm: str) -> str:
    normalized = algorithm.lower().replace("-", "")
    hasher = hashlib.new(normalized)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_one(record: FileRecord, root: Path, timeout: int = 180) -> None:
    variable_dir = root / record.variable
    variable_dir.mkdir(parents=True, exist_ok=True)
    target = variable_dir / record.title
    partial = target.with_suffix(target.suffix + ".part")
    record.local_path = str(target)

    if target.exists():
        if record.checksum and digest(target, record.checksum_type) == record.checksum.lower():
            record.status = "existing-verified"
            print(f"  verified existing {target.name}")
            return
        print(f"  existing file failed checksum; downloading again: {target.name}")
        target.unlink()

    errors = []
    for url in record.candidate_urls:
        try:
            resume_at = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "CaseA-CMIP6-downloader/1.0"}
            if resume_at:
                headers["Range"] = f"bytes={resume_at}-"
            request = urllib.request.Request(url, headers=headers)
            print(f"  downloading {target.name} from {urllib.parse.urlparse(url).netloc}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = resume_at > 0 and getattr(response, "status", None) == 206
                if resume_at and not append:
                    resume_at = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)

            if record.checksum:
                actual = digest(partial, record.checksum_type)
                if actual != record.checksum.lower():
                    raise RuntimeError(
                        f"{record.checksum_type} mismatch: expected {record.checksum}, got {actual}"
                    )
            partial.replace(target)
            record.status = "downloaded-verified" if record.checksum else "downloaded"
            return
        except Exception as exc:  # try the next ESGF mirror
            errors.append(f"{url}: {exc}")
            print(f"    mirror failed: {exc}", file=sys.stderr)
            if partial.exists():
                partial.unlink()
            time.sleep(1)

    record.status = "failed"
    raise RuntimeError("All ESGF mirrors failed for " + record.title + "\n" + "\n".join(errors))


def human_size(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def write_manifest(path: Path, records: Iterable[FileRecord], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection": SELECTION,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "search_url": ESGF_SEARCH_URL,
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variables", nargs="+", choices=sorted(VARIABLE_TABLES), default=list(VARIABLE_TABLES))
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "drivers" / "raw")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "drivers" / "esgf_manifest.json")
    parser.add_argument("--list-only", action="store_true", help="Query and print the plan without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be no later than --end-year")

    all_records: list[FileRecord] = []
    for variable in args.variables:
        table = VARIABLE_TABLES[variable]
        print(f"Querying ESGF: {variable} ({table}) ...")
        records = build_records(variable, table, args.start_year, args.end_year)
        if not records:
            print(f"  WARNING: no matching files found for {variable} ({table})", file=sys.stderr)
        all_records.extend(records)

    total = sum(record.size for record in all_records)
    print(f"\nPlan: {len(all_records)} files, approximately {human_size(total)}")
    for record in all_records:
        print(f"  {record.variable:5s} {record.start_year}-{record.end_year} {human_size(record.size):>10s}  {record.title}")
    write_manifest(args.manifest, all_records, args)

    if args.list_only:
        print(f"\nManifest written to {args.manifest}")
        return 0

    for index, record in enumerate(all_records, start=1):
        print(f"\n[{index}/{len(all_records)}] {record.variable}: {record.start_year}-{record.end_year}")
        download_one(record, args.output)
        write_manifest(args.manifest, all_records, args)

    failures = [record for record in all_records if record.status == "failed"]
    print(f"\nFinished. Manifest: {args.manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
