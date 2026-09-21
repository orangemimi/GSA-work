from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


SNR_LEVELS = [
    0.05,
    0.08,
    0.1,
    0.13,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    1.0,
    1.2,
    1.5,
    1.8,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    15.0,
    18.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    80.0,
    100.0,
    130.0,
    170.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
    math.inf,
]

INTERFERENCE_SNR_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, math.inf]
HETEROSCEDASTIC_NOISE_STRUCTURES = ["increasing", "decreasing", "middle_high"]
SAMPLING_DISTRIBUTIONS = ["beta_2_5", "beta_5_2", "beta_5_5", "two_clusters", "three_clusters"]

DEFAULT_N = 500
DEFAULT_R = 100
VAR_GRID_SIZE = 10_000
LABEL_GRID_SIZE = 4_001
EPS = 1e-12

_SIGNAL_VARIANCE_CACHE: dict[tuple[str, int, str], float] = {}
_CLEAN_RANGE_CACHE: dict[tuple[str, int, str], float] = {}
_BASE_LABEL_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class FunctionSpec:
    function_type: str
    function_name: str
    config_index: int
    formula: str
    params: dict[str, Any]
    fn: Callable[[np.ndarray], np.ndarray]

    @property
    def key(self) -> tuple[str, int]:
        return self.function_type, self.config_index


def _json_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, np.bool_)):
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    return str(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return _json_default(value)


def dumps_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _safe_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmax(values) - np.nanmin(values))


def _reference_grid(size: int = VAR_GRID_SIZE) -> np.ndarray:
    return np.linspace(0.0, 1.0, size)


def _spec_cache_key(spec: FunctionSpec) -> tuple[str, int, str]:
    return spec.function_type, spec.config_index, dumps_json(spec.params)


def _normalized_fn(
    raw_fn: Callable[[np.ndarray], np.ndarray],
    *,
    amplitude: float = 1.0,
    center: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    ref = raw_fn(_reference_grid())
    low = float(np.nanmin(ref))
    high = float(np.nanmax(ref))
    span = high - low

    def fn(x: np.ndarray) -> np.ndarray:
        raw = raw_fn(np.asarray(x, dtype=float))
        if span <= EPS:
            scaled = np.zeros_like(raw, dtype=float)
        else:
            scaled = (raw - low) / span
        if center:
            scaled = scaled - 0.5
        return amplitude * scaled

    return fn


def _add_spec(
    specs: list[FunctionSpec],
    counts: dict[str, int],
    function_type: str,
    function_name: str,
    formula: str,
    params: dict[str, Any],
    fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    counts[function_type] = counts.get(function_type, 0) + 1
    specs.append(
        FunctionSpec(
            function_type=function_type,
            function_name=function_name,
            config_index=counts[function_type],
            formula=formula,
            params={k: _json_default(v) for k, v in params.items()},
            fn=fn,
        )
    )


def build_function_specs() -> list[FunctionSpec]:
    """Build the 420 function configurations described in the workflow document."""
    specs: list[FunctionSpec] = []
    counts: dict[str, int] = {}

    for constant in np.linspace(0.05, 0.95, 20):
        value = float(round(constant, 6))
        _add_spec(
            specs,
            counts,
            "F00",
            "Random",
            "f(x) = constant",
            {"constant": value, "null_variance_reference": 1.0},
            lambda x, value=value: np.full_like(np.asarray(x, dtype=float), value, dtype=float),
        )

    for slope in [-5, -3, -2, -1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1, 2, 3, 5, 0.01, 0.05, 10, 15, 20, 50]:
        _add_spec(
            specs,
            counts,
            "F01",
            "Linear",
            "f(x) = a*x",
            {"a": float(slope)},
            lambda x, slope=float(slope): slope * np.asarray(x, dtype=float),
        )

    for p in [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        _add_spec(
            specs,
            counts,
            "F02",
            "Power-Convex",
            "f(x) = x**p",
            {"p": float(p)},
            lambda x, p=float(p): np.asarray(x, dtype=float) ** p,
        )

    for p in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02]:
        _add_spec(
            specs,
            counts,
            "F03",
            "Power-Concave",
            "f(x) = x**p",
            {"p": float(p)},
            lambda x, p=float(p): np.asarray(x, dtype=float) ** p,
        )

    for k in [0.3, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 40]:
        _add_spec(
            specs,
            counts,
            "F04",
            "Saturation",
            "f(x) = 1 - exp(-k*x)",
            {"k": float(k)},
            lambda x, k=float(k): 1.0 - np.exp(-k * np.asarray(x, dtype=float)),
        )

    for a in [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000]:
        _add_spec(
            specs,
            counts,
            "F05",
            "Log",
            "f(x) = log(1 + a*x)",
            {"a": float(a)},
            lambda x, a=float(a): np.log1p(a * np.asarray(x, dtype=float)),
        )

    for b in [0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 15]:
        _add_spec(
            specs,
            counts,
            "F06",
            "Exponential",
            "f(x) = exp(b*x) - 1",
            {"b": float(b)},
            lambda x, b=float(b): np.exp(b * np.asarray(x, dtype=float)) - 1.0,
        )

    s_curve_configs = [
        (3, 0.5),
        (5, 0.3),
        (5, 0.5),
        (5, 0.7),
        (8, 0.3),
        (8, 0.5),
        (8, 0.7),
        (12, 0.3),
        (12, 0.5),
        (12, 0.7),
        (20, 0.3),
        (20, 0.5),
        (20, 0.7),
        (30, 0.5),
        (50, 0.3),
        (50, 0.5),
        (50, 0.7),
        (80, 0.5),
        (120, 0.5),
        (200, 0.5),
    ]
    for k, c in s_curve_configs:
        _add_spec(
            specs,
            counts,
            "F07",
            "S-curve",
            "f(x) = 1/(1 + exp(-k*(x-c)))",
            {"k": float(k), "c": float(c)},
            lambda x, k=float(k), c=float(c): 1.0 / (1.0 + np.exp(-k * (np.asarray(x, dtype=float) - c))),
        )

    for gap in [0.5, 1, 2, 5]:
        for delta in [0.005, 0.01, 0.03, 0.05, 0.1]:
            c = 0.5
            a = 0.0
            b = float(gap)

            def threshold_fn(x: np.ndarray, a: float = a, b: float = b, c: float = c, delta: float = float(delta)) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                left = c - delta
                right = c + delta
                transition = a + (b - a) / (2.0 * delta) * (x - c + delta)
                return np.where(x < left, a, np.where(x > right, b, transition))

            _add_spec(
                specs,
                counts,
                "F08",
                "Threshold",
                "piecewise stable A -> transition -> stable B",
                {"gap": float(gap), "delta": float(delta), "c": c},
                threshold_fn,
            )

    for a in [1, 2, 4, 8]:
        for c in [0.2, 0.35, 0.5, 0.65, 0.8]:
            raw = lambda x, a=float(a), c=float(c): -a * (np.asarray(x, dtype=float) - c) ** 2
            _add_spec(
                specs,
                counts,
                "F09",
                "Quadratic Peak",
                "f(x) = -a*(x-c)**2 + d",
                {"a": float(a), "c": float(c)},
                _normalized_fn(raw, center=False),
            )

    for a in [1, 2, 4, 8]:
        for c in [0.2, 0.35, 0.5, 0.65, 0.8]:
            raw = lambda x, a=float(a), c=float(c): a * (np.asarray(x, dtype=float) - c) ** 2
            _add_spec(
                specs,
                counts,
                "F10",
                "Quadratic Valley",
                "f(x) = a*(x-c)**2 + d",
                {"a": float(a), "c": float(c)},
                _normalized_fn(raw, center=False),
            )

    for width in [0.01, 0.02, 0.05, 0.1, 0.2]:
        for height in [0.5, 1, 2, 5]:
            _add_spec(
                specs,
                counts,
                "F11",
                "Spike",
                "f(x) = h*max(1 - abs(x-c)/(w/2), 0)",
                {"w": float(width), "h": float(height), "c": 0.5},
                lambda x, width=float(width), height=float(height): height
                * np.maximum(1.0 - np.abs(np.asarray(x, dtype=float) - 0.5) / (width / 2.0), 0.0),
            )

    for width in [0.01, 0.02, 0.05, 0.1, 0.2]:
        for height in [0.5, 1, 2, 5]:
            _add_spec(
                specs,
                counts,
                "F12",
                "L-shaped Valley",
                "f(x) = -h*max(1 - abs(x-c)/(w/2), 0)",
                {"w": float(width), "h": float(height), "c": 0.5},
                lambda x, width=float(width), height=float(height): -height
                * np.maximum(1.0 - np.abs(np.asarray(x, dtype=float) - 0.5) / (width / 2.0), 0.0),
            )

    cubic_root_configs: list[tuple[float, float, float]] = []
    for span in [0.3, 0.5, 0.7]:
        for center_offset in [-0.1, 0.0, 0.1]:
            cubic_root_configs.append((span, center_offset, 1.0))
    cubic_root_configs.append((0.9, 0.0, 1.0))
    cubic_root_configs = cubic_root_configs + [(span, offset, 0.6) for span, offset, _ in cubic_root_configs]

    for span, center_offset, amplitude in cubic_root_configs:
        center = 0.5 + center_offset
        r1 = center - span / 2.0
        r2 = center
        r3 = center + span / 2.0
        raw = lambda x, r1=float(r1), r2=float(r2), r3=float(r3): (
            np.asarray(x, dtype=float) - r1
        ) * (np.asarray(x, dtype=float) - r2) * (np.asarray(x, dtype=float) - r3)
        _add_spec(
            specs,
            counts,
            "F13",
            "Cubic M",
            "f(x) = a*(x-r1)*(x-r2)*(x-r3)",
            {"span": float(span), "center_offset": float(center_offset), "amplitude": float(amplitude), "r1": r1, "r2": r2, "r3": r3},
            _normalized_fn(raw, amplitude=float(amplitude), center=True),
        )

    for span, center_offset, amplitude in cubic_root_configs:
        center = 0.5 + center_offset
        r1 = center - span / 2.0
        r2 = center
        r3 = center + span / 2.0
        raw = lambda x, r1=float(r1), r2=float(r2), r3=float(r3): -(
            (np.asarray(x, dtype=float) - r1)
            * (np.asarray(x, dtype=float) - r2)
            * (np.asarray(x, dtype=float) - r3)
        )
        _add_spec(
            specs,
            counts,
            "F14",
            "Cubic W",
            "f(x) = -a*(x-r1)*(x-r2)*(x-r3)",
            {"span": float(span), "center_offset": float(center_offset), "amplitude": float(amplitude), "r1": r1, "r2": r2, "r3": r3},
            _normalized_fn(raw, amplitude=float(amplitude), center=True),
        )

    ratios = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
    gaussian_configs = [(distance, ratio, sigma) for distance in [0.2, 0.3, 0.4, 0.6] for ratio in ratios for sigma in [0.05, 0.1]][:20]
    for distance, (a1, a2), sigma in gaussian_configs:
        mu1 = 0.5 - distance / 2.0
        mu2 = 0.5 + distance / 2.0
        _add_spec(
            specs,
            counts,
            "F15",
            "Double Gaussian",
            "f(x) = A1*exp(-(x-mu1)**2/(2*sigma**2)) + A2*exp(-(x-mu2)**2/(2*sigma**2))",
            {"distance": float(distance), "A1": float(a1), "A2": float(a2), "sigma": float(sigma), "mu1": mu1, "mu2": mu2},
            lambda x, mu1=float(mu1), mu2=float(mu2), a1=float(a1), a2=float(a2), sigma=float(sigma): a1
            * np.exp(-((np.asarray(x, dtype=float) - mu1) ** 2) / (2.0 * sigma**2))
            + a2 * np.exp(-((np.asarray(x, dtype=float) - mu2) ** 2) / (2.0 * sigma**2)),
        )

    for omega_pi in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for amplitude in [0.3, 1.0]:
            _add_spec(
                specs,
                counts,
                "F16",
                "Pure Oscillation",
                "f(x) = A*sin(omega*x)",
                {"omega_pi": float(omega_pi), "A": float(amplitude)},
                lambda x, omega_pi=float(omega_pi), amplitude=float(amplitude): amplitude * np.sin(omega_pi * math.pi * np.asarray(x, dtype=float)),
            )

    for omega_pi in [2, 4, 6, 8, 12]:
        for slope in [-1, -0.3, 0.3, 1]:
            amplitude = 0.3 * abs(float(slope))
            _add_spec(
                specs,
                counts,
                "F17",
                "Oscillation + Trend",
                "f(x) = b*x + A*sin(omega*x)",
                {"omega_pi": float(omega_pi), "b": float(slope), "A": float(amplitude)},
                lambda x, omega_pi=float(omega_pi), slope=float(slope), amplitude=float(amplitude): slope
                * np.asarray(x, dtype=float)
                + amplitude * np.sin(omega_pi * math.pi * np.asarray(x, dtype=float)),
            )

    for omega_pi in [4, 6, 8, 12, 16]:
        for damping in [0.5, 1, 3, 5]:
            _add_spec(
                specs,
                counts,
                "F18",
                "Damped Oscillation",
                "f(x) = A*exp(-lambda*x)*sin(omega*x)",
                {"omega_pi": float(omega_pi), "lambda": float(damping), "A": 1.0},
                lambda x, omega_pi=float(omega_pi), damping=float(damping): np.exp(-damping * np.asarray(x, dtype=float))
                * np.sin(omega_pi * math.pi * np.asarray(x, dtype=float)),
            )

    for omega_pi in [4, 6, 8, 12, 16]:
        for growth in [0.3, 0.5, 1, 2]:
            _add_spec(
                specs,
                counts,
                "F19",
                "Growing Oscillation",
                "f(x) = A*exp(lambda*x)*sin(omega*x)",
                {"omega_pi": float(omega_pi), "lambda": float(growth), "A": 1.0},
                lambda x, omega_pi=float(omega_pi), growth=float(growth): np.exp(growth * np.asarray(x, dtype=float))
                * np.sin(omega_pi * math.pi * np.asarray(x, dtype=float)),
            )

    for omega_pi in [3, 5, 7, 10, 14]:
        for alpha in [0.5, 1, 2, 3]:
            _add_spec(
                specs,
                counts,
                "F20",
                "Varying Frequency",
                "f(x) = A*sin(omega*x*(1 + alpha*x))",
                {"omega_pi": float(omega_pi), "alpha": float(alpha), "A": 1.0},
                lambda x, omega_pi=float(omega_pi), alpha=float(alpha): np.sin(
                    omega_pi * math.pi * np.asarray(x, dtype=float) * (1.0 + alpha * np.asarray(x, dtype=float))
                ),
            )

    if len(specs) != 420:
        raise ValueError(f"Expected 420 function specs, got {len(specs)}")
    counts_by_family = {family: 0 for family in [f"F{i:02d}" for i in range(21)]}
    for spec in specs:
        counts_by_family[spec.function_type] += 1
    bad_counts = {k: v for k, v in counts_by_family.items() if v != 20}
    if bad_counts:
        raise ValueError(f"Each family must have 20 configs, bad counts: {bad_counts}")
    return specs


def get_spec_lookup(specs: Iterable[FunctionSpec] | None = None) -> dict[tuple[str, int], FunctionSpec]:
    if specs is None:
        specs = build_function_specs()
    return {spec.key: spec for spec in specs}


def snr_token(snr: float) -> str:
    if math.isinf(float(snr)):
        return "inf"
    text = f"{float(snr):g}"
    return text.replace("-", "m").replace(".", "p")


def _case_id(
    experiment: str,
    spec: FunctionSpec,
    snr: float,
    noise_structure: str,
    x_distribution: str,
) -> str:
    return (
        f"{experiment}_{spec.function_type}_cfg{spec.config_index:02d}"
        f"_snr{snr_token(snr)}_{noise_structure}_{x_distribution}"
    )


def stable_hash_int(text: str, modulo: int = 1_000_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def signal_variance(spec: FunctionSpec) -> float:
    cache_key = _spec_cache_key(spec)
    if cache_key in _SIGNAL_VARIANCE_CACHE:
        return _SIGNAL_VARIANCE_CACHE[cache_key]
    grid = _reference_grid()
    values = spec.fn(grid)
    variance = float(np.nanvar(values))
    if variance <= EPS and spec.function_type == "F00":
        variance = float(spec.params.get("null_variance_reference", 1.0))
    _SIGNAL_VARIANCE_CACHE[cache_key] = variance
    return variance


def clean_range(spec: FunctionSpec) -> float:
    cache_key = _spec_cache_key(spec)
    if cache_key in _CLEAN_RANGE_CACHE:
        return _CLEAN_RANGE_CACHE[cache_key]
    grid = _reference_grid()
    span = _safe_range(spec.fn(grid))
    if span <= EPS and spec.function_type == "F00":
        span = 1.0
    _CLEAN_RANGE_CACHE[cache_key] = span
    return span


def compute_sigma_epsilon(spec: FunctionSpec, snr: float) -> float:
    snr = float(snr)
    if math.isinf(snr):
        return 0.0
    if snr <= 0:
        raise ValueError("SNR must be positive or math.inf")
    return math.sqrt(signal_variance(spec) / snr)


def noise_multiplier(x: np.ndarray, noise_structure: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if noise_structure == "constant":
        return np.ones_like(x, dtype=float)
    if noise_structure == "increasing":
        return 0.2 + 1.6 * x
    if noise_structure == "decreasing":
        return 1.8 - 1.6 * x
    if noise_structure == "middle_high":
        return 0.3 + 1.4 * np.exp(-((x - 0.5) ** 2) / 0.02)
    raise ValueError(f"Unknown noise_structure: {noise_structure}")


@lru_cache(maxsize=None)
def average_noise_multiplier(noise_structure: str) -> float:
    return float(np.mean(noise_multiplier(_reference_grid(), noise_structure)))


def sample_x(rng: np.random.Generator, n: int, x_distribution: str) -> np.ndarray:
    if x_distribution == "uniform":
        return rng.uniform(0.0, 1.0, n)
    if x_distribution == "beta_2_5":
        return rng.beta(2.0, 5.0, n)
    if x_distribution == "beta_5_2":
        return rng.beta(5.0, 2.0, n)
    if x_distribution == "beta_5_5":
        return rng.beta(5.0, 5.0, n)
    if x_distribution == "two_clusters":
        choose_right = rng.random(n) >= 0.5
        x = np.empty(n, dtype=float)
        x[~choose_right] = rng.uniform(0.0, 0.3, np.sum(~choose_right))
        x[choose_right] = rng.uniform(0.7, 1.0, np.sum(choose_right))
        return x
    if x_distribution == "three_clusters":
        components = rng.integers(0, 3, n)
        x = np.empty(n, dtype=float)
        mask = components == 0
        x[mask] = rng.uniform(0.0, 0.15, np.sum(mask))
        mask = components == 1
        x[mask] = rng.uniform(0.4, 0.6, np.sum(mask))
        mask = components == 2
        x[mask] = rng.uniform(0.85, 1.0, np.sum(mask))
        return x
    raise ValueError(f"Unknown x_distribution: {x_distribution}")


def strength_label(snr: float) -> str:
    snr = float(snr)
    if math.isinf(snr) or snr > 10.0:
        return "Strong"
    if snr < 0.3:
        return "Very Weak"
    if snr <= 1.0:
        return "Weak"
    return "Medium"


def _linearity_error(spec: FunctionSpec) -> float | None:
    grid = np.linspace(0.0, 1.0, LABEL_GRID_SIZE)
    y = spec.fn(grid)
    span = _safe_range(y)
    if span <= EPS:
        return None
    slope, intercept = np.polyfit(grid, y, 1)
    fit = slope * grid + intercept
    return float(np.nanmax(np.abs(y - fit)) / span)


def _direction_label(spec: FunctionSpec) -> str | None:
    grid = np.linspace(0.0, 1.0, LABEL_GRID_SIZE)
    y = spec.fn(grid)
    span = _safe_range(y)
    if span <= EPS:
        return None
    diff = float(y[-1] - y[0])
    if abs(diff) / span < 0.05:
        return None
    return "Positive" if diff > 0 else "Negative"


def _late_early_slope_ratio(spec: FunctionSpec) -> float | None:
    grid = np.linspace(0.0, 1.0, LABEL_GRID_SIZE)
    y = spec.fn(grid)
    span = _safe_range(y)
    if span <= EPS:
        return None
    dy = np.gradient(y, grid)
    window = max(5, int(0.1 * LABEL_GRID_SIZE))
    early = float(np.nanmean(np.abs(dy[:window])))
    late = float(np.nanmean(np.abs(dy[-window:])))
    if early <= EPS:
        if late <= EPS:
            return 1.0
        return math.inf
    return late / early


def _saturation_label(spec: FunctionSpec) -> tuple[str, float | None]:
    if spec.function_type in {"F00", "F08", "F09", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20"}:
        return "No", _late_early_slope_ratio(spec)
    ratio = _late_early_slope_ratio(spec)
    if ratio is None:
        return "No", None
    if ratio < 0.1:
        return "Strong", ratio
    if ratio < 0.3:
        return "Weak", ratio
    return "No", ratio


def _base_shape_labels(spec: FunctionSpec) -> dict[str, Any]:
    cache_key = _spec_cache_key(spec)
    if cache_key in _BASE_LABEL_CACHE:
        return dict(_BASE_LABEL_CACHE[cache_key])

    family = spec.function_type
    linearity_error = _linearity_error(spec)
    linearity = None if linearity_error is None else ("Linear" if linearity_error < 0.05 else "Nonlinear")
    direction = _direction_label(spec)
    saturation, slope_ratio = _saturation_label(spec)

    labels: dict[str, Any] = {
        "direction": direction,
        "monotonicity": None,
        "linearity": linearity,
        "linearity_error": linearity_error,
        "convexity": None,
        "saturation": saturation,
        "late_slope_early_slope_ratio": slope_ratio,
        "tp_count": 0,
        "tp_types": None,
        "tp_pattern": None,
        "tp_trend": None,
        "s_curve": False,
        "threshold": False,
    }

    if family == "F00":
        labels.update({"linearity": None, "linearity_error": None, "saturation": "No", "tp_count": 0})
    elif family == "F01":
        labels.update({"monotonicity": "Monotonic", "convexity": None, "tp_count": 0, "saturation": "No"})
    elif family == "F02":
        labels.update({"monotonicity": "Monotonic", "convexity": "Convex", "tp_count": 0})
    elif family in {"F03", "F04", "F05"}:
        labels.update({"monotonicity": "Monotonic", "convexity": "Concave", "tp_count": 0})
    elif family == "F06":
        labels.update({"monotonicity": "Monotonic", "convexity": "Convex", "tp_count": 0})
    elif family == "F07":
        labels.update({"monotonicity": "Monotonic", "convexity": "Mixed", "tp_count": 0, "s_curve": True})
    elif family == "F08":
        labels.update(
            {
                "direction": "Positive",
                "monotonicity": "Monotonic",
                "convexity": "Piecewise",
                "linearity": "Nonlinear",
                "tp_count": 0,
                "threshold": True,
                "saturation": "No",
            }
        )
    elif family == "F09":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 1,
                "tp_types": ["peak"],
                "tp_pattern": "peak",
            }
        )
    elif family == "F10":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 1,
                "tp_types": ["valley"],
                "tp_pattern": "valley",
            }
        )
    elif family == "F11":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 1,
                "tp_types": ["peak"],
                "tp_pattern": "peak",
            }
        )
    elif family == "F12":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 1,
                "tp_types": ["valley"],
                "tp_pattern": "valley",
            }
        )
    elif family == "F13":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 2,
                "tp_types": ["peak", "valley"],
                "tp_pattern": "peak-valley",
            }
        )
    elif family == "F14":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 2,
                "tp_types": ["valley", "peak"],
                "tp_pattern": "valley-peak",
            }
        )
    elif family == "F15":
        labels.update(
            {
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": 2,
                "tp_types": ["peak", "peak"],
                "tp_pattern": "two-peaks",
            }
        )
    elif family == "F16":
        omega_pi = float(spec.params["omega_pi"])
        count = int(round(omega_pi - 1.0))
        labels.update(
            {
                "direction": None,
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": count if count <= 2 else "multiple",
                "tp_types": None,
                "tp_pattern": "alternating",
                "tp_trend": "none",
            }
        )
    elif family == "F17":
        slope = float(spec.params["b"])
        labels.update(
            {
                "direction": "Positive" if slope > 0 else "Negative",
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": "multiple",
                "tp_pattern": "alternating",
                "tp_trend": "positive" if slope > 0 else "negative",
            }
        )
    elif family == "F18":
        labels.update(
            {
                "direction": None,
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": "multiple",
                "tp_pattern": "alternating",
                "tp_trend": "damped",
            }
        )
    elif family == "F19":
        labels.update(
            {
                "direction": None,
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": "multiple",
                "tp_pattern": "alternating",
                "tp_trend": "growing",
            }
        )
    elif family == "F20":
        labels.update(
            {
                "direction": None,
                "monotonicity": "Non-monotonic",
                "convexity": None,
                "tp_count": "multiple",
                "tp_pattern": "alternating",
                "tp_trend": "frequency-increasing",
            }
        )
    else:
        raise ValueError(f"Unknown function family: {family}")

    _BASE_LABEL_CACHE[cache_key] = dict(labels)
    return labels


def derive_labels(spec: FunctionSpec, snr: float, noise_structure: str, x_distribution: str) -> dict[str, Any]:
    labels = _base_shape_labels(spec)
    sigma_epsilon = compute_sigma_epsilon(spec, snr)
    spread_denominator = clean_range(spec)
    high_spread_ratio = (
        math.inf if spread_denominator <= EPS and sigma_epsilon > 0 else sigma_epsilon * average_noise_multiplier(noise_structure) / spread_denominator
    )
    labels.update(
        {
            "strength": strength_label(snr),
            "high_spread": bool(high_spread_ratio > 0.3),
            "high_spread_ratio": high_spread_ratio,
            "changing_spread": noise_structure != "constant",
            "density": "Even" if x_distribution == "uniform" else "Uneven",
            "clusters": x_distribution in {"two_clusters", "three_clusters"},
        }
    )
    return labels


def _typical_specs(specs: Iterable[FunctionSpec]) -> list[FunctionSpec]:
    lookup = {(spec.function_type, dumps_json(spec.params)): spec for spec in specs}

    def find(function_type: str, **params: Any) -> FunctionSpec:
        expected = dumps_json(params)
        for spec in specs:
            if spec.function_type == function_type and dumps_json(spec.params) == expected:
                return spec
        raise KeyError(f"Could not find {function_type} with params {params}")

    del lookup
    return [
        find("F01", a=1.0),
        find("F04", k=10.0),
        find("F07", k=20.0, c=0.5),
        find("F08", gap=1.0, delta=0.01, c=0.5),
        find("F09", a=4.0, c=0.5),
        find("F13", span=0.5, center_offset=0.0, amplitude=1.0, r1=0.25, r2=0.5, r3=0.75),
        find("F17", omega_pi=4.0, b=1.0, A=0.3),
        next(spec for spec in specs if spec.function_type == "F00" and spec.config_index == 10),
    ]


def build_case_table(
    specs: Iterable[FunctionSpec] | None = None,
    *,
    n: int = DEFAULT_N,
    R: int = DEFAULT_R,
    include_interference: bool = True,
) -> pd.DataFrame:
    if specs is None:
        specs = build_function_specs()
    specs = list(specs)
    rows: list[dict[str, Any]] = []

    def add_case(experiment: str, spec: FunctionSpec, snr: float, noise_structure: str, x_distribution: str) -> None:
        sigma = compute_sigma_epsilon(spec, snr)
        labels = derive_labels(spec, snr, noise_structure, x_distribution)
        case_id = _case_id(experiment, spec, snr, noise_structure, x_distribution)
        row = {
            "case_index": len(rows) + 1,
            "case_id": case_id,
            "base_seed": (len(rows) + 1) * 1000,
            "experiment": experiment,
            "function_type": spec.function_type,
            "function_name": spec.function_name,
            "config_index": spec.config_index,
            "function_formula": spec.formula,
            "shape_params": dumps_json(spec.params),
            "snr": float(snr),
            "snr_label": snr_token(snr),
            "sigma_epsilon": sigma,
            "noise_structure": noise_structure,
            "x_distribution": x_distribution,
            "n": int(n),
            "R": int(R),
        }
        row.update(labels)
        rows.append(row)

    for spec in specs:
        for snr in SNR_LEVELS:
            add_case("main", spec, snr, "constant", "uniform")

    if include_interference:
        typical = _typical_specs(specs)
        for spec in typical:
            for snr in INTERFERENCE_SNR_LEVELS:
                for noise_structure in HETEROSCEDASTIC_NOISE_STRUCTURES:
                    add_case("heteroscedasticity", spec, snr, noise_structure, "uniform")

        for spec in typical:
            for snr in INTERFERENCE_SNR_LEVELS:
                for x_distribution in SAMPLING_DISTRIBUTIONS:
                    add_case("sampling", spec, snr, "constant", x_distribution)

    df = pd.DataFrame(rows)
    expected = 21_640 if include_interference else 21_000
    if len(df) != expected:
        raise ValueError(f"Expected {expected} cases, got {len(df)}")
    return df


def _row_value(case: pd.Series | dict[str, Any], key: str) -> Any:
    if isinstance(case, pd.Series):
        return case[key]
    return case[key]


def _spec_for_case(case: pd.Series | dict[str, Any], specs: Iterable[FunctionSpec] | None = None) -> FunctionSpec:
    lookup = get_spec_lookup(specs)
    return lookup[(str(_row_value(case, "function_type")), int(_row_value(case, "config_index")))]


def generate_case_data(
    case: pd.Series | dict[str, Any],
    specs: Iterable[FunctionSpec] | None = None,
    *,
    n: int | None = None,
    R: int | None = None,
    dtype: np.dtype | str = np.float32,
) -> dict[str, Any]:
    spec = _spec_for_case(case, specs)
    n = int(n if n is not None else _row_value(case, "n"))
    R = int(R if R is not None else _row_value(case, "R"))
    base_seed = int(_row_value(case, "base_seed"))
    sigma0 = float(_row_value(case, "sigma_epsilon"))
    noise_structure = str(_row_value(case, "noise_structure"))
    x_distribution = str(_row_value(case, "x_distribution"))

    realizations = np.empty((R, n, 2), dtype=dtype)
    f_x_clean = np.empty((R, n), dtype=dtype)
    sigma_x = np.empty((R, n), dtype=dtype)

    for r in range(R):
        rng = np.random.default_rng(base_seed + r + 1)
        x = sample_x(rng, n, x_distribution)
        clean = spec.fn(x)
        local_sigma = sigma0 * noise_multiplier(x, noise_structure)
        noise = rng.normal(0.0, local_sigma)
        y = clean + noise
        realizations[r, :, 0] = x.astype(dtype, copy=False)
        realizations[r, :, 1] = y.astype(dtype, copy=False)
        f_x_clean[r] = clean.astype(dtype, copy=False)
        sigma_x[r] = local_sigma.astype(dtype, copy=False)

    metadata = {key: _json_default(value) for key, value in dict(case).items()}
    if isinstance(metadata.get("shape_params"), str):
        metadata["shape_params"] = json.loads(metadata["shape_params"])
    metadata["n"] = n
    metadata["R"] = R
    return {
        "case_metadata": metadata,
        "ground_truth_labels": {
            key: metadata[key]
            for key in [
                "direction",
                "monotonicity",
                "linearity",
                "convexity",
                "saturation",
                "late_slope_early_slope_ratio",
                "tp_count",
                "tp_types",
                "tp_pattern",
                "tp_trend",
                "s_curve",
                "threshold",
                "strength",
                "high_spread",
                "changing_spread",
                "density",
                "clusters",
            ]
        },
        "data": {
            "realizations": realizations,
            "f_x_clean": f_x_clean,
            "sigma_x": sigma_x,
        },
    }


def write_case_npz(
    case: pd.Series | dict[str, Any],
    output_dir: str | Path,
    specs: Iterable[FunctionSpec] | None = None,
    *,
    n: int | None = None,
    R: int | None = None,
    dtype: np.dtype | str = np.float32,
    compressed: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_case_data(case, specs, n=n, R=R, dtype=dtype)
    case_id = str(_row_value(case, "case_id"))
    path = output_dir / f"{case_id}.npz"
    save = np.savez_compressed if compressed else np.savez
    save(
        path,
        realizations=generated["data"]["realizations"],
        f_x_clean=generated["data"]["f_x_clean"],
        sigma_x=generated["data"]["sigma_x"],
        metadata_json=dumps_json(generated["case_metadata"]),
        labels_json=dumps_json(generated["ground_truth_labels"]),
    )
    return path


def export_case_metadata(case_table: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "case_metadata.csv"
    jsonl_path = output_dir / "case_metadata.jsonl"
    case_table.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in case_table.to_dict(orient="records"):
            if isinstance(record.get("shape_params"), str):
                record["shape_params"] = json.loads(record["shape_params"])
            handle.write(dumps_json(record) + "\n")
    return {"csv": csv_path, "jsonl": jsonl_path}


def write_dataset_cases(
    case_table: pd.DataFrame,
    output_dir: str | Path,
    specs: Iterable[FunctionSpec] | None = None,
    *,
    limit: int | None = None,
    dtype: np.dtype | str = np.float32,
    compressed: bool = True,
) -> list[Path]:
    """Write one .npz file per case.

    The full design is very large, so callers should use limit or filter the
    case table unless they intentionally want to materialize the full dataset.
    """
    specs = list(specs) if specs is not None else build_function_specs()
    paths: list[Path] = []
    iterable = case_table.head(limit).iterrows() if limit is not None else case_table.iterrows()
    for _, row in iterable:
        paths.append(write_case_npz(row, output_dir, specs, dtype=dtype, compressed=compressed))
    return paths
