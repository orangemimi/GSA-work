from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("case/caseA").resolve()))
from branch_benchmark import detect_multires_hexbin

ROOT = Path("/Volumes/mimi-T9/CMIP6")
MODELS = ["CESM2", "CNRM-CM6-1", "CanESM5", "GFDL-CM4", "CMCC-CM2-SR5"]
DOMAINS = ["all_land", "WW", "WD", "CW", "CD", "LI"]
PAIRS = [("P", "Q", "P → Q"), ("mrros", "Q", "mrros → Q"), ("hfls", "Q", "hfls → Q")]


def clipped(x, y, q):
    if q == 0:
        return x, y
    xlo, xhi = np.quantile(x, [q, 1 - q])
    ylo, yhi = np.quantile(y, [q, 1 - q])
    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    return x[keep], y[keep]


rows = []
for model in MODELS:
    path = next((ROOT / model / "historical").glob("*/*/land/zones/zone_climatology_1985_2014.parquet"))
    frame = pd.read_parquet(path).rename(columns={"R": "Q"})
    for xvar, yvar, pair in PAIRS:
        all_finite = np.isfinite(frame[xvar]) & np.isfinite(frame[yvar])
        gxlo, gxhi = np.quantile(frame.loc[all_finite, xvar], [.01, .99])
        gylo, gyhi = np.quantile(frame.loc[all_finite, yvar], [.01, .99])
        for domain in DOMAINS:
            sub = frame if domain == "all_land" else frame.loc[frame.analysis_zone == domain]
            x = sub[xvar].to_numpy(float)
            y = sub[yvar].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            trim = finite & (x >= gxlo) & (x <= gxhi) & (y >= gylo) & (y <= gyhi)
            for version, keep in [("raw", finite), ("trimmed", trim)]:
                xv, yv = x[keep], y[keep]
                for q in [0, .0025, .005, .01]:
                    xc, yc = clipped(xv, yv, q)
                    result = detect_multires_hexbin(
                        xc, yc,
                        min_cluster_mass=0,
                        max_path_roughness=np.inf,
                    )
                    rows.append({
                        "model": model, "domain": domain, "pair": pair,
                        "version": version, "clip": q,
                        "detected": result.branch_detected,
                        "support": round(result.score * 4),
                        "coverage": result.x_coverage,
                    })

table = pd.DataFrame(rows)
print(table.groupby(["clip", "pair", "version"]).detected.sum().to_string())
print("\nP-Q raw detections by clip")
print(table.loc[(table.pair == "P → Q") & (table.version == "raw") & table.detected,
                ["clip", "model", "domain", "support", "coverage"]].to_string(index=False))
print("\nP-Q trimmed detections by clip")
print(table.loc[(table.pair == "P → Q") & (table.version == "trimmed") & table.detected,
                ["clip", "model", "domain", "support", "coverage"]].to_string(index=False))
table.to_csv("/tmp/hex_extent_probe.csv", index=False)
