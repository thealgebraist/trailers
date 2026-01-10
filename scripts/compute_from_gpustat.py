#!/usr/bin/env python3
"""Compute capacity and profit scenarios from gpustat snapshots.

Usage:
  - Place gpustat JSON/text snapshots in ./gpustat/ (workspace root) and run the script.
  - Or run the builtin `--test` to validate with mock data.

The script outputs CSV-like summaries and a simple recomputation of outputs/hr and profit given rental/ownership costs.
"""
import sys
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
GPUDIR = ROOT / "gpustat"

# Default assumptions (can be tuned)
ASSUMPTIONS = {
    "M": 80.0,  # GB HBM
    "overhead_gb": 10.0,
    "price_per_output": 0.02,
    "rental_rates": [3.0, 8.0, 15.0],
}


def parse_snapshot(obj):
    """Parse a gpustat JSON object (dict) and return per-GPU metrics.

    Expected minimal shape: {"gpus": [{"index":0, "name":"H200", "memory.total":nn, "memory.used":nn, "utilization.gpu":nn, "power.draw":nn}, ...]}
    The script attempts several common key names.
    """
    gpus = obj.get("gpus") or obj.get("gpu_stats") or obj.get("devices")
    if not gpus:
        # maybe the object is a list already
        if isinstance(obj, list):
            gpus = obj
        else:
            raise ValueError("No 'gpus' array found in snapshot")
    out = []
    for g in gpus:
        # flexible key access
        mt = g.get("memory.total") or g.get("memory_total") or g.get("memory_total_mb")
        mu = g.get("memory.used") or g.get("memory_used") or g.get("memory_used_mb")
        util = g.get("utilization.gpu") or g.get("utilization_gpu") or g.get("util")
        power = g.get("power.draw") or g.get("power_draw") or g.get("power")
        name = g.get("name") or g.get("model") or g.get("gpu_name")
        idx = g.get("index") or g.get("id")
        out.append({
            "index": idx,
            "name": name,
            "memory_total": float(mt) if mt is not None else None,
            "memory_used": float(mu) if mu is not None else None,
            "util": float(util) if util is not None else None,
            "power": float(power) if power is not None else None,
        })
    return out


def collect_snapshots(dirpath: Path):
    snaps = []
    if not dirpath.exists():
        return snaps
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix in (".json", ".txt"):
            try:
                txt = p.read_text()
                obj = json.loads(txt)
            except Exception:
                # try line-per-gpu JSON or other simple parsing
                continue
            try:
                snaps.append(parse_snapshot(obj))
            except Exception:
                continue
    return snaps


def summarize(snaps):
    # Flatten per-GPU across snapshots (assume consistent indices)
    if not snaps:
        return None
    # compute average metrics per GPU index present in first snapshot
    idxs = [g["index"] for g in snaps[0]]
    summary = {}
    for i in idxs:
        mem_total = []
        mem_used = []
        utils = []
        power = []
        name = None
        for s in snaps:
            # find gpu with index i
            entry = next((x for x in s if x["index"] == i), None)
            if not entry:
                continue
            name = name or entry.get("name")
            if entry.get("memory_total") is not None:
                mem_total.append(entry["memory_total"]) 
            if entry.get("memory_used") is not None:
                mem_used.append(entry["memory_used"]) 
            if entry.get("util") is not None:
                utils.append(entry["util"]) 
            if entry.get("power") is not None:
                power.append(entry["power"]) 
        summary[i] = {
            "name": name,
            "memory_total_gb": mean(mem_total) / 1024.0 if mem_total and max(mem_total) > 50 else (mean(mem_total) if mem_total else None),
            "memory_used_gb": mean(mem_used) / 1024.0 if mem_used and max(mem_used) > 50 else (mean(mem_used) if mem_used else None),
            "util_pct": mean(utils) if utils else None,
            "power_w": mean(power) if power else None,
        }
    return summary


def compute_capacity_and_profit(summary, assumptions=ASSUMPTIONS):
    # For each GPU, compute c (contexts) using simple per-session memory estimates and earlier logic.
    results = {}
    for idx, info in summary.items():
        M = info.get("memory_total_gb") or assumptions["M"]
        # assume model weight sizes for scenarios (GB)
        models = {"7B":14.0, "13B":26.0, "30B":60.0}
        per_token_mem = {"7B":0.00045, "13B":0.0008, "30B":0.0012}
        L = 1024
        r = 1.0/50.0
        S = 5.0
        results[idx] = {"name": info.get("name"), "M_gb":M}
        for mname, W in models.items():
            M_avail = M - W - assumptions["overhead_gb"]
            C_L = per_token_mem[mname] * L
            c = max(0, math.floor(M_avail / C_L))
            mu = 1.0 / S
            cap_req_s = c * mu
            U_max = cap_req_s / r if r>0 else None
            outputs_per_hr = cap_req_s * 3600.0
            revenue_hr = outputs_per_hr * assumptions["price_per_output"]
            results[idx][mname] = {
                "M_avail_gb": M_avail,
                "contexts": c,
                "outputs_per_hr": outputs_per_hr,
                "revenue_hr": revenue_hr,
            }
    return results


def format_results(results):
    lines = []
    for idx, r in results.items():
        lines.append(f"GPU {idx}: {r.get('name')} (HBM={r.get('M_gb')} GB)")
        for m in ("7B","13B","30B"):
            v = r[m]
            lines.append(f"  {m}: contexts={v['contexts']}, outputs/hr={v['outputs_per_hr']:.1f}, revenue/hr=${v['revenue_hr']:.2f}")
    return "\n".join(lines)


def run_with_dir(dirpath: Path):
    snaps = collect_snapshots(dirpath)
    if not snaps:
        print(f"No gpustat snapshots found in {dirpath}. Place JSON snapshots there or run with --test")
        return 2
    summary = summarize(snaps)
    results = compute_capacity_and_profit(summary)
    print(format_results(results))
    return 0


def test():
    # create a mocked snapshot for a single H200-like GPU (82000 MB total)
    mock = [{
        "index": 0,
        "name": "H200-80GB",
        "memory.total": 82000,
        "memory.used": 20000,
        "utilization.gpu": 25.0,
        "power.draw": 350.0,
    }]
    snaps = [mock for _ in range(6)]
    summary = summarize(snaps)
    results = compute_capacity_and_profit(summary)
    print(format_results(results))
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(test())
    sys.exit(run_with_dir(GPUDIR))
