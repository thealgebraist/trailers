#!/usr/bin/env python3
"""Simulate profit scenarios and save results.

Usage:
  python3 scripts/profit_simulator.py [--outdir outputs] [--append-tex PATH]

Outputs:
  - JSON and CSV files under `outputs/` by default
  - If `--append-tex PATH` is given, append a small appendix table to the specified .tex file
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_FILE = ROOT / "scripts" / "providers_defaults.json"

# defaults
GPU_MODES = {
    "H200": {
        "HBM_GB": 80.0,
        "overhead_gb": 10.0,
        "monthly_rate_per_hr": 3.0,
        "weekday_rate_per_hr": 4.5,
        "service_time_s": 5.0,
    },
    "3060": {
        "HBM_GB": 12.0,
        "overhead_gb": 3.0,
        "monthly_rate_per_hr": 0.5,
        "weekday_rate_per_hr": 0.9,
        "service_time_s": 6.0,
    }
}

# per-token memory approx used earlier
PER_TOKEN_MEM = {"7B": 0.00045, "13B": 0.0008, "30B": 0.0012}
MODEL_WEIGHTS = {"7B": 14.0, "13B": 26.0, "30B": 60.0}
L = 1024

# Monte Carlo settings
MC_ITERS = 2000


def load_providers(path=PROVIDERS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Missing providers file: {path}")
    return json.loads(path.read_text())


def compute_contexts(M, W, overhead):
    M_avail = M - W - overhead
    C_L = PER_TOKEN_MEM["13B"] * L
    if C_L <= 0:
        return 0, M_avail
    c = max(0, int(math.floor(M_avail / C_L)))
    return c, M_avail


def gpu_capacity_outputs_per_hr(gpu_mode, model_key="13B"):
    gpu = GPU_MODES[gpu_mode]
    M = gpu["HBM_GB"]
    W = MODEL_WEIGHTS[model_key]
    c, M_avail = compute_contexts(M, W, gpu["overhead_gb"])
    S = gpu["service_time_s"]
    mu = 1.0 / S
    outputs_per_s = c * mu
    outputs_per_hr = outputs_per_s * 3600.0
    return {"contexts": c, "outputs_per_hr": outputs_per_hr, "M_avail": M_avail}


def simulate_profit(provider_price_per_1M, gpu_mode, rental_mode="monthly", model_key="13B", mc_iters=MC_ITERS):
    gpu = GPU_MODES[gpu_mode]
    rate = gpu["monthly_rate_per_hr"] if rental_mode == "monthly" else gpu["weekday_rate_per_hr"]
    cap = gpu_capacity_outputs_per_hr(gpu_mode, model_key)
    outputs_per_hr = cap["outputs_per_hr"]
    price_per_token = provider_price_per_1M / 1_000_000.0

    samples_profit = []
    for _ in range(mc_iters):
        demand_multiplier = random.lognormvariate(mu=0.0, sigma=0.3)
        tokens_per_request = random.lognormvariate(mu=5.5, sigma=0.8)
        demand_req_hr = demand_multiplier * 100.0
        served_requests = min(outputs_per_hr, demand_req_hr)
        served_tokens = served_requests * tokens_per_request
        revenue_hr = served_tokens * price_per_token
        profit_hr = revenue_hr - rate
        samples_profit.append(profit_hr)

    mean = statistics.mean(samples_profit)
    var = statistics.pvariance(samples_profit)
    p5 = sorted(samples_profit)[int(0.05 * len(samples_profit))]
    p95 = sorted(samples_profit)[int(0.95 * len(samples_profit))]
    return {
        "mean_profit_hr": mean,
        "var_profit_hr": var,
        "p5": p5,
        "p95": p95,
        "capacity_outputs_hr": outputs_per_hr,
        "rental_rate_hr": rate,
    }


def save_results(results, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "profit_results.json"
    csv_path = outdir / "profit_results.csv"
    json_path.write_text(json.dumps(results, indent=2))

    # flatten to CSV rows
    rows = []
    for provider, details in results.items():
        for scenario, val in details.items():
            rows.append({
                "provider": provider,
                "scenario": scenario,
                "mean_profit_hr": val["mean_profit_hr"],
                "var_profit_hr": val["var_profit_hr"],
                "p5": val["p5"],
                "p95": val["p95"],
                "capacity_outputs_hr": val["capacity_outputs_hr"],
                "rental_rate_hr": val["rental_rate_hr"],
            })

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return json_path, csv_path


def append_tex(results, tex_path: Path):
    # append a small appendix with a tabular summary
    safe = lambda s: s.replace("_", "\\_")
    lines = ["\n% -- profit_simulator appendix --\n", "\\section*{Profit Simulator Results}\n", "\\begin{tabular}{l l r r r r}\n", "Provider & Scenario & Mean (\$ / hr) & Var & P5 & P95\\\\\n", "\\hline\n"]
    for provider, details in results.items():
        for scenario, v in details.items():
            lines.append(f"{safe(provider)} & {safe(scenario)} & {v['mean_profit_hr']:.4f} & {v['var_profit_hr']:.4g} & {v['p5']:.4f} & {v['p95']:.4f}\\\\\n")
    lines.append("\\end{tabular}\n")
    tex_path.write_text(tex_path.read_text() + "".join(lines))


def run_all(outdir: Path, append_tex_path: Path = None, mc_iters: int = MC_ITERS):
    providers = load_providers()
    results = {}
    for name, info in providers.items():
        price = info.get("price_per_1M_output_tokens")
        results[name] = {}
        for gpu in GPU_MODES.keys():
            for rental in ("monthly", "weekday"):
                key = f"{gpu}_{rental}"
                res = simulate_profit(price, gpu, rental_mode=("monthly" if rental == "monthly" else "weekday"), mc_iters=mc_iters)
                results[name][key] = res

    json_path, csv_path = save_results(results, outdir)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    if append_tex_path:
        append_tex(results, Path(append_tex_path))
        print(f"Appended results to: {append_tex_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--append-tex", dest="append_tex", default=None, help="Path to .tex file to append results to")
    p.add_argument("--iters", dest="iters", default=MC_ITERS, type=int, help="Monte Carlo iterations")
    args = p.parse_args()
    outdir = Path(args.outdir)
    run_all(outdir, append_tex_path=args.append_tex, mc_iters=args.iters)


if __name__ == "__main__":
    main()
