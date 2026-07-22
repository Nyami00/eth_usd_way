#!/usr/bin/env python3
"""Run the v1 prototype backtest over the 6 structural configs and save results.

Configs: {both, long-only, short-only} x {squeeze_mode pctile, ttm}.

For each config it prints trades, net return %, Sharpe, maxDD %, win rate,
avg R, profit factor, exposure, a trade-count-by-side line, and both
significance tests (bootstrap p, random-entry p with n=1000). Everything is
saved to results/prototype_{label}.json.

Bar-count-based parameters (bb_n=20, etc.) are NOT rescaled for finer bars:
they are the same integer counts regardless of the bar interval.

Usage:
  python3 scripts/run_prototype.py [--data PATH] [--label LABEL]
    --data   CSV path (default: data/eth_usd_1h.csv)
    --label  filename tag -> results/prototype_{label}.json (default: 1h)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import data as data_mod
from backtest import engine as engine_mod
from backtest import metrics as metrics_mod
from backtest import stats as stats_mod
from backtest import strategy as strategy_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT, "data", "eth_usd_1h.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

EVAL_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# v1 default parameters (from docs/implementation_spec_v1.md).
BASE_PARAMS = {
    "bb_n": 20,
    "bb_k": 2.0,
    "squeeze_mode": "pctile",
    "bw_lookback": 120,
    "bw_q": 20.0,
    "kc_mult": 1.5,
    "min_squeeze_bars": 6,
    "release_window": 3,
    "trend_ema": 200,
    "allow_long": True,
    "allow_short": True,
    "atr_n": 14,
    "stop_atr": 2.0,
    "trail_atr": 3.0,
    "flip_on_opposite": False,
    "fee": 0.0005,
    "slip": 0.0002,
}

SIDE_CONFIGS = [
    ("both", True, True),
    ("long", True, False),
    ("short", False, True),
]
MODES = ["pctile", "ttm"]


def _fmt(v, spec="{:.3f}"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return "n/a"
    return spec.format(v)


def _sanitize(obj):
    """Recursively replace inf/nan floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def run_config(bars, mode, side_label, allow_long, allow_short):
    params = dict(BASE_PARAMS)
    params["squeeze_mode"] = mode
    params["allow_long"] = allow_long
    params["allow_short"] = allow_short

    signals, ind_data = strategy_mod.generate_signals(bars, params)
    atr_stop = ind_data["atr_stop"]

    bt = engine_mod.run_backtest(bars, signals, atr_stop, params, EVAL_START)
    trades = bt["trades"]
    equity_full = bt["equity"]
    in_pos = bt["in_pos_flags"]

    # Restrict to the evaluation window.
    eval_equity = [(ts, eq) for ts, eq in equity_full if ts >= EVAL_START]
    eval_idx = [i for i, b in enumerate(bars) if b.ts >= EVAL_START]
    n_eval = len(eval_idx)
    exposure = (sum(1 for i in eval_idx if in_pos[i]) / n_eval) if n_eval else 0.0

    metrics = metrics_mod.compute_metrics(eval_equity, trades, exposure)

    n_long = sum(1 for t in trades if t["side"] == "long")
    n_short = sum(1 for t in trades if t["side"] == "short")

    r_multiples = [t["r_multiple"] for t in trades]
    boot = stats_mod.bootstrap_pvalue(r_multiples, n=10000, seed=42)
    rand = stats_mod.random_entry_test(
        bars,
        EVAL_START,
        trades,
        params,
        atr_stop,
        metrics["sharpe"],
        n=1000,
        seed=42,
    )

    return {
        "config": "%s_%s" % (mode, side_label),
        "squeeze_mode": mode,
        "side": side_label,
        "params": params,
        "metrics": metrics,
        "trade_count_by_side": {"long": n_long, "short": n_short},
        "bootstrap": boot,
        "random_entry": rand,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the v1 prototype backtest.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="CSV data path")
    parser.add_argument("--label", default="1h", help="output filename tag")
    args = parser.parse_args()

    data_path = args.data
    results_path = os.path.join(RESULTS_DIR, "prototype_%s.json" % args.label)

    t0 = time.time()
    bars = data_mod.load_bars(data_path)
    print("Data: %s  label: %s" % (data_path, args.label))
    print("Loaded %d bars (%s .. %s)" % (len(bars), bars[0].ts.isoformat(), bars[-1].ts.isoformat()))
    print("Evaluation window: >= %s\n" % EVAL_START.isoformat())

    results = []
    for mode in MODES:
        for side_label, al, ash in SIDE_CONFIGS:
            res = run_config(bars, mode, side_label, al, ash)
            results.append(res)

    # ---- print comparison table ----
    header = (
        "%-14s %6s %10s %8s %8s %8s %8s %8s %9s %10s %10s"
        % (
            "config",
            "trades",
            "netRet%",
            "Sharpe",
            "maxDD%",
            "win%",
            "avgR",
            "PF",
            "exposure",
            "boot_p",
            "rand_p",
        )
    )
    print(header)
    print("-" * len(header))
    for res in results:
        m = res["metrics"]
        b = res["bootstrap"]
        r = res["random_entry"]
        print(
            "%-14s %6d %10s %8s %8s %8s %8s %8s %9s %10s %10s"
            % (
                res["config"],
                m["trades"],
                _fmt(m["net_return_pct"], "{:.2f}"),
                _fmt(m["sharpe"], "{:.3f}"),
                _fmt(m["max_dd_pct"], "{:.2f}"),
                _fmt(m["win_rate"], "{:.1f}"),
                _fmt(m["avg_r"], "{:.3f}"),
                _fmt(m["profit_factor"], "{:.3f}"),
                _fmt(m["exposure"], "{:.4f}"),
                _fmt(b["p_value"], "{:.4f}"),
                _fmt(r["p_value"], "{:.4f}"),
            )
        )

    print("\nPer-config trade counts by side:")
    for res in results:
        c = res["trade_count_by_side"]
        print(
            "  %-14s long=%-4d short=%-4d total=%-4d"
            % (res["config"], c["long"], c["short"], c["long"] + c["short"])
        )

    print("\nSignificance detail:")
    for res in results:
        b = res["bootstrap"]
        r = res["random_entry"]
        print(
            "  %-14s bootstrap p=%s (mean_R=%s) | random-entry p=%s "
            "(actual_Sharpe=%s, rand_mean=%s, rand_p95=%s, n=%d)"
            % (
                res["config"],
                _fmt(b["p_value"], "{:.4f}"),
                _fmt(b["actual_mean_r"], "{:.3f}"),
                _fmt(r["p_value"], "{:.4f}"),
                _fmt(r["actual_sharpe"], "{:.3f}"),
                _fmt(r["random_mean_sharpe"], "{:.3f}"),
                _fmt(r["random_p95_sharpe"], "{:.3f}"),
                r["n"],
            )
        )

    # ---- save JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "label": args.label,
        "n_bars": len(bars),
        "eval_start": EVAL_START.isoformat(),
        "base_params": BASE_PARAMS,
        "configs": results,
    }
    with open(results_path, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)

    print("\nSaved results to %s" % results_path)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
