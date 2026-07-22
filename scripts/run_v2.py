#!/usr/bin/env python3
"""Run the v2 pre-registered variants on 1h and 2h data.

Structure (per docs/strategy_v2_spec.md): v1 TTM base with
  F1  higher-timeframe daily-EMA50 trend filter  (always ON)
  F2  TTM momentum oscillator confirmation        (ON/OFF variable)
  F3  breakout-volume confirmation                (ON/OFF variable)
and min_squeeze_bars lowered 6 -> 4.

8 variants = direction {short, both} x F2 {on, off} x F3 {on, off}.

1h: full significance tests per variant (bootstrap n=10000, random-entry
n=1000, seed=42). 2h (resampled from 1h, k=2): metrics only -- it is a
sign-replication check. Sharpe is daily-resampled x sqrt(365) on both.

Outputs results/v2_1h.json and results/v2_2h.json.

Usage: python3 scripts/run_v2.py
"""

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
DATA_1H = os.path.join(ROOT, "data", "eth_usd_1h.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

EVAL_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# v2 base parameters (docs/strategy_v2_spec.md). ttm mode; trend_ema disabled;
# F1 always on; min_squeeze_bars 4; costs/sizing unchanged.
V2_BASE = {
    "bb_n": 20,
    "bb_k": 2.0,
    "squeeze_mode": "ttm",
    "bw_lookback": 120,
    "bw_q": 20.0,
    "kc_mult": 1.5,
    "min_squeeze_bars": 4,
    "release_window": 3,
    "trend_ema": 0,
    "htf_trend": True,
    "htf_ema_n": 50,
    "momentum_filter": False,
    "volume_filter": False,
    "vol_mult": 1.3,
    "mom_n": 20,
    "vol_sma_n": 20,
    "allow_long": True,
    "allow_short": True,
    "atr_n": 14,
    "stop_atr": 2.0,
    "trail_atr": 3.0,
    "flip_on_opposite": False,
    "fee": 0.0005,
    "slip": 0.0002,
}

DIRECTIONS = [("short", False, True), ("both", True, True)]
F2_OPTS = [("F2on", True), ("F2off", False)]
F3_OPTS = [("F3on", True), ("F3off", False)]


def _fmt(v, spec="{:.3f}"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return "n/a"
    return spec.format(v)


def _sanitize(obj):
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def build_variants():
    variants = []
    for dir_label, al, ash in DIRECTIONS:
        for f2_label, f2 in F2_OPTS:
            for f3_label, f3 in F3_OPTS:
                params = dict(V2_BASE)
                params["allow_long"] = al
                params["allow_short"] = ash
                params["momentum_filter"] = f2
                params["volume_filter"] = f3
                label = "%s_%s_%s" % (dir_label, f2_label, f3_label)
                variants.append((label, dir_label, f2_label, f3_label, params))
    return variants


def run_variant(bars, label, dir_label, f2_label, f3_label, params, with_stats):
    signals, ind_data = strategy_mod.generate_signals(bars, params)
    atr_stop = ind_data["atr_stop"]

    bt = engine_mod.run_backtest(bars, signals, atr_stop, params, EVAL_START)
    trades = bt["trades"]
    equity_full = bt["equity"]
    in_pos = bt["in_pos_flags"]

    eval_equity = [(ts, eq) for ts, eq in equity_full if ts >= EVAL_START]
    eval_idx = [i for i, b in enumerate(bars) if b.ts >= EVAL_START]
    n_eval = len(eval_idx)
    exposure = (sum(1 for i in eval_idx if in_pos[i]) / n_eval) if n_eval else 0.0

    metrics = metrics_mod.compute_metrics(eval_equity, trades, exposure)

    n_long = sum(1 for t in trades if t["side"] == "long")
    n_short = sum(1 for t in trades if t["side"] == "short")

    out = {
        "variant": label,
        "direction": dir_label,
        "f2": f2_label,
        "f3": f3_label,
        "params": params,
        "metrics": metrics,
        "trade_count_by_side": {"long": n_long, "short": n_short},
        "trades": trades,
    }

    if with_stats:
        r_multiples = [t["r_multiple"] for t in trades]
        out["bootstrap"] = stats_mod.bootstrap_pvalue(r_multiples, n=10000, seed=42)
        out["random_entry"] = stats_mod.random_entry_test(
            bars, EVAL_START, trades, params, atr_stop, metrics["sharpe"], n=1000, seed=42
        )
    return out


def print_table_1h(results):
    header = "%-20s %6s %10s %8s %8s %7s %8s %8s %9s %9s" % (
        "variant", "trades", "netRet%", "Sharpe", "maxDD%", "win%",
        "avgR", "PF", "boot_p", "rand_p",
    )
    print(header)
    print("-" * len(header))
    for res in results:
        m = res["metrics"]
        b = res["bootstrap"]
        r = res["random_entry"]
        print(
            "%-20s %6d %10s %8s %8s %7s %8s %8s %9s %9s"
            % (
                res["variant"], m["trades"],
                _fmt(m["net_return_pct"], "{:.2f}"), _fmt(m["sharpe"], "{:.3f}"),
                _fmt(m["max_dd_pct"], "{:.2f}"), _fmt(m["win_rate"], "{:.1f}"),
                _fmt(m["avg_r"], "{:.3f}"), _fmt(m["profit_factor"], "{:.3f}"),
                _fmt(b["p_value"], "{:.4f}"), _fmt(r["p_value"], "{:.4f}"),
            )
        )


def print_table_2h(results):
    header = "%-20s %6s %10s %8s %8s" % ("variant", "trades", "netRet%", "Sharpe", "maxDD%")
    print(header)
    print("-" * len(header))
    for res in results:
        m = res["metrics"]
        print(
            "%-20s %6d %10s %8s %8s"
            % (
                res["variant"], m["trades"],
                _fmt(m["net_return_pct"], "{:.2f}"), _fmt(m["sharpe"], "{:.3f}"),
                _fmt(m["max_dd_pct"], "{:.2f}"),
            )
        )


def print_side_counts(results):
    print("\nTrade counts by side:")
    for res in results:
        c = res["trade_count_by_side"]
        print(
            "  %-20s long=%-4d short=%-4d total=%-4d"
            % (res["variant"], c["long"], c["short"], c["long"] + c["short"])
        )


def save(path, tf, bars, results):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": tf,
        "n_bars": len(bars),
        "eval_start": EVAL_START.isoformat(),
        "base_params": V2_BASE,
        "variants": results,
    }
    with open(path, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)


def main():
    t0 = time.time()
    variants = build_variants()

    bars_1h = data_mod.load_bars(DATA_1H)
    bars_2h = data_mod.resample(bars_1h, 2)

    # ---- 1h: full significance tests ----
    print("=" * 60)
    print("1h  (%d bars, %s .. %s)" % (len(bars_1h), bars_1h[0].ts.isoformat(), bars_1h[-1].ts.isoformat()))
    print("=" * 60)
    res_1h = [run_variant(bars_1h, *v, with_stats=True) for v in variants]
    print_table_1h(res_1h)
    print_side_counts(res_1h)
    print("\nSignificance detail (1h):")
    for res in res_1h:
        b = res["bootstrap"]
        r = res["random_entry"]
        print(
            "  %-20s boot p=%s (mean_R=%s) | rand p=%s "
            "(Sharpe=%s, rand_mean=%s, rand_p95=%s, n=%d)"
            % (
                res["variant"], _fmt(b["p_value"], "{:.4f}"), _fmt(b["actual_mean_r"], "{:.3f}"),
                _fmt(r["p_value"], "{:.4f}"), _fmt(r["actual_sharpe"], "{:.3f}"),
                _fmt(r["random_mean_sharpe"], "{:.3f}"), _fmt(r["random_p95_sharpe"], "{:.3f}"), r["n"],
            )
        )

    # ---- 2h: metrics only ----
    print("\n" + "=" * 60)
    print("2h  (%d bars, resampled k=2 from 1h)" % len(bars_2h))
    print("=" * 60)
    res_2h = [run_variant(bars_2h, *v, with_stats=False) for v in variants]
    print_table_2h(res_2h)
    print_side_counts(res_2h)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    p1 = os.path.join(RESULTS_DIR, "v2_1h.json")
    p2 = os.path.join(RESULTS_DIR, "v2_2h.json")
    save(p1, "1h", bars_1h, res_1h)
    save(p2, "2h", bars_2h, res_2h)
    print("\nSaved %s and %s" % (p1, p2))
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
