#!/usr/bin/env python3
"""v4 significance-gate run (docs/strategy_v4_spec.md).

Fixed structure = v3 primary (short-only, HTF daily-EMA50 filter, TTM squeeze,
bb 20/2.0, kc_mult 1.5, min_squeeze_bars 4, release_window 3, stop 2.0 ATR,
trail 3.0 ATR, 3% risk, 5x cap, fee 0.0005 / slip 0.0002).

Two configs:
  v4a : + trade management (scale-out 50% at +1.5R, breakeven stop on the
        remainder kept with the chandelier trail; time stop at 12 bars if not
        scaled out and unrealized R < 0).
  v4b : v4a + pull-back re-entries within an episode (short-only, <=2 per
        episode, within 100 bars of the release bar, HTF still short-permissive,
        close crossing down through BBmid).

Both run on 1h (gate window 2025-11-20 + YTD window 2026-01-01; full gate-window
significance: bootstrap n=10000 seed=42, random-entry n=5000 seed=42, entries
placed in the gate window) and on 2h (resampled k=2; gate+YTD metrics only).

Outputs results/v4_gate.json.

Usage: python3 scripts/run_v4.py
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "v4_gate.json")

GATE_ENTRY_START = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
YTD_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# v3 primary structure + v4 trade management (scale-out + time stop).
V4_BASE = {
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
    "allow_long": False,
    "allow_short": True,
    "atr_n": 14,
    "stop_atr": 2.0,
    "trail_atr": 3.0,
    "flip_on_opposite": False,
    "fee": 0.0005,
    "slip": 0.0002,
    # v4 management
    "scale_out": True,
    "time_stop_bars": 12,
    "reentry_enabled": False,
    "reentry_max": 2,
    "reentry_window": 100,
}

CONFIGS = [
    ("v4a", {"reentry_enabled": False}),
    ("v4b", {"reentry_enabled": True}),
]

BOOT_N = 10000
RAND_N = 5000


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


def _parse(ts):
    return datetime.fromisoformat(ts)


def summarize_trades(trades):
    """scaled-out count, exit-reason breakdown, sorted R distribution (2dp)."""
    reasons = {}
    scaled = 0
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        if t["scaled_out"]:
            scaled += 1
    r_dist = sorted(round(t["r_multiple"], 2) for t in trades)
    return {"scaled_out": scaled, "exit_reasons": reasons, "r_distribution": r_dist}


def window_metrics(equity, trades, in_pos, bars, window_start):
    weq = [(ts, eq) for ts, eq in equity if ts >= window_start]
    wtr = [t for t in trades if _parse(t["entry_ts"]) >= window_start]
    widx = [i for i, b in enumerate(bars) if b.ts >= window_start]
    nw = len(widx)
    expo = (sum(1 for i in widx if in_pos[i]) / nw) if nw else 0.0
    m = metrics_mod.compute_metrics(weq, wtr, expo)
    return m, wtr


def run_config(bars, name, overrides, with_stats):
    params = dict(V4_BASE)
    params.update(overrides)

    signals, ind_data = strategy_mod.generate_signals(bars, params)
    atr_stop = ind_data["atr_stop"]

    bt = engine_mod.run_backtest(
        bars,
        signals,
        atr_stop,
        params,
        GATE_ENTRY_START,
        episode_release=ind_data.get("episode_release"),
        pullback_short=ind_data.get("pullback_short"),
    )
    trades = bt["trades"]
    equity = bt["equity"]
    in_pos = bt["in_pos_flags"]

    gate_m, gate_trades = window_metrics(equity, trades, in_pos, bars, GATE_ENTRY_START)
    ytd_m, ytd_trades = window_metrics(equity, trades, in_pos, bars, YTD_START)

    n_long = sum(1 for t in trades if t["side"] == "long")
    n_short = sum(1 for t in trades if t["side"] == "short")
    n_reentry = sum(1 for t in trades if t.get("is_reentry"))

    out = {
        "n_bars": len(bars),
        "gate_metrics": gate_m,
        "ytd_metrics": ytd_m,
        "gate_summary": summarize_trades(gate_trades),
        "ytd_summary": summarize_trades(ytd_trades),
        "trade_count_by_side": {"long": n_long, "short": n_short},
        "reentry_trades": n_reentry,
        "trades": trades,
    }

    if with_stats:
        r_multiples = [t["r_multiple"] for t in gate_trades]
        out["bootstrap"] = stats_mod.bootstrap_pvalue(r_multiples, n=BOOT_N, seed=42)
        out["random_entry"] = stats_mod.random_entry_test(
            bars, GATE_ENTRY_START, gate_trades, params, atr_stop, gate_m["sharpe"], n=RAND_N, seed=42
        )
    return out


def _row(label, m, extra=""):
    return "%-12s %6d %10s %8s %8s %7s %8s %8s%s" % (
        label,
        m["trades"],
        _fmt(m["net_return_pct"], "{:.2f}"),
        _fmt(m["sharpe"], "{:.3f}"),
        _fmt(m["max_dd_pct"], "{:.2f}"),
        _fmt(m["win_rate"], "{:.1f}"),
        _fmt(m["avg_r"], "{:.3f}"),
        _fmt(m["profit_factor"], "{:.3f}"),
        extra,
    )


def main():
    t0 = time.time()
    bars_1h = data_mod.load_bars(DATA_1H)
    bars_2h = data_mod.resample(bars_1h, 2)

    results = []
    for name, overrides in CONFIGS:
        r1 = run_config(bars_1h, name, overrides, with_stats=True)
        r2 = run_config(bars_2h, name, overrides, with_stats=False)
        params = dict(V4_BASE)
        params.update(overrides)
        results.append(
            {"config": name, "overrides": overrides, "params": params, "timeframes": {"1h": r1, "2h": r2}}
        )

    # Gate-window table
    print("Gate window (>= %s)" % GATE_ENTRY_START.isoformat())
    hdr = "%-12s %6s %10s %8s %8s %7s %8s %8s %9s %9s" % (
        "config/tf", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF", "boot_p", "rand_p",
    )
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        for tf in ("1h", "2h"):
            m = res["timeframes"][tf]["gate_metrics"]
            if tf == "1h":
                b = res["timeframes"][tf]["bootstrap"]
                r = res["timeframes"][tf]["random_entry"]
                extra = " %9s %9s" % (_fmt(b["p_value"], "{:.4f}"), _fmt(r["p_value"], "{:.4f}"))
            else:
                extra = " %9s %9s" % ("--", "--")
            print(_row("%s/%s" % (res["config"], tf), m, extra))

    # YTD-window table
    print("\n2026 YTD window (>= %s)" % YTD_START.isoformat())
    hdr2 = "%-12s %6s %10s %8s %8s %7s %8s %8s" % (
        "config/tf", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF",
    )
    print(hdr2)
    print("-" * len(hdr2))
    for res in results:
        for tf in ("1h", "2h"):
            print(_row("%s/%s" % (res["config"], tf), res["timeframes"][tf]["ytd_metrics"]))

    # side counts / re-entries
    print("\nTrade counts by side (full simulation = gate window) & re-entries:")
    for res in results:
        for tf in ("1h", "2h"):
            tfd = res["timeframes"][tf]
            c = tfd["trade_count_by_side"]
            print(
                "  %-10s long=%-3d short=%-3d total=%-3d reentries=%-3d"
                % ("%s/%s" % (res["config"], tf), c["long"], c["short"], c["long"] + c["short"], tfd["reentry_trades"])
            )

    # management breakdown (gate window)
    print("\nScale-out & exit-reason breakdown (gate window):")
    for res in results:
        for tf in ("1h", "2h"):
            s = res["timeframes"][tf]["gate_summary"]
            print(
                "  %-10s scaled_out=%-3d exit_reasons=%s"
                % ("%s/%s" % (res["config"], tf), s["scaled_out"], s["exit_reasons"])
            )

    # significance detail + R distribution (1h gate)
    print("\nSignificance detail (1h, gate window):")
    for res in results:
        b = res["timeframes"]["1h"]["bootstrap"]
        r = res["timeframes"]["1h"]["random_entry"]
        print(
            "  %-5s boot p=%s (mean_R=%s, n=%d) | rand p=%s (Sharpe=%s, rand_mean=%s, rand_p95=%s, n=%d)"
            % (
                res["config"], _fmt(b["p_value"], "{:.4f}"), _fmt(b["actual_mean_r"], "{:.3f}"), b["n_trades"],
                _fmt(r["p_value"], "{:.4f}"), _fmt(r["actual_sharpe"], "{:.3f}"),
                _fmt(r["random_mean_sharpe"], "{:.3f}"), _fmt(r["random_p95_sharpe"], "{:.3f}"), r["n"],
            )
        )
    print("\nR distributions (1h gate window, sorted, 2dp):")
    for res in results:
        print("  %-5s %s" % (res["config"], res["timeframes"]["1h"]["gate_summary"]["r_distribution"]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_window_start": GATE_ENTRY_START.isoformat(),
        "ytd_window_start": YTD_START.isoformat(),
        "base_params": V4_BASE,
        "bootstrap_n": BOOT_N,
        "random_entry_n": RAND_N,
        "configs": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)
    print("\nSaved %s" % RESULTS_PATH)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
