#!/usr/bin/env python3
"""v3 significance-gate run (docs/strategy_v3_spec.md).

The v2 structure is fixed to short_F2off_F3off (short-only, HTF daily-EMA50
filter F1 on, TTM squeeze, F2/F3 off). To raise test power the *significance
window* is widened to the full warmed-up period (gate_entry_start =
2025-11-20T00:00:00Z, the first bar at which the daily EMA50 is seeded). The
performance-requirement window (2026 YTD) is unchanged and reported separately.

One simulation per config (entries suppressed before gate_entry_start); two
metric windows are computed from the same equity curve:
  (a) gate window  : equity/trades from 2025-11-20 -> used for the gate + tests
  (b) 2026 YTD     : equity re-based at 2026-01-01 -> reference performance

3 configs (sequential registration order):
  1. primary       : v2 short_F2off_F3off exact
  2. alt1_kc1.75   : same but kc_mult = 1.75
  3. alt2_rw5      : same but release_window = 5

All three run on 1h with full gate-window significance tests (bootstrap
n=10000, random-entry n=1000, seed=42; random entries placed within the gate
window) and on 2h (resampled k=2) with gate-window metrics only.

Outputs results/v3_gate.json.

Usage: python3 scripts/run_v3.py
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "v3_gate.json")

# Entries are allowed to fill from this timestamp onward (and it is also the
# start of the widened significance window).
GATE_ENTRY_START = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
# Reference performance window (unchanged from v1/v2).
YTD_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Fixed v2 short_F2off_F3off structure.
V3_BASE = {
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
}

# Sequential registration order (primary first, then the two fallbacks).
CONFIGS = [
    ("primary", {}),
    ("alt1_kc1.75", {"kc_mult": 1.75}),
    ("alt2_rw5", {"release_window": 5}),
]


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


def window_metrics(equity, trades, in_pos, bars, window_start):
    """Metrics on the sub-window [window_start, end] of a single simulation.

    Equity is sliced (re-based at the window start via compute_metrics, which
    takes eq_start = first sliced value); trades are those whose entry falls in
    the window; exposure is over the window's bars.
    """
    weq = [(ts, eq) for ts, eq in equity if ts >= window_start]
    wtr = [t for t in trades if _parse(t["entry_ts"]) >= window_start]
    widx = [i for i, b in enumerate(bars) if b.ts >= window_start]
    nw = len(widx)
    expo = (sum(1 for i in widx if in_pos[i]) / nw) if nw else 0.0
    return metrics_mod.compute_metrics(weq, wtr, expo)


def run_config(bars, name, overrides, with_stats):
    params = dict(V3_BASE)
    params.update(overrides)

    signals, ind_data = strategy_mod.generate_signals(bars, params)
    atr_stop = ind_data["atr_stop"]

    # Single simulation; entries suppressed before the gate entry start.
    bt = engine_mod.run_backtest(bars, signals, atr_stop, params, GATE_ENTRY_START)
    trades = bt["trades"]
    equity = bt["equity"]
    in_pos = bt["in_pos_flags"]

    gate = window_metrics(equity, trades, in_pos, bars, GATE_ENTRY_START)
    ytd = window_metrics(equity, trades, in_pos, bars, YTD_START)

    n_long = sum(1 for t in trades if t["side"] == "long")
    n_short = sum(1 for t in trades if t["side"] == "short")

    out = {
        "n_bars": len(bars),
        "gate_metrics": gate,
        "ytd_metrics": ytd,
        "trade_count_by_side": {"long": n_long, "short": n_short},
        "trades": trades,
    }

    if with_stats:
        # Significance on the gate window: all trades fall in it by construction.
        gate_trades = [t for t in trades if _parse(t["entry_ts"]) >= GATE_ENTRY_START]
        r_multiples = [t["r_multiple"] for t in gate_trades]
        out["bootstrap"] = stats_mod.bootstrap_pvalue(r_multiples, n=10000, seed=42)
        out["random_entry"] = stats_mod.random_entry_test(
            bars,
            GATE_ENTRY_START,
            gate_trades,
            params,
            atr_stop,
            gate["sharpe"],
            n=1000,
            seed=42,
        )
    return out


def _metric_row(label, m, extra=""):
    return "%-22s %6d %10s %8s %8s %7s %8s %8s%s" % (
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
        params = dict(V3_BASE)
        params.update(overrides)
        results.append(
            {
                "config": name,
                "overrides": overrides,
                "params": params,
                "timeframes": {"1h": r1, "2h": r2},
            }
        )

    # ---- Gate-window table ----
    print("Gate window (>= %s) -- used for significance / gate" % GATE_ENTRY_START.isoformat())
    hdr = "%-22s %6s %10s %8s %8s %7s %8s %8s %9s %9s" % (
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
            print(_metric_row("%s/%s" % (res["config"], tf), m, extra))

    # ---- YTD-window table ----
    print("\n2026 YTD window (>= %s) -- reference performance" % YTD_START.isoformat())
    hdr2 = "%-22s %6s %10s %8s %8s %7s %8s %8s" % (
        "config/tf", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF",
    )
    print(hdr2)
    print("-" * len(hdr2))
    for res in results:
        for tf in ("1h", "2h"):
            m = res["timeframes"][tf]["ytd_metrics"]
            print(_metric_row("%s/%s" % (res["config"], tf), m))

    # ---- side counts ----
    print("\nTrade counts by side (full simulation = gate window):")
    for res in results:
        for tf in ("1h", "2h"):
            c = res["timeframes"][tf]["trade_count_by_side"]
            print(
                "  %-18s long=%-4d short=%-4d total=%-4d"
                % ("%s/%s" % (res["config"], tf), c["long"], c["short"], c["long"] + c["short"])
            )

    # ---- significance detail (1h) ----
    print("\nSignificance detail (1h, gate window):")
    for res in results:
        b = res["timeframes"]["1h"]["bootstrap"]
        r = res["timeframes"]["1h"]["random_entry"]
        print(
            "  %-14s boot p=%s (mean_R=%s, n_trades=%d) | rand p=%s "
            "(Sharpe=%s, rand_mean=%s, rand_p95=%s, n=%d)"
            % (
                res["config"], _fmt(b["p_value"], "{:.4f}"), _fmt(b["actual_mean_r"], "{:.3f}"),
                b["n_trades"], _fmt(r["p_value"], "{:.4f}"), _fmt(r["actual_sharpe"], "{:.3f}"),
                _fmt(r["random_mean_sharpe"], "{:.3f}"), _fmt(r["random_p95_sharpe"], "{:.3f}"), r["n"],
            )
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_entry_start": GATE_ENTRY_START.isoformat(),
        "gate_window_start": GATE_ENTRY_START.isoformat(),
        "ytd_window_start": YTD_START.isoformat(),
        "base_params": V3_BASE,
        "configs": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)

    print("\nSaved %s" % RESULTS_PATH)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
