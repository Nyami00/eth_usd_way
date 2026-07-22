#!/usr/bin/env python3
"""v5 out-of-sample validation of the FROZEN v4a rule (docs/strategy_v5_spec.md).

The rule is frozen (= v4a) and validated on untouched 2020-2025 history:
  short-only TTM squeeze breakout (BB 20/2.0 inside KC 20/1.5xATR20, >=4 bars),
  release within 3 bars, close < daily EMA50 (prior completed day),
  stop 2.0xATR14, chandelier trail 3.0xATR14, scale-out 50% at +1.5R with the
  remainder moved to breakeven, time stop at 12 bars if unrealized R<0 and not
  scaled out, 3% risk, 5x cap, fee 0.0005 / slip 0.0002.

Two runs:
  (a) VALIDATION : entries allowed 2020-03-01 .. 2025-09-30 (entry cutoff +
      window-end force-close so nothing bleeds into Q4-2025). Metrics and
      significance (bootstrap n=10000 seed=42, random-entry n=1000 seed=42,
      random entries placed within this window only) on that window.
  (b) 2026 YTD REFERENCE : identical to the v4a run (entries from 2025-11-20,
      gate + YTD windows, gate-window significance) re-emitted for completeness.

Auxiliary (no gate impact): per-calendar-year metrics, bear-regime share
(close < daily EMA50) per year, sorted R distribution, 2022 full-year metrics,
leverage-cap-binding trade count.

Outputs results/v5_validation.json.

Usage: python3 scripts/run_v5.py   (requires data/eth_usd_1h.csv starting 2020)
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "v5_validation.json")

VAL_ENTRY_START = datetime(2020, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
VAL_ENTRY_END = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
GATE_ENTRY_START = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
YTD_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

VAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Frozen v4a parameters (short-only; scale-out + time stop; HTF daily EMA50).
V5_PARAMS = {
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
    "scale_out": True,
    "time_stop_bars": 12,
    "reentry_enabled": False,
    "reentry_max": 2,
    "reentry_window": 100,
}

BOOT_N = 10000
VAL_RAND_N = 1000
REF_RAND_N = 5000


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


def window_metrics(equity, trades, in_pos, bars, win_start, win_end=None):
    weq = [(ts, eq) for ts, eq in equity if ts >= win_start and (win_end is None or ts <= win_end)]
    wtr = [
        t
        for t in trades
        if _parse(t["entry_ts"]) >= win_start and (win_end is None or _parse(t["entry_ts"]) <= win_end)
    ]
    widx = [i for i, b in enumerate(bars) if b.ts >= win_start and (win_end is None or b.ts <= win_end)]
    nw = len(widx)
    expo = (sum(1 for i in widx if in_pos[i]) / nw) if nw else 0.0
    return metrics_mod.compute_metrics(weq, wtr, expo), wtr


def bear_regime_share(bars, htf_ema):
    """Per-year fraction of bars (with an available daily EMA50) with close < EMA50."""
    by_year = {}
    for i, b in enumerate(bars):
        he = htf_ema[i] if htf_ema is not None else None
        if he is None:
            continue
        d = by_year.setdefault(b.ts.year, [0, 0])
        d[1] += 1
        if b.close < he:
            d[0] += 1
    return {str(y): round(v[0] / v[1], 4) for y, v in sorted(by_year.items()) if v[1] > 0}


def summarize_trades(trades):
    reasons = {}
    scaled = 0
    capped = 0
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        if t["scaled_out"]:
            scaled += 1
        if t.get("leverage_capped"):
            capped += 1
    return {
        "scaled_out": scaled,
        "leverage_capped": capped,
        "exit_reasons": reasons,
        "r_distribution": sorted(round(t["r_multiple"], 2) for t in trades),
    }


def _row(label, m, extra=""):
    return "%-16s %6d %10s %8s %8s %7s %8s %8s%s" % (
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
    bars = data_mod.load_bars(DATA_1H)
    print("Loaded %d 1h bars (%s .. %s)" % (len(bars), bars[0].ts.isoformat(), bars[-1].ts.isoformat()))
    if bars[0].ts > VAL_ENTRY_START:
        print(
            "WARNING: data starts %s (> %s). The 2020-2025 validation window will "
            "be empty/partial -- refresh data via the Actions workflow first."
            % (bars[0].ts.isoformat(), VAL_ENTRY_START.isoformat())
        )

    signals, ind_data = strategy_mod.generate_signals(bars, V5_PARAMS)
    atr_stop = ind_data["atr_stop"]
    htf_ema = ind_data.get("htf_ema")

    # ---------------------------------------------------------------- (a) VALIDATION
    bt = engine_mod.run_backtest(
        bars,
        signals,
        atr_stop,
        V5_PARAMS,
        VAL_ENTRY_START,
        episode_release=ind_data.get("episode_release"),
        pullback_short=ind_data.get("pullback_short"),
        entry_end=VAL_ENTRY_END,
    )
    v_trades = bt["trades"]
    v_equity = bt["equity"]
    v_inpos = bt["in_pos_flags"]

    val_m, val_trades = window_metrics(v_equity, v_trades, v_inpos, bars, VAL_ENTRY_START, VAL_ENTRY_END)
    val_summary = summarize_trades(val_trades)
    val_boot = stats_mod.bootstrap_pvalue([t["r_multiple"] for t in val_trades], n=BOOT_N, seed=42)
    val_rand = stats_mod.random_entry_test(
        bars, VAL_ENTRY_START, val_trades, V5_PARAMS, atr_stop, val_m["sharpe"],
        n=VAL_RAND_N, seed=42, eval_end=VAL_ENTRY_END,
    )

    per_year = {}
    for y in VAL_YEARS:
        ystart = datetime(y, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        yend = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        ws = max(ystart, VAL_ENTRY_START)
        we = min(yend, VAL_ENTRY_END)
        if ws > we:
            continue
        ym, _ = window_metrics(v_equity, v_trades, v_inpos, bars, ws, we)
        per_year[str(y)] = ym

    bear = bear_regime_share(bars, htf_ema)

    validation = {
        "entry_start": VAL_ENTRY_START.isoformat(),
        "entry_end": VAL_ENTRY_END.isoformat(),
        "metrics": val_m,
        "bootstrap": val_boot,
        "random_entry": val_rand,
        "trade_count": len(val_trades),
        "scaled_out": val_summary["scaled_out"],
        "leverage_capped_trades": val_summary["leverage_capped"],
        "exit_reasons": val_summary["exit_reasons"],
        "r_distribution": val_summary["r_distribution"],
        "per_year": per_year,
        "year_2022_metrics": per_year.get("2022"),
        "bear_regime_share": bear,
        "trades": val_trades,
    }

    # ---------------------------------------------------------------- (b) 2026 REFERENCE
    bt2 = engine_mod.run_backtest(
        bars,
        signals,
        atr_stop,
        V5_PARAMS,
        GATE_ENTRY_START,
        episode_release=ind_data.get("episode_release"),
        pullback_short=ind_data.get("pullback_short"),
    )
    r_trades = bt2["trades"]
    r_equity = bt2["equity"]
    r_inpos = bt2["in_pos_flags"]

    gate_m, gate_trades = window_metrics(r_equity, r_trades, r_inpos, bars, GATE_ENTRY_START)
    ytd_m, ytd_trades = window_metrics(r_equity, r_trades, r_inpos, bars, YTD_START)
    ref_boot = stats_mod.bootstrap_pvalue([t["r_multiple"] for t in gate_trades], n=BOOT_N, seed=42)
    ref_rand = stats_mod.random_entry_test(
        bars, GATE_ENTRY_START, gate_trades, V5_PARAMS, atr_stop, gate_m["sharpe"], n=REF_RAND_N, seed=42
    )
    reference = {
        "gate_entry_start": GATE_ENTRY_START.isoformat(),
        "ytd_start": YTD_START.isoformat(),
        "gate_metrics": gate_m,
        "ytd_metrics": ytd_m,
        "gate_bootstrap": ref_boot,
        "gate_random_entry": ref_rand,
        "gate_summary": summarize_trades(gate_trades),
        "ytd_summary": summarize_trades(ytd_trades),
        "trades": r_trades,
    }

    # ---------------------------------------------------------------- REPORT
    print("\n=== VALIDATION window %s .. %s ===" % (VAL_ENTRY_START.date(), VAL_ENTRY_END.date()))
    hdr = "%-16s %6s %10s %8s %8s %7s %8s %8s %9s %9s" % (
        "window", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF", "boot_p", "rand_p",
    )
    print(hdr)
    print("-" * len(hdr))
    print(_row("validation", val_m, " %9s %9s" % (_fmt(val_boot["p_value"], "{:.4f}"), _fmt(val_rand["p_value"], "{:.4f}"))))

    print("\nPer-calendar-year metrics:")
    yh = "%-16s %6s %10s %8s %8s %7s %8s %8s" % (
        "year", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF",
    )
    print(yh)
    print("-" * len(yh))
    for y in VAL_YEARS:
        if str(y) in per_year:
            print(_row(str(y), per_year[str(y)]))

    print("\nBear-regime share (close < daily EMA50) per year:")
    for y, s in bear.items():
        print("  %s: %.4f" % (y, s))
    print("\n2022 full-year (longest bear):", per_year.get("2022"))
    print("Leverage-cap-binding trades (validation):", val_summary["leverage_capped"])
    print("Validation exit reasons:", val_summary["exit_reasons"], "scaled_out:", val_summary["scaled_out"])
    print("Validation significance: boot_p=%s rand_p=%s (actual_sharpe=%s, rand_mean=%s, rand_p95=%s, n_trades=%d)" % (
        _fmt(val_boot["p_value"], "{:.4f}"), _fmt(val_rand["p_value"], "{:.4f}"),
        _fmt(val_rand["actual_sharpe"], "{:.3f}"), _fmt(val_rand["random_mean_sharpe"], "{:.3f}"),
        _fmt(val_rand["random_p95_sharpe"], "{:.3f}"), val_boot["n_trades"],
    ))
    print("Validation R distribution (sorted, 2dp):", val_summary["r_distribution"])

    print("\n=== 2026 REFERENCE (frozen v4a) ===")
    print(hdr)
    print("-" * len(hdr))
    print(_row("gate(>=11-20)", gate_m, " %9s %9s" % (_fmt(ref_boot["p_value"], "{:.4f}"), _fmt(ref_rand["p_value"], "{:.4f}"))))
    print(_row("YTD(>=01-01)", ytd_m, " %9s %9s" % ("--", "--")))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_first_ts": bars[0].ts.isoformat(),
        "data_last_ts": bars[-1].ts.isoformat(),
        "n_bars": len(bars),
        "frozen_params": V5_PARAMS,
        "bootstrap_n": BOOT_N,
        "validation_random_entry_n": VAL_RAND_N,
        "reference_random_entry_n": REF_RAND_N,
        "validation": validation,
        "reference_2026": reference,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)
    print("\nSaved %s" % RESULTS_PATH)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
