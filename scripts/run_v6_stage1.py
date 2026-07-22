#!/usr/bin/env python3
"""v6 Stage 1 -- coarse structural-family search (docs/strategy_v6_spec.md).

DESIGN DISCIPLINE: every evaluation is confined to the design window
2020-03-01 .. 2025-09-30 (entry cutoff + window-end force-close). Nothing after
2025-09-30 is ever touched here. The 2026 holdout is evaluated only later, on
explicit instruction, for at most 3 pre-named configs.

Fixed: BB 20/2.0, ATR14, daily-EMA50 regime filter, stop 2.5, trail 3.0,
scale_out on (1.5R/50%), time_stop 12, fee 0.0005 / slip 0.0002, 3% risk, 5x cap.

Grid dimensions (the 5 enumerated in the spec):
  direction        : {both (with-regime), short, long}   (3)
  squeeze_mode     : {ttm (kc 1.5), pctile (L120, q20)}  (2)
  confirm_bars     : {0, 1}                                (2)
  min_squeeze_bars : {4, 8}                                (2)
  release_window   : {3, 5}                                (2)
=> 3*2*2*2*2 = 48 configs. (The spec's "~96/~100" is an approximation; the
explicit dimension product is 48.)

Indicators that do not vary across the grid (BB/KC/ATR/daily-EMA/pctile) are
computed once and shared. No significance tests at this stage -- metrics only.

Outputs results/v6_stage1.json.

Usage: python3 scripts/run_v6_stage1.py
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
from backtest import indicators as ind
from backtest import metrics as metrics_mod
from backtest import strategy as strategy_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_1H = os.path.join(ROOT, "data", "eth_usd_1h.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "v6_stage1.json")

DESIGN_START = datetime(2020, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
DESIGN_END = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
DESIGN_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

FIXED = {
    "bb_n": 20,
    "bb_k": 2.0,
    "bw_lookback": 120,
    "bw_q": 20.0,
    "kc_mult": 1.5,
    "trend_ema": 0,
    "htf_trend": True,
    "htf_ema_n": 50,
    "momentum_filter": False,
    "volume_filter": False,
    "vol_mult": 1.3,
    "mom_n": 20,
    "vol_sma_n": 20,
    "atr_n": 14,
    "stop_atr": 2.5,
    "trail_atr": 3.0,
    "flip_on_opposite": False,
    "fee": 0.0005,
    "slip": 0.0002,
    "scale_out": True,
    "time_stop_bars": 12,
    "reentry_enabled": False,
}

DIRECTIONS = [
    ("both", True, True),   # with-regime both (long above EMA, short below)
    ("short", False, True),
    ("long", True, False),
]
MODES = ["ttm", "pctile"]
CONFIRMS = [0, 1]
MIN_SQUEEZE = [4, 8]
RELEASE_WINDOWS = [3, 5]


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


def build_cache(bars):
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    bb_mid, bb_up, bb_lo = ind.bollinger(closes, FIXED["bb_n"], FIXED["bb_k"])
    bw = ind.bandwidth(bb_up, bb_lo, bb_mid)
    _, kc_up, kc_lo = ind.keltner(highs, lows, closes, FIXED["bb_n"], FIXED["kc_mult"])
    bw_pct = ind.bw_percentile(bw, FIXED["bw_lookback"])
    htf_ema = ind.daily_ema_ffill(bars, FIXED["htf_ema_n"])
    atr_stop = ind.atr(highs, lows, closes, FIXED["atr_n"])
    return {
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "bb_lo": bb_lo,
        "bw": bw,
        "kc_up": kc_up,
        "kc_lo": kc_lo,
        "bw_pct": bw_pct,
        "ema_trend": None,
        "osc": None,
        "sma_vol": None,
        "htf_ema": htf_ema,
        "atr_stop": atr_stop,
    }


def window_metrics(equity, trades, in_pos, bars, ws, we):
    weq = [(ts, eq) for ts, eq in equity if ws <= ts <= we]
    wtr = [t for t in trades if ws <= _parse(t["entry_ts"]) <= we]
    widx = [i for i, b in enumerate(bars) if ws <= b.ts <= we]
    nw = len(widx)
    expo = (sum(1 for i in widx if in_pos[i]) / nw) if nw else 0.0
    return metrics_mod.compute_metrics(weq, wtr, expo)


def per_year_returns(equity, trades, in_pos, bars):
    out = {}
    for y in DESIGN_YEARS:
        ys = datetime(y, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ye = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        ws = max(ys, DESIGN_START)
        we = min(ye, DESIGN_END)
        if ws > we:
            continue
        m = window_metrics(equity, trades, in_pos, bars, ws, we)
        out[str(y)] = m["net_return_pct"]
    return out


def build_grid():
    variants = []
    for dlabel, al, ash in DIRECTIONS:
        for mode in MODES:
            for cf in CONFIRMS:
                for ms in MIN_SQUEEZE:
                    for rw in RELEASE_WINDOWS:
                        params = dict(FIXED)
                        params["squeeze_mode"] = mode
                        params["allow_long"] = al
                        params["allow_short"] = ash
                        params["confirm_bars"] = cf
                        params["min_squeeze_bars"] = ms
                        params["release_window"] = rw
                        label = "%s_%s_cf%d_ms%d_rw%d" % (dlabel, mode, cf, ms, rw)
                        variants.append((label, dlabel, mode, cf, ms, rw, params))
    return variants


def main():
    t0 = time.time()
    bars = data_mod.load_bars(DATA_1H)
    print("Loaded %d 1h bars (%s .. %s)" % (len(bars), bars[0].ts.isoformat(), bars[-1].ts.isoformat()))
    print("Design window: %s .. %s (entries confined here; nothing after is touched)\n"
          % (DESIGN_START.isoformat(), DESIGN_END.isoformat()))

    cache = build_cache(bars)
    variants = build_grid()

    results = []
    for (label, dlabel, mode, cf, ms, rw, params) in variants:
        ind_data = dict(cache)  # shallow copy; generate_signals only adds keys
        signals, _ = strategy_mod.generate_signals(bars, params, ind_data=ind_data)
        bt = engine_mod.run_backtest(
            bars, signals, cache["atr_stop"], params, DESIGN_START, entry_end=DESIGN_END
        )
        trades = bt["trades"]
        equity = bt["equity"]
        in_pos = bt["in_pos_flags"]
        m = window_metrics(equity, trades, in_pos, bars, DESIGN_START, DESIGN_END)
        yr = per_year_returns(equity, trades, in_pos, bars)
        pos_years = sum(1 for v in yr.values() if v is not None and v > 0)
        n_long = sum(1 for t in trades if t["side"] == "long")
        n_short = sum(1 for t in trades if t["side"] == "short")
        results.append(
            {
                "config": label,
                "direction": dlabel,
                "squeeze_mode": mode,
                "confirm_bars": cf,
                "min_squeeze_bars": ms,
                "release_window": rw,
                "metrics": m,
                "per_year_net_return_pct": yr,
                "positive_years": pos_years,
                "trade_count_by_side": {"long": n_long, "short": n_short},
                "params": params,
            }
        )

    # Sort by design-window Sharpe descending (None -> bottom).
    def sharpe_key(r):
        s = r["metrics"]["sharpe"]
        return s if s is not None else -1e9

    results.sort(key=sharpe_key, reverse=True)

    # ---- report table ----
    hdr = "%-26s %6s %10s %8s %8s %7s %8s %8s %7s" % (
        "config", "trades", "netRet%", "Sharpe", "maxDD%", "win%", "avgR", "PF", "+yrs",
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        m = r["metrics"]
        print(
            "%-26s %6d %10s %8s %8s %7s %8s %8s %7d"
            % (
                r["config"], m["trades"],
                _fmt(m["net_return_pct"], "{:.2f}"), _fmt(m["sharpe"], "{:.3f}"),
                _fmt(m["max_dd_pct"], "{:.2f}"), _fmt(m["win_rate"], "{:.1f}"),
                _fmt(m["avg_r"], "{:.3f}"), _fmt(m["profit_factor"], "{:.3f}"),
                r["positive_years"],
            )
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": 1,
        "design_start": DESIGN_START.isoformat(),
        "design_end": DESIGN_END.isoformat(),
        "data_first_ts": bars[0].ts.isoformat(),
        "data_last_ts": bars[-1].ts.isoformat(),
        "n_configs": len(results),
        "fixed_params": FIXED,
        "grid": {
            "direction": [d[0] for d in DIRECTIONS],
            "squeeze_mode": MODES,
            "confirm_bars": CONFIRMS,
            "min_squeeze_bars": MIN_SQUEEZE,
            "release_window": RELEASE_WINDOWS,
        },
        "configs_sorted_by_sharpe": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)

    dt = time.time() - t0
    print("\n%d configs; saved %s" % (len(results), RESULTS_PATH))
    print("Total runtime: %.2fs (%.3fs/config)" % (dt, dt / max(1, len(results))))


if __name__ == "__main__":
    main()
