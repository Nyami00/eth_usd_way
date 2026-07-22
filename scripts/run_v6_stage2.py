#!/usr/bin/env python3
"""v6 Stage 2 -- numeric refinement of the two winning families.

Families (from Stage 1): A = both_ttm_cf1, B = long_ttm_cf1.
All ttm mode, confirm_bars=1, scale_out ON (fixed), regime filter on.

REGISTERED DEVIATIONS from the spec grid (decided before seeing Stage-2 results):
  * added ema_slope_filter {off,on} dimension
  * fixed scale_out = ON
  * dropped time_stop 24 (kept {off,12}) and min_squeeze 12 (kept {3,4,6,8})
    -- runtime.

Grid per family:
  kc_mult {1.25,1.5,1.75} x min_squeeze {3,4,6,8} x release_window {2,3,5}
  x htf_ema_days {30,50,100} x stop_atr {1.5,2.0,2.5,3.0,3.5}
  x trail_atr {2.0,3.0,4.0,5.0} x time_stop {off,12} x ema_slope_filter {off,on}
  = 3*4*3*3*5*4*2*2 = 8640 per family, 17,280 total.

DESIGN DISCIPLINE: entries confined to 2020-03-01 .. 2025-09-30 (entry cutoff +
window-end force-close); metrics/per-year confined to the same window. Nothing
after 2025-09-30 is touched. Significance tests (design window) run only for the
top-6 by plateau score.

Outputs results/v6_stage2.json (params + metrics only; no trade lists).

Usage: python3 scripts/run_v6_stage2.py [--limit N]
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
from backtest import indicators as ind
from backtest import metrics as metrics_mod
from backtest import stats as stats_mod
from backtest import strategy as strategy_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_1H = os.path.join(ROOT, "data", "eth_usd_1h.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "v6_stage2.json")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "v6_stage2_progress.txt")

DESIGN_START = datetime(2020, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
DESIGN_END = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
DESIGN_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

FIXED = {
    "bb_n": 20, "bb_k": 2.0, "squeeze_mode": "ttm", "confirm_bars": 1,
    "trend_ema": 0, "htf_trend": True,
    "momentum_filter": False, "volume_filter": False, "vol_mult": 1.3,
    "mom_n": 20, "vol_sma_n": 20, "bw_lookback": 120, "bw_q": 20.0,
    "atr_n": 14, "flip_on_opposite": False, "fee": 0.0005, "slip": 0.0002,
    "scale_out": True, "reentry_enabled": False,
}

FAMILIES = [("both", True, True), ("long", True, False)]  # (label, allow_long, allow_short)
KC_MULTS = [1.25, 1.5, 1.75]
MIN_SQUEEZE = [3, 4, 6, 8]
RELEASE_WINDOWS = [2, 3, 5]
HTF_DAYS = [30, 50, 100]
STOP_ATRS = [1.5, 2.0, 2.5, 3.0, 3.5]
TRAIL_ATRS = [2.0, 3.0, 4.0, 5.0]
TIME_STOPS = [0, 12]
SLOPES = [False, True]

NUMERIC_DIMS = {
    "kc_mult": KC_MULTS,
    "min_squeeze_bars": MIN_SQUEEZE,
    "release_window": RELEASE_WINDOWS,
    "htf_ema_days": HTF_DAYS,
    "stop_atr": STOP_ATRS,
    "trail_atr": TRAIL_ATRS,
}

BOOT_N = 10000
RAND_N = 1000


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


# --------------------------------------------------------------------------- #
# Indicator cache
# --------------------------------------------------------------------------- #

def build_cache(bars):
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    n = len(bars)
    bb_mid, bb_up, bb_lo = ind.bollinger(closes, 20, 2.0)
    bw = ind.bandwidth(bb_up, bb_lo, bb_mid)
    ema20 = ind.ema(closes, 20)
    atr20 = ind.atr(highs, lows, closes, 20)
    kc = {}
    for m in KC_MULTS:
        up = [None] * n
        lo = [None] * n
        for i in range(n):
            if ema20[i] is not None and atr20[i] is not None:
                up[i] = ema20[i] + m * atr20[i]
                lo[i] = ema20[i] - m * atr20[i]
        kc[m] = (up, lo)
    atr14 = ind.atr(highs, lows, closes, 14)
    htf_level = {d: ind.daily_ema_ffill(bars, d) for d in HTF_DAYS}
    htf_slope = {d: ind.daily_ema_slope_ffill(bars, d) for d in HTF_DAYS}
    return {
        "bb_mid": bb_mid, "bb_up": bb_up, "bb_lo": bb_lo, "bw": bw,
        "kc": kc, "atr14": atr14, "htf_level": htf_level, "htf_slope": htf_slope,
    }


def ind_data_for(cache, kc_mult, htf_days):
    up, lo = cache["kc"][kc_mult]
    return {
        "bb_mid": cache["bb_mid"], "bb_up": cache["bb_up"], "bb_lo": cache["bb_lo"],
        "bw": cache["bw"], "kc_up": up, "kc_lo": lo,
        "ema_trend": None, "osc": None, "sma_vol": None,
        "htf_ema": cache["htf_level"][htf_days], "htf_ema_slope": cache["htf_slope"][htf_days],
        "atr_stop": cache["atr14"],
    }


# --------------------------------------------------------------------------- #
# Window index structures (precomputed once on the truncated bars)
# --------------------------------------------------------------------------- #

def build_window_index(bars_eng):
    n = len(bars_eng)
    lo = 0
    while lo < n and bars_eng[lo].ts < DESIGN_START:
        lo += 1
    hi = n - 1  # bars_eng is already truncated at DESIGN_END
    # UTC day-close indices within [lo, hi]
    day_close = []
    for i in range(lo, hi + 1):
        if i == hi or bars_eng[i + 1].ts.date() != bars_eng[i].ts.date():
            day_close.append(i)
    # per-year (first, last) indices within the window
    year_bounds = {}
    for y in DESIGN_YEARS:
        ys = datetime(y, 1, 1, tzinfo=timezone.utc)
        ye = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        ws = max(ys, DESIGN_START)
        we = min(ye, DESIGN_END)
        if ws > we:
            year_bounds[y] = (None, None)
            continue
        a = None
        b = None
        for i in range(lo, hi + 1):
            ts = bars_eng[i].ts
            if a is None and ts >= ws:
                a = i
            if ts <= we:
                b = i
        year_bounds[y] = (a, b)
    return lo, hi, day_close, year_bounds


def fast_metrics(eqvals, trades, lo, hi, day_close, year_bounds):
    eq_lo = eqvals[lo]
    eq_hi = eqvals[hi]
    net = (eq_hi / eq_lo - 1.0) * 100.0 if eq_lo else None
    peak = eqvals[lo]
    maxdd = 0.0
    for i in range(lo, hi + 1):
        v = eqvals[i]
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < maxdd:
                maxdd = dd
    maxdd_pct = -maxdd * 100.0
    daily = [eqvals[i] for i in day_close]
    sharpe = metrics_mod.sharpe_from_daily(daily)
    nt = len(trades)
    if nt:
        pnls = [t["pnl"] for t in trades]
        rs = [t["r_multiple"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = -sum(p for p in pnls if p < 0)
        win = wins / nt * 100.0
        avgr = sum(rs) / nt
        pf = (gp / gl) if gl > 0 else None
    else:
        win = avgr = pf = None
    yr = {}
    for y, (a, b) in year_bounds.items():
        if a is None:
            continue
        s = eqvals[a]
        e = eqvals[b]
        yr[str(y)] = (e / s - 1.0) * 100.0 if s else None
    pos_years = sum(1 for v in yr.values() if v is not None and v > 0)
    return {
        "trades": nt, "net_return_pct": net, "sharpe": sharpe, "max_dd_pct": maxdd_pct,
        "win_rate": win, "avg_r": avgr, "profit_factor": pf,
        "per_year_net_return_pct": yr, "positive_years": pos_years,
    }


def make_params(family, kc_mult, ms, rw, htf_days, stop_atr, trail_atr, time_stop, slope, al, ash):
    p = dict(FIXED)
    p.update({
        "kc_mult": kc_mult, "min_squeeze_bars": ms, "release_window": rw,
        "htf_ema_n": htf_days, "stop_atr": stop_atr, "trail_atr": trail_atr,
        "time_stop_bars": time_stop, "ema_slope_filter": slope,
        "allow_long": al, "allow_short": ash,
    })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N configs (0=all)")
    args = ap.parse_args()

    t0 = time.time()
    bars = data_mod.load_bars(DATA_1H)
    # Truncate the engine's bar range at the design-window end (nothing after is used).
    hi_full = len(bars) - 1
    while hi_full >= 0 and bars[hi_full].ts > DESIGN_END:
        hi_full -= 1
    bars_eng = bars[: hi_full + 1]
    atr_full = None  # set after cache

    cache = build_cache(bars)  # cache over FULL bars (daily EMAs need all days)
    atr14_eng = cache["atr14"][: hi_full + 1]
    lo, hi, day_close, year_bounds = build_window_index(bars_eng)

    print("bars=%d bars_eng=%d design[%d..%d] days=%d" % (len(bars), len(bars_eng), lo, hi, len(day_close)))
    sys.stdout.flush()

    results = []
    sharpe_by_key = {}
    total = 2 * len(KC_MULTS) * len(MIN_SQUEEZE) * len(RELEASE_WINDOWS) * len(HTF_DAYS) * len(STOP_ATRS) * len(TRAIL_ATRS) * len(TIME_STOPS) * len(SLOPES)
    done = 0
    limit = args.limit if args.limit > 0 else total

    for (fam, al, ash) in FAMILIES:
        # signal-distinct loop
        for kc_mult in KC_MULTS:
            for htf_days in HTF_DAYS:
                idata = ind_data_for(cache, kc_mult, htf_days)
                for ms in MIN_SQUEEZE:
                    for rw in RELEASE_WINDOWS:
                        for slope in SLOPES:
                            # signals reused across (stop, trail, time_stop)
                            sp = make_params(fam, kc_mult, ms, rw, htf_days, 2.0, 3.0, 0, slope, al, ash)
                            id_copy = dict(idata)
                            signals, _ = strategy_mod.generate_signals(bars, sp, ind_data=id_copy)
                            signals_eng = signals[: hi_full + 1]
                            for stop_atr in STOP_ATRS:
                                for trail_atr in TRAIL_ATRS:
                                    for ts_bars in TIME_STOPS:
                                        params = make_params(
                                            fam, kc_mult, ms, rw, htf_days, stop_atr,
                                            trail_atr, ts_bars, slope, al, ash,
                                        )
                                        bt = engine_mod.run_backtest(
                                            bars_eng, signals_eng, atr14_eng, params,
                                            DESIGN_START, entry_end=DESIGN_END,
                                        )
                                        eqvals = [e for _, e in bt["equity"]]
                                        m = fast_metrics(eqvals, bt["trades"], lo, hi, day_close, year_bounds)
                                        rec = {
                                            "family": fam, "kc_mult": kc_mult, "min_squeeze_bars": ms,
                                            "release_window": rw, "htf_ema_days": htf_days,
                                            "stop_atr": stop_atr, "trail_atr": trail_atr,
                                            "time_stop_bars": ts_bars, "ema_slope_filter": slope,
                                        }
                                        rec.update(m)
                                        results.append(rec)
                                        key = (fam, kc_mult, ms, rw, htf_days, stop_atr, trail_atr, ts_bars, slope)
                                        sharpe_by_key[key] = m["sharpe"] if m["sharpe"] is not None else 0.0
                                        done += 1
                                        if done % 2000 == 0:
                                            dt = time.time() - t0
                                            eta = dt / done * (total - done)
                                            msg = "%d/%d (%.1f%%) elapsed=%.0fs eta=%.0fs" % (
                                                done, total, 100.0 * done / total, dt, eta)
                                            print(msg)
                                            sys.stdout.flush()
                                            with open(PROGRESS_PATH, "w") as pf:
                                                pf.write(msg + "\n")
                                        if done >= limit:
                                            break
                                    if done >= limit:
                                        break
                                if done >= limit:
                                    break
                            if done >= limit:
                                break
                        if done >= limit:
                            break
                    if done >= limit:
                        break
                if done >= limit:
                    break
            if done >= limit:
                break
        if done >= limit:
            break

    print("computed %d configs in %.1fs" % (len(results), time.time() - t0))
    sys.stdout.flush()

    # ---- plateau scores ----
    def key_of(r, overrides=None):
        d = dict(
            family=r["family"], kc_mult=r["kc_mult"], min_squeeze_bars=r["min_squeeze_bars"],
            release_window=r["release_window"], htf_ema_days=r["htf_ema_days"],
            stop_atr=r["stop_atr"], trail_atr=r["trail_atr"], time_stop_bars=r["time_stop_bars"],
            ema_slope_filter=r["ema_slope_filter"],
        )
        if overrides:
            d.update(overrides)
        return (d["family"], d["kc_mult"], d["min_squeeze_bars"], d["release_window"],
                d["htf_ema_days"], d["stop_atr"], d["trail_atr"], d["time_stop_bars"],
                d["ema_slope_filter"])

    def plateau(r):
        vals = [sharpe_by_key.get(key_of(r), 0.0)]
        for dim, grid in NUMERIC_DIMS.items():
            cur = r[dim]
            idx = grid.index(cur)
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(grid):
                    k = key_of(r, {dim: grid[j]})
                    if k in sharpe_by_key:
                        vals.append(sharpe_by_key[k])
        return sum(vals) / len(vals)

    for r in results:
        r["plateau_score"] = plateau(r)

    # ---- candidate selection ----
    def qualifies(r):
        return (
            r["trades"] >= 100
            and r["sharpe"] is not None and r["sharpe"] >= 1.0
            and r["positive_years"] >= 4
        )

    candidates = [r for r in results if qualifies(r)]
    candidates.sort(key=lambda r: r["plateau_score"], reverse=True)
    qual_by_family = {}
    slope_on_qual = 0
    for r in candidates:
        qual_by_family[r["family"]] = qual_by_family.get(r["family"], 0) + 1
        if r["ema_slope_filter"]:
            slope_on_qual += 1

    # ---- significance for top-6 by plateau ----
    top6 = candidates[:6]
    for r in top6:
        al = True
        ash = r["family"] != "long"
        params = make_params(
            r["family"], r["kc_mult"], r["min_squeeze_bars"], r["release_window"],
            r["htf_ema_days"], r["stop_atr"], r["trail_atr"], r["time_stop_bars"],
            r["ema_slope_filter"], al, ash,
        )
        idata = dict(ind_data_for(cache, r["kc_mult"], r["htf_ema_days"]))
        signals, _ = strategy_mod.generate_signals(bars, params, ind_data=idata)
        bt = engine_mod.run_backtest(
            bars_eng, signals[: hi_full + 1], atr14_eng, params, DESIGN_START, entry_end=DESIGN_END
        )
        tr = bt["trades"]
        boot = stats_mod.bootstrap_pvalue([t["r_multiple"] for t in tr], n=BOOT_N, seed=42)
        rand = stats_mod.random_entry_test(
            bars_eng, DESIGN_START, tr, params, atr14_eng, r["sharpe"],
            n=RAND_N, seed=42, eval_end=DESIGN_END,
        )
        r["bootstrap"] = boot
        r["random_entry"] = rand

    # ---- save ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": 2,
        "design_start": DESIGN_START.isoformat(),
        "design_end": DESIGN_END.isoformat(),
        "n_configs": len(results),
        "families": [f[0] for f in FAMILIES],
        "registered_deviations": {
            "added": "ema_slope_filter {off,on} dimension",
            "fixed": "scale_out = ON",
            "dropped": ["time_stop 24 (kept off/12)", "min_squeeze 12 (kept 3/4/6/8)"],
            "reason": "runtime; decided before seeing stage-2 results",
        },
        "grid": {
            "kc_mult": KC_MULTS, "min_squeeze_bars": MIN_SQUEEZE, "release_window": RELEASE_WINDOWS,
            "htf_ema_days": HTF_DAYS, "stop_atr": STOP_ATRS, "trail_atr": TRAIL_ATRS,
            "time_stop_bars": TIME_STOPS, "ema_slope_filter": SLOPES,
        },
        "candidate_criteria": {"trades>=": 100, "sharpe>=": 1.0, "positive_years>=": 4},
        "n_qualifying": len(candidates),
        "qualifying_by_family": qual_by_family,
        "qualifying_slope_on": slope_on_qual,
        "top_candidates_by_plateau": candidates[:30],
        "top6_significance": [
            {k: r.get(k) for k in (
                "family", "kc_mult", "min_squeeze_bars", "release_window", "htf_ema_days",
                "stop_atr", "trail_atr", "time_stop_bars", "ema_slope_filter",
                "trades", "sharpe", "max_dd_pct", "positive_years", "plateau_score",
                "per_year_net_return_pct", "bootstrap", "random_entry",
            )}
            for r in top6
        ],
        "all_configs": results,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)

    # ---- report ----
    print("\n=== v6 Stage 2 ===")
    print("configs: %d | qualifying (trades>=100, Sharpe>=1.0, +yrs>=4): %d" % (len(results), len(candidates)))
    print("qualifying by family:", qual_by_family)
    print("qualifying with ema_slope_filter=ON: %d / %d" % (slope_on_qual, len(candidates)))

    print("\nTop-15 candidates by plateau score:")
    hh = "%-4s kc%-4s ms%-2s rw%-1s htf%-3s stop%-3s trail%-3s tstop%-2s slope%-1s | %5s %7s %7s %4s %8s"
    print(hh % ("fam", "", "", "", "", "", "", "", "", "trd", "Sharpe", "maxDD", "+yr", "plateau"))
    for r in candidates[:15]:
        print(
            "%-4s kc%-4s ms%-2d rw%-1d htf%-3d stop%-3.1f trail%-3.1f tstop%-2d slope%-1d | %5d %7.3f %7.2f %4d %8.3f"
            % (
                r["family"], r["kc_mult"], r["min_squeeze_bars"], r["release_window"], r["htf_ema_days"],
                r["stop_atr"], r["trail_atr"], r["time_stop_bars"], int(r["ema_slope_filter"]),
                r["trades"], r["sharpe"], r["max_dd_pct"], r["positive_years"], r["plateau_score"],
            )
        )

    print("\nSignificance (top-6 by plateau, design window):")
    for r in top6:
        b = r["bootstrap"]
        rd = r["random_entry"]
        print(
            "  %s kc%.2f ms%d rw%d htf%d stop%.1f trail%.1f tstop%d slope%d | Sharpe=%.3f trades=%d boot_p=%.4f rand_p=%.4f"
            % (
                r["family"], r["kc_mult"], r["min_squeeze_bars"], r["release_window"], r["htf_ema_days"],
                r["stop_atr"], r["trail_atr"], r["time_stop_bars"], int(r["ema_slope_filter"]),
                r["sharpe"], r["trades"], b["p_value"], rd["p_value"],
            )
        )
        yr = r["per_year_net_return_pct"]
        print("      per-year net%%: " + ", ".join("%s=%.1f" % (y, yr[y]) for y in sorted(yr)))
        print("      bear-year detail: 2022=%s 2025=%s" % (
            ("%.1f" % yr.get("2022")) if yr.get("2022") is not None else "n/a",
            ("%.1f" % yr.get("2025")) if yr.get("2025") is not None else "n/a"))

    print("\nSaved %s" % RESULTS_PATH)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
