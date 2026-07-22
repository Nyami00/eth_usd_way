#!/usr/bin/env python3
"""v6 HOLDOUT evaluation + Monte-Carlo robustness suite.

Sequential holdout test C1 -> C2 -> C3, stop at the first PASS. PASS =
YTD (2026-01-01..end) Sharpe >= 1.5 AND YTD maxDD < 25% AND YTD trades >= 10.
For the first passing config, run the registered MC suite (optimization_mc_spec
section 2), cost-doubled sensitivity, and descriptive holdout significance.

All configs are frozen from Stage 2 (ttm, confirm_bars=1, scale_out on,
BB 20/2.0, ATR14, fee 0.0005/slip 0.0002, 3% risk, 5x cap).

Outputs results/v6_holdout.json.
"""

import json
import math
import os
import random
import sys
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "v6_holdout.json")
STAGE2_PATH = os.path.join(RESULTS_DIR, "v6_stage2.json")

DESIGN_START = datetime(2020, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
DESIGN_END = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
GATE_ENTRY_START = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
YTD_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

SQRT365 = math.sqrt(365.0)

FIXED = {
    "bb_n": 20, "bb_k": 2.0, "squeeze_mode": "ttm", "confirm_bars": 1,
    "trend_ema": 0, "htf_trend": True,
    "momentum_filter": False, "volume_filter": False, "vol_mult": 1.3,
    "mom_n": 20, "vol_sma_n": 20, "bw_lookback": 120, "bw_q": 20.0,
    "atr_n": 14, "flip_on_opposite": False, "fee": 0.0005, "slip": 0.0002,
    "scale_out": True, "reentry_enabled": False,
}

# Final named candidates (plateau order + both-constraint).
CANDIDATES = [
    {"name": "C1", "family": "long", "kc_mult": 1.75, "min_squeeze_bars": 6, "release_window": 5,
     "htf_ema_days": 100, "stop_atr": 2.0, "trail_atr": 4.0, "time_stop_bars": 0, "ema_slope_filter": False},
    {"name": "C2", "family": "long", "kc_mult": 1.75, "min_squeeze_bars": 8, "release_window": 5,
     "htf_ema_days": 100, "stop_atr": 1.5, "trail_atr": 4.0, "time_stop_bars": 0, "ema_slope_filter": False},
    {"name": "C3", "family": "both", "kc_mult": 1.50, "min_squeeze_bars": 4, "release_window": 3,
     "htf_ema_days": 30, "stop_atr": 2.0, "trail_atr": 4.0, "time_stop_bars": 12, "ema_slope_filter": True},
]

# Stage-2 grid axes (for MC4 neighbor lookup).
GRID = {
    "kc_mult": [1.25, 1.5, 1.75], "min_squeeze_bars": [3, 4, 6, 8],
    "release_window": [2, 3, 5], "htf_ema_days": [30, 50, 100],
    "stop_atr": [1.5, 2.0, 2.5, 3.0, 3.5], "trail_atr": [2.0, 3.0, 4.0, 5.0],
    "time_stop_bars": [0, 12], "ema_slope_filter": [False, True],
}


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _parse(ts):
    return datetime.fromisoformat(ts)


def make_params(c, fee=None, slip=None):
    p = dict(FIXED)
    p.update({
        "kc_mult": c["kc_mult"], "min_squeeze_bars": c["min_squeeze_bars"],
        "release_window": c["release_window"], "htf_ema_n": c["htf_ema_days"],
        "stop_atr": c["stop_atr"], "trail_atr": c["trail_atr"],
        "time_stop_bars": c["time_stop_bars"], "ema_slope_filter": c["ema_slope_filter"],
        "allow_long": True, "allow_short": c["family"] != "long",
    })
    if fee is not None:
        p["fee"] = fee
    if slip is not None:
        p["slip"] = slip
    return p


def window_metrics(equity, trades, in_pos, bars, ws, we):
    weq = [(ts, eq) for ts, eq in equity if ws <= ts <= we]
    wtr = [t for t in trades if ws <= _parse(t["entry_ts"]) <= we]
    widx = [i for i, b in enumerate(bars) if ws <= b.ts <= we]
    nw = len(widx)
    expo = (sum(1 for i in widx if in_pos[i]) / nw) if nw else 0.0
    m = metrics_mod.compute_metrics(weq, wtr, expo)
    nl = sum(1 for t in wtr if t["side"] == "long")
    nsh = sum(1 for t in wtr if t["side"] == "short")
    m["by_side"] = {"long": nl, "short": nsh}
    return m, wtr, weq


def window_daily_closes(equity, ws, we):
    weq = [(ts, eq) for ts, eq in equity if ws <= ts <= we]
    return metrics_mod.daily_closes(weq)


def monthly_returns(equity, ws, we):
    weq = [(ts, eq) for ts, eq in equity if ws <= ts <= we]
    out = {}
    for ts, eq in weq:
        key = "%04d-%02d" % (ts.year, ts.month)
        if key not in out:
            out[key] = [eq, eq]  # first, last
        else:
            out[key][1] = eq
    return {k: (v[1] / v[0] - 1.0) * 100.0 if v[0] else None for k, v in out.items()}


def sharpe_from_returns(rets):
    if len(rets) < 2:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return m / sd * SQRT365 if sd > 0 else 0.0


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


# ---- MC tests --------------------------------------------------------------

def mc1_block_bootstrap(daily_rets, n=10000, block=5, seed=42):
    L = len(daily_rets)
    if L < 3:
        return None
    rng = random.Random(seed)
    p = 1.0 / block
    sharpes = []
    for _ in range(n):
        series = []
        idx = rng.randrange(L)
        for _ in range(L):
            series.append(daily_rets[idx])
            if rng.random() < p:
                idx = rng.randrange(L)
            else:
                idx = (idx + 1) % L
        sharpes.append(sharpe_from_returns(series))
    sharpes.sort()
    return {
        "p5_sharpe": pct(sharpes, 5), "median_sharpe": pct(sharpes, 50),
        "p_sharpe_gt0": sum(1 for s in sharpes if s > 0) / n,
        "p_sharpe_gt1_5": sum(1 for s in sharpes if s > 1.5) / n, "n": n, "block": block,
    }


def mc2_trade_bootstrap(rmults, n=10000, risk=0.03, seed=42):
    m = len(rmults)
    if m == 0:
        return None
    rng = random.Random(seed)
    totals = []
    for _ in range(n):
        eq = 1.0
        for R in rng.choices(rmults, k=m):
            eq *= (1.0 + risk * R)
        totals.append((eq - 1.0) * 100.0)
    totals.sort()
    return {
        "p5_total_return_pct": pct(totals, 5), "median_total_return_pct": pct(totals, 50),
        "p_return_gt0": sum(1 for t in totals if t > 0) / n, "n": n,
    }


def mc3_order_shuffle(rmults, n=10000, risk=0.03, seed=42):
    m = len(rmults)
    if m == 0:
        return None
    rng = random.Random(seed)
    base = list(rmults)
    mdds = []
    for _ in range(n):
        rng.shuffle(base)
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        for R in base:
            eq *= (1.0 + risk * R)
            if eq > peak:
                peak = eq
            dd = eq / peak - 1.0
            if dd < mdd:
                mdd = dd
        mdds.append(-mdd * 100.0)
    mdds.sort()
    return {"p95_maxdd_pct": pct(mdds, 95), "median_maxdd_pct": pct(mdds, 50), "n": n}


def mc4_param_perturbation(cfg):
    """One-step neighbors in the stage-2 grid, design Sharpe from v6_stage2.json."""
    s2 = json.load(open(STAGE2_PATH))
    key_dims = ["family", "kc_mult", "min_squeeze_bars", "release_window", "htf_ema_days",
                "stop_atr", "trail_atr", "time_stop_bars", "ema_slope_filter"]
    sh = {}
    for c in s2["all_configs"]:
        sh[tuple(c[d] for d in key_dims)] = c["sharpe"]

    def key(overrides=None):
        d = {k: cfg[k] for k in key_dims}
        if overrides:
            d.update(overrides)
        return tuple(d[k] for k in key_dims)

    neigh = []
    for dim in ("kc_mult", "min_squeeze_bars", "release_window", "htf_ema_days",
                "stop_atr", "trail_atr", "time_stop_bars", "ema_slope_filter"):
        grid = GRID[dim]
        idx = grid.index(cfg[dim])
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(grid):
                k = key({dim: grid[j]})
                if k in sh and sh[k] is not None:
                    neigh.append(sh[k])
    if not neigh:
        return None
    neigh.sort()
    frac_pos = sum(1 for s in neigh if s > 0) / len(neigh)
    return {
        "n_neighbors": len(neigh), "frac_sharpe_gt0": frac_pos,
        "median_sharpe": pct(neigh, 50), "neighbor_sharpes": neigh,
    }


def mc5_entry_jitter(bars, base_signals, atr, params, eval_start, entry_end, ws, we, n=500, seed=42):
    rng = random.Random(seed)
    sig_positions = [(t, base_signals[t]) for t in range(len(base_signals)) if base_signals[t] is not None]
    N = len(bars)
    sharpes = []
    for _ in range(n):
        jit = [None] * N
        for t, dirn in sig_positions:
            d = rng.randint(0, 2)
            j = t + d
            if j < N:
                jit[j] = dirn
        bt = engine_mod.run_backtest(bars, jit, atr, params, eval_start, entry_end=entry_end)
        dclose = window_daily_closes(bt["equity"], ws, we)
        sharpes.append(metrics_mod.sharpe_from_daily(dclose))
    sharpes.sort()
    return {"median_sharpe": pct(sharpes, 50), "p5_sharpe": pct(sharpes, 5),
            "p_sharpe_gt1": sum(1 for s in sharpes if s >= 1.0) / n, "n": n}


def main():
    bars = data_mod.load_bars(DATA_1H)
    last_ts = bars[-1].ts
    print("Loaded %d bars (%s .. %s)" % (len(bars), bars[0].ts.isoformat(), last_ts.isoformat()))

    tested = []
    passing = None
    for c in CANDIDATES:
        params = make_params(c)
        signals, ind_data = strategy_mod.generate_signals(bars, params)
        atr = ind_data["atr_stop"]
        bt = engine_mod.run_backtest(bars, signals, atr, params, GATE_ENTRY_START)
        m, ytd_trades, _ = window_metrics(bt["equity"], bt["trades"], bt["in_pos_flags"], bars, YTD_START, last_ts)
        mon = monthly_returns(bt["equity"], YTD_START, last_ts)
        pass_ = (
            m["sharpe"] is not None and m["sharpe"] >= 1.5
            and m["max_dd_pct"] < 25.0 and m["trades"] >= 10
        )
        rec = {"name": c["name"], "config": c, "ytd_metrics": m, "monthly_net_return_pct": mon, "passed": pass_}
        tested.append(rec)
        print("\n%s (%s) YTD: trades=%d netRet=%.2f Sharpe=%.3f maxDD=%.2f win=%.1f avgR=%.3f PF=%s side=%s -> %s"
              % (c["name"], c["family"], m["trades"], m["net_return_pct"] or 0, m["sharpe"] or 0,
                 m["max_dd_pct"] or 0, m["win_rate"] or 0, m["avg_r"] or 0,
                 ("%.3f" % m["profit_factor"]) if m["profit_factor"] else "n/a", m["by_side"],
                 "PASS" if pass_ else "FAIL"))
        print("   monthly: " + ", ".join("%s=%.1f" % (k, mon[k]) for k in sorted(mon)))
        if pass_:
            passing = (c, params, signals, atr)
            break

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design_window": [DESIGN_START.isoformat(), DESIGN_END.isoformat()],
        "gate_entry_start": GATE_ENTRY_START.isoformat(),
        "ytd_window": [YTD_START.isoformat(), last_ts.isoformat()],
        "pass_criteria": {"ytd_sharpe>=": 1.5, "ytd_maxdd<": 25.0, "ytd_trades>=": 10},
        "candidates_tested": tested,
        "first_pass": passing[0]["name"] if passing else None,
    }

    if passing is None:
        print("\nNo candidate passed. Reporting all three; no MC suite run.")
        payload["mc_suite"] = None
    else:
        c, params, signals, atr = passing
        print("\n=== MC suite for first passing config: %s ===" % c["name"])

        # Design-window simulation (entry cutoff + force close).
        bt_d = engine_mod.run_backtest(bars, signals, atr, params, DESIGN_START, entry_end=DESIGN_END)
        dm, d_trades, _ = window_metrics(bt_d["equity"], bt_d["trades"], bt_d["in_pos_flags"], bars, DESIGN_START, DESIGN_END)
        d_daily = window_daily_closes(bt_d["equity"], DESIGN_START, DESIGN_END)
        d_daily_rets = metrics_mod.daily_returns(d_daily)

        # Holdout (YTD) simulation.
        bt_h = engine_mod.run_backtest(bars, signals, atr, params, GATE_ENTRY_START)
        hm, h_trades, _ = window_metrics(bt_h["equity"], bt_h["trades"], bt_h["in_pos_flags"], bars, YTD_START, last_ts)
        h_daily = window_daily_closes(bt_h["equity"], YTD_START, last_ts)
        h_daily_rets = metrics_mod.daily_returns(h_daily)

        d_r = [t["r_multiple"] for t in d_trades]
        h_r = [t["r_multiple"] for t in h_trades]

        mc1_d = mc1_block_bootstrap(d_daily_rets, n=10000, block=5, seed=42)
        mc1_h = mc1_block_bootstrap(h_daily_rets, n=10000, block=5, seed=42)
        mc2_d = mc2_trade_bootstrap(d_r, n=10000, seed=42)
        mc2_h = mc2_trade_bootstrap(h_r, n=10000, seed=42)
        mc3_d = mc3_order_shuffle(d_r, n=10000, seed=42)
        mc3_h = mc3_order_shuffle(h_r, n=10000, seed=42)
        mc4 = mc4_param_perturbation(c)
        mc5_d = mc5_entry_jitter(bars, signals, atr, params, DESIGN_START, DESIGN_END, DESIGN_START, DESIGN_END, n=500, seed=42)
        mc5_h = mc5_entry_jitter(bars, signals, atr, params, GATE_ENTRY_START, None, YTD_START, last_ts, n=500, seed=42)

        # Cost-doubled sensitivity.
        p2 = make_params(c, fee=0.001, slip=0.0004)
        sig2, ind2 = strategy_mod.generate_signals(bars, p2)
        atr2 = ind2["atr_stop"]
        bt_d2 = engine_mod.run_backtest(bars, sig2, atr2, p2, DESIGN_START, entry_end=DESIGN_END)
        dm2, _, _ = window_metrics(bt_d2["equity"], bt_d2["trades"], bt_d2["in_pos_flags"], bars, DESIGN_START, DESIGN_END)
        bt_h2 = engine_mod.run_backtest(bars, sig2, atr2, p2, GATE_ENTRY_START)
        hm2, _, _ = window_metrics(bt_h2["equity"], bt_h2["trades"], bt_h2["in_pos_flags"], bars, YTD_START, last_ts)

        # Descriptive holdout significance.
        h_boot = stats_mod.bootstrap_pvalue(h_r, n=10000, seed=42)
        h_rand = stats_mod.random_entry_test(bars, YTD_START, h_trades, params, atr, hm["sharpe"], n=1000, seed=42)

        crit = {
            "MC1_design_p5_sharpe_gt0": (mc1_d["p5_sharpe"] > 0) if mc1_d else None,
            "MC2_design_p5_totalreturn_gt0": (mc2_d["p5_total_return_pct"] > 0) if mc2_d else None,
            "MC3_design_p95_maxdd_lt35": (mc3_d["p95_maxdd_pct"] < 35.0) if mc3_d else None,
            "MC4_frac_sharpe_gt0_ge70pct": (mc4["frac_sharpe_gt0"] >= 0.70) if mc4 else None,
            "MC4_median_sharpe_ge1_0": (mc4["median_sharpe"] >= 1.0) if mc4 else None,
            "MC5_design_median_sharpe_ge1_0": (mc5_d["median_sharpe"] >= 1.0) if mc5_d else None,
        }

        payload["mc_suite"] = {
            "config": c,
            "design_metrics": dm, "ytd_metrics": hm,
            "MC1_block_bootstrap": {"design": mc1_d, "holdout": mc1_h},
            "MC2_trade_bootstrap": {"design": mc2_d, "holdout": mc2_h},
            "MC3_order_shuffle": {"design": mc3_d, "holdout": mc3_h},
            "MC4_param_perturbation": mc4,
            "MC5_entry_jitter": {"design": mc5_d, "holdout": mc5_h},
            "cost_doubled": {"fee": 0.001, "slip": 0.0004,
                             "design_metrics": dm2, "ytd_metrics": hm2},
            "holdout_significance": {"bootstrap": h_boot, "random_entry": h_rand},
            "criteria_pass_fail": crit,
        }
        payload["equity_daily_closes"] = {"design": d_daily, "holdout": h_daily}

        # ---- print MC report ----
        print("MC1 block-bootstrap (block=5, n=10000):")
        print("  design : 5%%ile=%.3f median=%.3f P(Sh>0)=%.3f P(Sh>1.5)=%.3f  [crit 5%%ile>0: %s]"
              % (mc1_d["p5_sharpe"], mc1_d["median_sharpe"], mc1_d["p_sharpe_gt0"], mc1_d["p_sharpe_gt1_5"], crit["MC1_design_p5_sharpe_gt0"]))
        print("  holdout: 5%%ile=%.3f median=%.3f P(Sh>0)=%.3f P(Sh>1.5)=%.3f"
              % (mc1_h["p5_sharpe"], mc1_h["median_sharpe"], mc1_h["p_sharpe_gt0"], mc1_h["p_sharpe_gt1_5"]))
        print("MC2 trade-bootstrap (n=10000, 3%% compound):")
        print("  design : 5%%ile totalRet=%.2f%% median=%.2f%% P(ret>0)=%.3f  [crit 5%%ile>0: %s]"
              % (mc2_d["p5_total_return_pct"], mc2_d["median_total_return_pct"], mc2_d["p_return_gt0"], crit["MC2_design_p5_totalreturn_gt0"]))
        print("  holdout: 5%%ile totalRet=%.2f%% median=%.2f%% P(ret>0)=%.3f"
              % (mc2_h["p5_total_return_pct"], mc2_h["median_total_return_pct"], mc2_h["p_return_gt0"]))
        print("MC3 order-shuffle (n=10000):")
        print("  design : 95%%ile maxDD=%.2f%% median=%.2f%%  [crit 95%%ile<35: %s]"
              % (mc3_d["p95_maxdd_pct"], mc3_d["median_maxdd_pct"], crit["MC3_design_p95_maxdd_lt35"]))
        print("  holdout: 95%%ile maxDD=%.2f%% median=%.2f%%" % (mc3_h["p95_maxdd_pct"], mc3_h["median_maxdd_pct"]))
        print("MC4 param-perturbation (%d neighbors): frac Sh>0=%.3f median Sh=%.3f  [crit >=70%%: %s | median>=1.0: %s]"
              % (mc4["n_neighbors"], mc4["frac_sharpe_gt0"], mc4["median_sharpe"], crit["MC4_frac_sharpe_gt0_ge70pct"], crit["MC4_median_sharpe_ge1_0"]))
        print("MC5 entry-jitter (n=500):")
        print("  design : median Sh=%.3f 5%%ile=%.3f P(Sh>=1)=%.3f  [crit median>=1.0: %s]"
              % (mc5_d["median_sharpe"], mc5_d["p5_sharpe"], mc5_d["p_sharpe_gt1"], crit["MC5_design_median_sharpe_ge1_0"]))
        print("  holdout: median Sh=%.3f 5%%ile=%.3f P(Sh>=1)=%.3f" % (mc5_h["median_sharpe"], mc5_h["p5_sharpe"], mc5_h["p_sharpe_gt1"]))
        print("Cost-doubled (fee 0.001, slip 0.0004): design Sharpe=%.3f netRet=%.2f%% | holdout Sharpe=%.3f netRet=%.2f%%"
              % (dm2["sharpe"] or 0, dm2["net_return_pct"] or 0, hm2["sharpe"] or 0, hm2["net_return_pct"] or 0))
        print("Holdout significance (descriptive): boot_p=%.4f rand_p=%.4f"
              % (h_boot["p_value"], h_rand["p_value"]))
        n_pass = sum(1 for v in crit.values() if v is True)
        print("MC criteria passed: %d / %d -> %s" % (n_pass, len(crit), crit))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_sanitize(payload), fh, indent=2)
    print("\nSaved %s" % RESULTS_PATH)


if __name__ == "__main__":
    main()
