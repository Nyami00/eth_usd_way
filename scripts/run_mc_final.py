#!/usr/bin/env python3
"""Final Monte-Carlo suite for the report (docs/optimization_mc_spec section 2).

Two subjects:
  * v4a  -- the 2026-tuned strategy (results/v4_gate.json): short-only, ttm
    kc1.5, ms4, rw3, confirm_bars 0, HTF daily-EMA50 level filter only,
    stop 2.0, trail 3.0, scale-out on, time_stop 12. MC on its 2026 YTD window.
  * C1/C2/C3 -- the v6 out-of-sample candidates. MC on their DESIGN window
    (2020-03-01..2025-09-30); MC4 from stage-2 grid neighbors (no recompute).

Registered pass criteria (all seeds 42):
  MC1 5%ile Sharpe > 0 | MC2 5%ile total return > 0 | MC3 95%ile maxDD < 35% |
  MC4 >=70% neighbors Sharpe>0 AND median Sharpe >= 1.0 | MC5 median Sharpe >= 1.0

Outputs results/mc_final.json.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_v6_holdout as H  # reuse MC machinery and window helpers
from backtest import data as data_mod
from backtest import engine as engine_mod
from backtest import metrics as metrics_mod
from backtest import strategy as strategy_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_1H = os.path.join(ROOT, "data", "eth_usd_1h.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "mc_final.json")

DESIGN_START = H.DESIGN_START
DESIGN_END = H.DESIGN_END
GATE_ENTRY_START = H.GATE_ENTRY_START
YTD_START = H.YTD_START

V4A = {
    "bb_n": 20, "bb_k": 2.0, "squeeze_mode": "ttm", "confirm_bars": 0,
    "trend_ema": 0, "htf_trend": True, "htf_ema_n": 50,
    "momentum_filter": False, "volume_filter": False, "vol_mult": 1.3,
    "mom_n": 20, "vol_sma_n": 20, "bw_lookback": 120, "bw_q": 20.0,
    "kc_mult": 1.5, "min_squeeze_bars": 4, "release_window": 3,
    "atr_n": 14, "stop_atr": 2.0, "trail_atr": 3.0, "flip_on_opposite": False,
    "fee": 0.0005, "slip": 0.0002, "scale_out": True, "time_stop_bars": 12,
    "reentry_enabled": False, "ema_slope_filter": False,
    "allow_long": False, "allow_short": True,
}


def weekly(daily, step=7):
    return daily[::step]


def crit_line(name, value, ok):
    return "%-40s %-14s -> %s" % (name, ("%.4f" % value) if value is not None else "n/a",
                                  "PASS" if ok else ("FAIL" if ok is False else "n/a"))


def run_subject(name, params, bars, last_ts, eval_start, entry_end, win_start, win_end, mc4_cfg):
    signals, ind = strategy_mod.generate_signals(bars, params)
    atr = ind["atr_stop"]
    bt = engine_mod.run_backtest(bars, signals, atr, params, eval_start, entry_end=entry_end)
    m, trades, _ = H.window_metrics(bt["equity"], bt["trades"], bt["in_pos_flags"], bars, win_start, win_end)
    daily = H.window_daily_closes(bt["equity"], win_start, win_end)
    daily_rets = metrics_mod.daily_returns(daily)
    rmults = [t["r_multiple"] for t in trades]

    mc1 = H.mc1_block_bootstrap(daily_rets, n=10000, block=5, seed=42)
    mc2 = H.mc2_trade_bootstrap(rmults, n=10000, seed=42)
    mc3 = H.mc3_order_shuffle(rmults, n=10000, seed=42)
    mc5 = H.mc5_entry_jitter(bars, signals, atr, params, eval_start, entry_end, win_start, win_end, n=500, seed=42)
    mc4 = H.mc4_param_perturbation(mc4_cfg) if mc4_cfg else None

    # cost-doubled
    p2 = dict(params)
    p2["fee"] = 0.001
    p2["slip"] = 0.0004
    sig2, ind2 = strategy_mod.generate_signals(bars, p2)
    bt2 = engine_mod.run_backtest(bars, sig2, ind2["atr_stop"], p2, eval_start, entry_end=entry_end)
    m2, _, _ = H.window_metrics(bt2["equity"], bt2["trades"], bt2["in_pos_flags"], bars, win_start, win_end)

    crit = {
        "MC1_p5_sharpe_gt0": (mc1["p5_sharpe"] > 0) if mc1 else None,
        "MC2_p5_totalreturn_gt0": (mc2["p5_total_return_pct"] > 0) if mc2 else None,
        "MC3_p95_maxdd_lt35": (mc3["p95_maxdd_pct"] < 35.0) if mc3 else None,
        "MC4_frac_gt0_ge70pct": (mc4["frac_sharpe_gt0"] >= 0.70) if mc4 else None,
        "MC4_median_ge1_0": (mc4["median_sharpe"] >= 1.0) if mc4 else None,
        "MC5_median_sharpe_ge1_0": (mc5["median_sharpe"] >= 1.0) if mc5 else None,
    }
    return {
        "name": name, "params": params, "window": [win_start.isoformat(), win_end.isoformat()],
        "metrics": m, "MC1": mc1, "MC2": mc2, "MC3": mc3, "MC4": mc4, "MC5": mc5,
        "cost_doubled_metrics": m2, "criteria": crit, "_daily": daily,
    }


def report(r):
    m = r["metrics"]
    print("\n=== %s (window %s..%s) ===" % (r["name"], r["window"][0][:10], r["window"][1][:10]))
    print("metrics: trades=%d netRet=%.2f%% Sharpe=%.3f maxDD=%.2f%% win=%.1f avgR=%.3f PF=%s"
          % (m["trades"], m["net_return_pct"] or 0, m["sharpe"] or 0, m["max_dd_pct"] or 0,
             m["win_rate"] or 0, m["avg_r"] or 0, ("%.3f" % m["profit_factor"]) if m["profit_factor"] else "n/a"))
    c = r["criteria"]
    if r["MC1"]:
        print("MC1 block-bootstrap: 5%%ile=%.3f median=%.3f P(Sh>0)=%.3f P(Sh>1.5)=%.3f"
              % (r["MC1"]["p5_sharpe"], r["MC1"]["median_sharpe"], r["MC1"]["p_sharpe_gt0"], r["MC1"]["p_sharpe_gt1_5"]))
    print("  " + crit_line("MC1 5%ile Sharpe > 0", r["MC1"]["p5_sharpe"] if r["MC1"] else None, c["MC1_p5_sharpe_gt0"]))
    if r["MC2"]:
        print("MC2 trade-bootstrap: 5%%ile totRet=%.2f%% median=%.2f%% P(ret>0)=%.3f"
              % (r["MC2"]["p5_total_return_pct"], r["MC2"]["median_total_return_pct"], r["MC2"]["p_return_gt0"]))
    print("  " + crit_line("MC2 5%ile totalReturn > 0", r["MC2"]["p5_total_return_pct"] if r["MC2"] else None, c["MC2_p5_totalreturn_gt0"]))
    if r["MC3"]:
        print("MC3 order-shuffle: 95%%ile maxDD=%.2f%% median=%.2f%%" % (r["MC3"]["p95_maxdd_pct"], r["MC3"]["median_maxdd_pct"]))
    print("  " + crit_line("MC3 95%ile maxDD < 35%", r["MC3"]["p95_maxdd_pct"] if r["MC3"] else None, c["MC3_p95_maxdd_lt35"]))
    if r["MC4"]:
        print("MC4 param-perturbation: %d neighbors frac(Sh>0)=%.3f median Sh=%.3f"
              % (r["MC4"]["n_neighbors"], r["MC4"]["frac_sharpe_gt0"], r["MC4"]["median_sharpe"]))
        print("  " + crit_line("MC4 frac Sharpe>0 >= 70%", r["MC4"]["frac_sharpe_gt0"], c["MC4_frac_gt0_ge70pct"]))
        print("  " + crit_line("MC4 median Sharpe >= 1.0", r["MC4"]["median_sharpe"], c["MC4_median_ge1_0"]))
    else:
        print("MC4 param-perturbation: n/a (not in stage-2 grid)")
    if r["MC5"]:
        print("MC5 entry-jitter (n=500): median Sh=%.3f 5%%ile=%.3f P(Sh>=1)=%.3f"
              % (r["MC5"]["median_sharpe"], r["MC5"]["p5_sharpe"], r["MC5"]["p_sharpe_gt1"]))
    print("  " + crit_line("MC5 median Sharpe >= 1.0", r["MC5"]["median_sharpe"] if r["MC5"] else None, c["MC5_median_sharpe_ge1_0"]))
    m2 = r["cost_doubled_metrics"]
    print("cost-doubled (fee 0.001/slip 0.0004): Sharpe=%.3f netRet=%.2f%% trades=%d maxDD=%.2f%%"
          % (m2["sharpe"] or 0, m2["net_return_pct"] or 0, m2["trades"], m2["max_dd_pct"] or 0))
    npass = sum(1 for v in c.values() if v is True)
    ntot = sum(1 for v in c.values() if v is not None)
    print("criteria passed: %d / %d" % (npass, ntot))


def main():
    t0 = time.time()
    bars = data_mod.load_bars(DATA_1H)
    last_ts = bars[-1].ts
    print("Loaded %d bars (%s .. %s)" % (len(bars), bars[0].ts.isoformat(), last_ts.isoformat()))

    # v4a on its 2026 YTD window (MC4 n/a).
    v4a = run_subject("v4a_YTD", V4A, bars, last_ts, GATE_ENTRY_START, None, YTD_START, last_ts, mc4_cfg=None)

    # C1/C2/C3 on their design window; MC4 from stage-2 neighbors.
    cand_results = []
    for cand in H.CANDIDATES:
        params = H.make_params(cand)
        r = run_subject(cand["name"] + "_DESIGN", params, bars, last_ts,
                        DESIGN_START, DESIGN_END, DESIGN_START, DESIGN_END, mc4_cfg=cand)
        cand_results.append((cand, r))

    report(v4a)
    for cand, r in cand_results:
        report(r)

    # ---- equity curves for charting (modest) ----
    equity = {
        "v4a_ytd_daily": v4a["_daily"],
        "c1_design_weekly": weekly(cand_results[0][1]["_daily"]),
        "c2_design_weekly": weekly(cand_results[1][1]["_daily"]),
        "c3_design_weekly": weekly(cand_results[2][1]["_daily"]),
    }

    def strip(r):
        r = dict(r)
        r.pop("_daily", None)
        return r

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pass_criteria": {
            "MC1": "5%ile Sharpe > 0", "MC2": "5%ile total return > 0",
            "MC3": "95%ile maxDD < 35%",
            "MC4": ">=70% neighbors Sharpe>0 AND median Sharpe >= 1.0",
            "MC5": "median Sharpe >= 1.0",
        },
        "v4a_ytd": strip(v4a),
        "candidates_design": [strip(r) for _, r in cand_results],
        "equity_daily_closes": equity,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(H._sanitize(payload), fh, indent=2)
    print("\nSaved %s" % RESULTS_PATH)
    print("Total runtime: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
