"""Event-loop backtest simulator (single position, risk-based sizing).

Baseline execution model (docs/implementation_spec_v1.md):
  * A signal at bar t fills at bar t+1 open with slippage; fees on both legs.
  * Size risks 3% of current equity to the initial stop, capped at 5x notional.
  * Chandelier ATR trailing stop tightens (never loosens) each bar.
  * Gap-aware stop fill: long min(open, stop)*(1-slip); short max(open, stop)*(1+slip).

v4 additions (docs/strategy_v4_spec.md), all gated behind params so v1-v3 runs
are byte-for-byte unchanged when the flags are off:
  * scale_out       : take 50% of the original qty off at +1.5R, then move the
                      remainder's stop to breakeven (kept together with the
                      chandelier trail via the more-protective side).
  * time_stop_bars  : if not scaled out and unrealized R < 0 at this many bars
                      after entry, exit the remainder at the next bar's open.
  * reentry_enabled : pull-back re-entries within an episode (short-only); needs
                      episode_release / pullback_short arrays from the strategy.

Within-bar ordering while holding (per spec): pending time-stop first, then the
incoming stop level, then the scale-out trigger, then the trail update. If a
stop and a scale-out could both trigger on the same bar, the stop wins.
"""

INITIAL_CAPITAL = 100000.0


def run_backtest(
    bars,
    signals,
    atr_stop,
    params,
    eval_start,
    episode_release=None,
    pullback_short=None,
):
    """Run the simulation over the whole series.

    Entries whose fill bar timestamp is before ``eval_start`` are suppressed.

    Returns dict with:
      trades         list of trade records (dicts)
      equity         list of (ts, equity) for every bar
      in_pos_flags   list[bool], True on bars where a position was held
    """
    fee = params["fee"]
    slip = params["slip"]
    stop_atr = params["stop_atr"]
    trail_atr = params["trail_atr"]
    scale_out = bool(params.get("scale_out", False))
    time_stop_bars = int(params.get("time_stop_bars", 0))
    reentry_enabled = bool(params.get("reentry_enabled", False))
    reentry_max = int(params.get("reentry_max", 2))
    reentry_window = int(params.get("reentry_window", 100))
    allow_short = params.get("allow_short", True)

    n = len(bars)
    equity = []
    trades = []
    in_pos_flags = [False] * n

    cash = INITIAL_CAPITAL
    eq = INITIAL_CAPITAL
    P = None  # open position (dict) or None

    # Episode state (for v4b pull-back re-entries).
    active_episode_r = None
    episode_initial_done = False
    episode_reentries = 0

    def open_position(c, direction, is_reentry):
        nonlocal cash, P
        t = c - 1  # signal bar
        atr_t = atr_stop[t]
        if atr_t is None or atr_t <= 0:
            return False
        sd = stop_atr * atr_t
        bar = bars[c]
        if direction == "long":
            ef = bar.open * (1.0 + slip)
            istop = ef - sd
        else:
            ef = bar.open * (1.0 - slip)
            istop = ef + sd
        if sd <= 0 or ef <= 0:
            return False
        q = (eq * 0.03) / sd
        cap = eq * 5.0 / ef  # 5x notional leverage cap
        if q > cap:
            q = cap
        if q <= 0:
            return False
        fee_in = fee * q * ef
        if direction == "long":
            cash -= q * ef + fee_in
        else:
            cash += q * ef - fee_in
        P = {
            "side": direction,
            "qty0": q,
            "qty": q,
            "entry_fill": ef,
            "entry_bar": c,
            "entry_ts": bar.ts,
            "stop": istop,
            "stop_dist": sd,
            "risk0": q * sd,
            "extreme": None,
            "realized_pnl": 0.0,
            "scaled_out": False,
            "is_reentry": is_reentry,
            "time_stop_pending": False,
        }
        return True

    def do_scaleout(sfill):
        nonlocal cash
        side = P["side"]
        ef = P["entry_fill"]
        q_leg = 0.5 * P["qty0"]
        fee_leg = fee * q_leg * sfill
        if side == "long":
            cash += q_leg * sfill - fee_leg
            leg = q_leg * (sfill - ef) - fee * q_leg * ef - fee_leg
        else:
            cash -= q_leg * sfill + fee_leg
            leg = q_leg * (ef - sfill) - fee * q_leg * ef - fee_leg
        P["realized_pnl"] += leg
        P["qty"] -= q_leg
        P["scaled_out"] = True
        # Move the remainder's stop to breakeven (more-protective side only).
        if side == "long":
            if P["stop"] < ef:
                P["stop"] = ef
        else:
            if P["stop"] > ef:
                P["stop"] = ef

    def close_remainder(c, exit_fill, reason):
        nonlocal cash, P
        side = P["side"]
        q = P["qty"]
        ef = P["entry_fill"]
        fee_out = fee * q * exit_fill
        if side == "long":
            cash += q * exit_fill - fee_out
            leg = q * (exit_fill - ef) - fee * q * ef - fee_out
        else:
            cash -= q * exit_fill + fee_out
            leg = q * (ef - exit_fill) - fee * q * ef - fee_out
        P["realized_pnl"] += leg
        risk0 = P["risk0"]
        rmult = P["realized_pnl"] / risk0 if risk0 > 0 else 0.0
        trades.append(
            {
                "side": side,
                "entry_ts": P["entry_ts"].isoformat(),
                "entry_px": ef,
                "exit_ts": bars[c].ts.isoformat(),
                "exit_px": exit_fill,
                "qty": P["qty0"],
                "bars_held": c - P["entry_bar"],
                "pnl": P["realized_pnl"],
                "r_multiple": rmult,
                "scaled_out": P["scaled_out"],
                "exit_reason": reason,
                "is_reentry": P["is_reentry"],
                "forced_close": reason == "end_of_data",
            }
        )
        P = None

    for c in range(n):
        bar = bars[c]

        # Episode reset detection (a new squeeze release replaces the episode).
        if episode_release is not None and c > 0:
            er = episode_release[c - 1]
            if er != active_episode_r:
                active_episode_r = er
                episode_initial_done = False
                episode_reentries = 0

        # 1. Entry (if flat): initial breakout, else pull-back re-entry (v4b).
        if P is None and c > 0 and bar.ts >= eval_start:
            t = c - 1
            entry_dir = None
            is_re = False
            if signals[t] is not None:
                entry_dir = signals[t]
            elif (
                reentry_enabled
                and pullback_short is not None
                and allow_short
                and active_episode_r is not None
                and episode_initial_done
                and episode_reentries < reentry_max
                and (t - active_episode_r) <= reentry_window
                and pullback_short[t]
            ):
                entry_dir = "short"
                is_re = True
            if entry_dir is not None:
                if open_position(c, entry_dir, is_re):
                    if is_re:
                        episode_reentries += 1
                    else:
                        episode_initial_done = True

        # 2. Holding logic for bar c.
        if P is not None:
            side = P["side"]
            if P["time_stop_pending"]:
                xf = bar.open * (1.0 - slip) if side == "long" else bar.open * (1.0 + slip)
                close_remainder(c, xf, "time_stop")
            else:
                # 2a. incoming stop
                stopped = False
                if side == "long":
                    if bar.low <= P["stop"]:
                        xf = min(bar.open, P["stop"]) * (1.0 - slip)
                        stopped = True
                else:
                    if bar.high >= P["stop"]:
                        xf = max(bar.open, P["stop"]) * (1.0 + slip)
                        stopped = True
                if stopped:
                    close_remainder(c, xf, "stop")
                else:
                    # 2b. scale-out at +1.5R (once)
                    if scale_out and not P["scaled_out"]:
                        ef = P["entry_fill"]
                        sd = P["stop_dist"]
                        if side == "long":
                            level = ef + 1.5 * sd
                            if bar.high >= level:
                                base = max(bar.open, level)  # gap in our favour -> open
                                do_scaleout(base * (1.0 - slip))
                        else:
                            level = ef - 1.5 * sd
                            if bar.low <= level:
                                base = min(bar.open, level)  # gap in our favour -> open
                                do_scaleout(base * (1.0 + slip))
                    # 2c. trail update (for subsequent bars)
                    at = atr_stop[c]
                    if side == "long":
                        P["extreme"] = bar.high if P["extreme"] is None else max(P["extreme"], bar.high)
                        if at is not None:
                            ns = P["extreme"] - trail_atr * at
                            if ns > P["stop"]:
                                P["stop"] = ns
                    else:
                        P["extreme"] = bar.low if P["extreme"] is None else min(P["extreme"], bar.low)
                        if at is not None:
                            ns = P["extreme"] + trail_atr * at
                            if ns < P["stop"]:
                                P["stop"] = ns
                    # 2d. time stop evaluation at exactly time_stop_bars after entry
                    if (
                        time_stop_bars > 0
                        and not P["scaled_out"]
                        and (c - P["entry_bar"]) == time_stop_bars
                    ):
                        if side == "long":
                            ur = (bar.close - P["entry_fill"]) / P["stop_dist"]
                        else:
                            ur = (P["entry_fill"] - bar.close) / P["stop_dist"]
                        if ur < 0:
                            P["time_stop_pending"] = True

        # 3. Mark-to-market equity at this bar's close.
        if P is not None:
            side = P["side"]
            if side == "long":
                eq = cash + P["qty"] * bar.close
            else:
                eq = cash - P["qty"] * bar.close
            in_pos_flags[c] = True
        else:
            eq = cash
        equity.append((bar.ts, eq))

    # Force-close any position still open at the last bar.
    if P is not None:
        c = n - 1
        bar = bars[c]
        side = P["side"]
        xf = bar.close * (1.0 - slip) if side == "long" else bar.close * (1.0 + slip)
        close_remainder(c, xf, "end_of_data")
        equity[-1] = (bar.ts, cash)

    return {"trades": trades, "equity": equity, "in_pos_flags": in_pos_flags}
