"""Event-loop backtest simulator (single position, risk-based sizing).

Execution model (see docs/implementation_spec_v1.md):
  * A signal at bar t fills at bar t+1 open with slippage; fees are charged on
    both legs.
  * Position size risks 3% of current equity to the initial stop, capped at 5x
    notional leverage.
  * A Chandelier ATR trailing stop tightens (never loosens) each bar.
  * Stop execution is gap-aware: for longs the fill is min(open, stop) with
    slippage (symmetric for shorts). The stop is also checked on the entry bar.

No look-ahead: the stop level active *during* bar c is the one established at
the close of bar c-1 (checked first), and only then is the trailing stop
updated using bar c's data for subsequent bars.
"""

INITIAL_CAPITAL = 100000.0


def run_backtest(bars, signals, atr_stop, params, eval_start):
    """Run the simulation over the whole series.

    Entries whose fill bar timestamp is before ``eval_start`` are suppressed.

    Returns a dict with:
      trades         list of trade records (dicts)
      equity         list of (ts, equity) for every bar
      in_pos_flags   list[bool], True on bars where a position was held
    """
    fee = params["fee"]
    slip = params["slip"]
    stop_atr = params["stop_atr"]
    trail_atr = params["trail_atr"]

    n = len(bars)
    equity = []
    trades = []
    in_pos_flags = [False] * n

    cash = INITIAL_CAPITAL
    eq = INITIAL_CAPITAL

    in_position = False
    side = None
    qty = 0.0
    entry_fill = 0.0
    entry_bar = 0
    entry_ts = None
    stop = 0.0
    initial_stop = 0.0
    stop_dist = 0.0
    fee_in = 0.0
    extreme = None

    def _record_exit(exit_bar, exit_fill, forced):
        nonlocal cash
        fee_out = fee * qty * exit_fill
        if side == "long":
            cash += qty * exit_fill - fee_out
            pnl = qty * (exit_fill - entry_fill) - fee_in - fee_out
        else:
            cash -= qty * exit_fill + fee_out
            pnl = qty * (entry_fill - exit_fill) - fee_in - fee_out
        risk0 = qty * stop_dist
        rmult = pnl / risk0 if risk0 > 0 else 0.0
        trades.append(
            {
                "side": side,
                "entry_ts": entry_ts.isoformat(),
                "entry_px": entry_fill,
                "exit_ts": bars[exit_bar].ts.isoformat(),
                "exit_px": exit_fill,
                "qty": qty,
                "bars_held": exit_bar - entry_bar,
                "pnl": pnl,
                "r_multiple": rmult,
                "forced_close": forced,
            }
        )

    for c in range(n):
        bar = bars[c]

        # 1. Entry: fill at this bar's open if the prior bar signalled and we
        #    are flat and the fill lands in the evaluation window.
        if not in_position and c > 0 and signals[c - 1] is not None and bar.ts >= eval_start:
            t = c - 1  # signal bar
            atr_t = atr_stop[t]
            if atr_t is not None and atr_t > 0:
                sd = stop_atr * atr_t
                direction = signals[t]
                if direction == "long":
                    ef = bar.open * (1.0 + slip)
                    istop = ef - sd
                else:
                    ef = bar.open * (1.0 - slip)
                    istop = ef + sd
                risk_amount = eq * 0.03
                if sd > 0 and ef > 0:
                    q = risk_amount / sd
                    cap = eq * 5.0 / ef  # 5x notional leverage cap
                    if q > cap:
                        q = cap
                    if q > 0:
                        fee_in = fee * q * ef
                        if direction == "long":
                            cash -= q * ef + fee_in
                        else:
                            cash += q * ef - fee_in
                        in_position = True
                        side = direction
                        qty = q
                        entry_fill = ef
                        entry_bar = c
                        entry_ts = bar.ts
                        initial_stop = istop
                        stop = istop
                        stop_dist = sd
                        extreme = None

        # 2. Holding logic for bar c: check the incoming stop first, then trail.
        if in_position:
            exited = False
            if side == "long":
                if bar.low <= stop:
                    exit_fill = min(bar.open, stop) * (1.0 - slip)
                    exited = True
            else:
                if bar.high >= stop:
                    exit_fill = max(bar.open, stop) * (1.0 + slip)
                    exited = True

            if exited:
                _record_exit(c, exit_fill, forced=False)
                in_position = False
            else:
                at = atr_stop[c]
                if side == "long":
                    extreme = bar.high if extreme is None else max(extreme, bar.high)
                    if at is not None:
                        new_stop = extreme - trail_atr * at
                        if new_stop > stop:
                            stop = new_stop
                else:
                    extreme = bar.low if extreme is None else min(extreme, bar.low)
                    if at is not None:
                        new_stop = extreme + trail_atr * at
                        if new_stop < stop:
                            stop = new_stop

        # 3. Mark-to-market equity at this bar's close.
        if in_position:
            if side == "long":
                eq = cash + qty * bar.close
            else:
                eq = cash - qty * bar.close
            in_pos_flags[c] = True
        else:
            eq = cash
        equity.append((bar.ts, eq))

    # Force-close any position still open at the last bar (mark realized).
    if in_position:
        c = n - 1
        bar = bars[c]
        exit_fill = bar.close * (1.0 - slip) if side == "long" else bar.close * (1.0 + slip)
        _record_exit(c, exit_fill, forced=True)
        in_position = False
        equity[-1] = (bar.ts, cash)

    return {"trades": trades, "equity": equity, "in_pos_flags": in_pos_flags}
