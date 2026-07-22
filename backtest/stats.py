"""Statistical significance tests (deterministic via random.Random(seed))."""

import random

from . import metrics


def bootstrap_pvalue(r_multiples, n=10000, seed=42):
    """Bootstrap p-value for H0: mean trade R-multiple <= 0.

    Resamples the R-multiples with replacement ``n`` times, builds the
    distribution of the mean, and returns
    p = (#{bootstrap_mean <= 0} + 1) / (n + 1).
    """
    m = len(r_multiples)
    if m == 0:
        return {
            "p_value": 1.0,
            "actual_mean_r": None,
            "boot_mean": None,
            "n": n,
            "n_trades": 0,
        }
    rng = random.Random(seed)
    actual_mean = sum(r_multiples) / m
    le_zero = 0
    boot_sum = 0.0
    for _ in range(n):
        sample = rng.choices(r_multiples, k=m)
        mean = sum(sample) / m
        boot_sum += mean
        if mean <= 0:
            le_zero += 1
    return {
        "p_value": (le_zero + 1) / (n + 1),
        "actual_mean_r": actual_mean,
        "boot_mean": boot_sum / n,
        "n": n,
        "n_trades": m,
    }


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in [0,100]) of a sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def random_entry_test(
    bars,
    eval_start,
    trades,
    params,
    atr_stop,
    actual_sharpe,
    n=1000,
    seed=42,
    eval_end=None,
):
    """Random-entry permutation test matched on the strategy's trade structure.

    Each of ``n`` trials places ``len(trades)`` non-overlapping trades at random
    bars in the evaluation window, with holding durations sampled (with
    replacement) from the actual trade durations and direction sampled at the
    actual long/short ratio. Sizing is the same 3%-risk rule with an initial
    stop of stop_atr*ATR at the signal bar; there is no trailing stop -- each
    trade exits at the earlier of a stop hit (checked each held bar) or the
    holding duration elapsing at the next bar's open. Costs are identical.

    If ``eval_end`` is given, both the random entry placement and the daily
    Sharpe are restricted to the window [eval_start, eval_end].

    Returns the p-value P(random_sharpe >= actual_sharpe) plus the random
    distribution's mean and 95th percentile.
    """
    n_tr = len(trades)
    if n_tr == 0:
        return {
            "p_value": 1.0,
            "actual_sharpe": actual_sharpe,
            "random_mean_sharpe": None,
            "random_p95_sharpe": None,
            "n": n,
            "n_trades": 0,
        }

    fee = params["fee"]
    slip = params["slip"]
    stop_atr = params["stop_atr"]

    opens = [b.open for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    nb = len(bars)
    last_idx = nb - 1

    # Evaluation window bar indices.
    eval_start_idx = None
    for i, b in enumerate(bars):
        if b.ts >= eval_start:
            eval_start_idx = i
            break
    if eval_start_idx is None or eval_start_idx >= last_idx:
        return {
            "p_value": 1.0,
            "actual_sharpe": actual_sharpe,
            "random_mean_sharpe": None,
            "random_p95_sharpe": None,
            "n": n,
            "n_trades": n_tr,
        }

    # Last bar index in the window (bounded by eval_end if provided).
    if eval_end is None:
        eval_end_idx = last_idx
    else:
        eval_end_idx = eval_start_idx
        for i in range(eval_start_idx, nb):
            if bars[i].ts <= eval_end:
                eval_end_idx = i
            else:
                break

    if eval_end_idx <= eval_start_idx:
        return {
            "p_value": 1.0,
            "actual_sharpe": actual_sharpe,
            "random_mean_sharpe": None,
            "random_p95_sharpe": None,
            "n": n,
            "n_trades": n_tr,
        }

    K = eval_end_idx - eval_start_idx + 1  # number of window bars

    # Positions (within the window array) that are UTC-day closes.
    day_close_ks = []
    for k in range(K):
        bi = eval_start_idx + k
        if k == K - 1 or bars[bi + 1].ts.date() != bars[bi].ts.date():
            day_close_ks.append(k)

    durations = [t["bars_held"] for t in trades]
    n_long = sum(1 for t in trades if t["side"] == "long")
    long_ratio = n_long / n_tr

    # Valid entry fill bars: within the window, with a valid prior-bar ATR and
    # room for at least a one-bar hold.
    first_entry = eval_start_idx
    last_entry = eval_end_idx - 1

    rng = random.Random(seed)
    sharpes = []

    for _ in range(n):
        # --- place non-overlapping trades ---
        placed = []  # (entry_bar, end_bar, direction)
        attempts = 0
        budget = n_tr * 60 + 60
        while len(placed) < n_tr and attempts < budget:
            attempts += 1
            f = rng.randint(first_entry, last_entry)
            dur = rng.choice(durations)
            end = f + dur
            if end > eval_end_idx:
                end = eval_end_idx
            overlap = False
            for (pf, pe, _d) in placed:
                if not (end < pf or f > pe):
                    overlap = True
                    break
            if overlap:
                continue
            direction = "long" if rng.random() < long_ratio else "short"
            placed.append((f, end, direction))
        placed.sort(key=lambda x: x[0])

        # --- simulate, building the hourly eval equity series ---
        eq_series = [0.0] * K
        eq_real = 100000.0
        ptr = 0  # next eval-array index still to be filled

        for (f, end, direction) in placed:
            kf = f - eval_start_idx
            # flat bars before this entry
            for k in range(ptr, kf):
                eq_series[k] = eq_real
            if ptr < kf:
                ptr = kf

            atr_t = atr_stop[f - 1]
            kend = end - eval_start_idx
            if atr_t is None or atr_t <= 0:
                for k in range(ptr, kend + 1):
                    eq_series[k] = eq_real
                ptr = kend + 1
                continue

            sd = stop_atr * atr_t
            if direction == "long":
                ef = opens[f] * (1.0 + slip)
                istop = ef - sd
            else:
                ef = opens[f] * (1.0 - slip)
                istop = ef + sd

            if sd <= 0 or ef <= 0:
                for k in range(ptr, kend + 1):
                    eq_series[k] = eq_real
                ptr = kend + 1
                continue

            q = (eq_real * 0.03) / sd
            cap = eq_real * 5.0 / ef
            if q > cap:
                q = cap
            if q <= 0:
                for k in range(ptr, kend + 1):
                    eq_series[k] = eq_real
                ptr = kend + 1
                continue

            fee_in = fee * q * ef

            # scan for a stop hit across the held bars [f, end-1]
            exit_bar = None
            exit_fill = None
            for c in range(f, end):
                if direction == "long":
                    if lows[c] <= istop:
                        exit_bar = c
                        exit_fill = min(opens[c], istop) * (1.0 - slip)
                        break
                else:
                    if highs[c] >= istop:
                        exit_bar = c
                        exit_fill = max(opens[c], istop) * (1.0 + slip)
                        break
            if exit_bar is None:
                exit_bar = end
                if direction == "long":
                    exit_fill = opens[end] * (1.0 - slip)
                else:
                    exit_fill = opens[end] * (1.0 + slip)

            fee_out = fee * q * exit_fill
            if direction == "long":
                pnl = q * (exit_fill - ef) - fee_in - fee_out
            else:
                pnl = q * (ef - exit_fill) - fee_in - fee_out

            # mark held bars [f, exit_bar-1] at unrealized value
            for c in range(f, exit_bar):
                k = c - eval_start_idx
                if direction == "long":
                    eq_series[k] = eq_real + q * (closes[c] - ef) - fee_in
                else:
                    eq_series[k] = eq_real + q * (ef - closes[c]) - fee_in
            eq_real += pnl
            kx = exit_bar - eval_start_idx
            eq_series[kx] = eq_real
            ptr = kx + 1

        for k in range(ptr, K):
            eq_series[k] = eq_real

        daily_eq = [eq_series[k] for k in day_close_ks]
        sharpes.append(metrics.sharpe_from_daily(daily_eq))

    ge = sum(1 for s in sharpes if s >= actual_sharpe)
    p = (ge + 1) / (n + 1)
    s_sorted = sorted(sharpes)
    return {
        "p_value": p,
        "actual_sharpe": actual_sharpe,
        "random_mean_sharpe": sum(sharpes) / len(sharpes),
        "random_p95_sharpe": _percentile(s_sorted, 95.0),
        "n": n,
        "n_trades": n_tr,
    }
