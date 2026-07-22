"""Technical indicators, pure standard-library.

Every series returns a list the same length as the input, with ``None`` at
positions where the indicator is not yet defined (warm-up). The value at index
``t`` is the value known at the close of bar ``t`` -- no look-ahead: an
indicator at ``t`` never uses data from any bar > ``t``.
"""

import math


def sma(values, n):
    """Simple moving average over ``n`` values. sma[t] uses values[t-n+1..t]."""
    out = [None] * len(values)
    if n <= 0:
        return out
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def stdev(values, n, ddof=0):
    """Rolling standard deviation over ``n`` values.

    Default ``ddof=0`` (population) -- this is the convention used for the
    Bollinger Bands per the spec.
    """
    out = [None] * len(values)
    if n <= 0 or n - ddof <= 0:
        return out
    for i in range(len(values)):
        if i >= n - 1:
            window = values[i - n + 1 : i + 1]
            mean = sum(window) / n
            var = sum((x - mean) ** 2 for x in window) / (n - ddof)
            out[i] = math.sqrt(var)
    return out


def ema(values, n):
    """Exponential moving average with initial value = SMA of the first n bars.

    ema[n-1] = SMA(values[0..n-1]); ema[i] = a*values[i] + (1-a)*ema[i-1],
    with a = 2/(n+1). Positions before n-1 are ``None``.
    """
    out = [None] * len(values)
    if n <= 0 or len(values) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    seed = sum(values[:n]) / n
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def true_range(highs, lows, closes):
    """True range series. TR[0] = high[0]-low[0] (no prior close)."""
    n = len(highs)
    out = [None] * n
    for i in range(n):
        if i == 0:
            out[i] = highs[i] - lows[i]
        else:
            pc = closes[i - 1]
            out[i] = max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc))
    return out


def atr(highs, lows, closes, n):
    """Wilder-smoothed ATR. Initial value = simple mean of the first n TRs.

    atr[n-1] = mean(TR[0..n-1]); atr[i] = (atr[i-1]*(n-1) + TR[i]) / n.
    """
    tr = true_range(highs, lows, closes)
    out = [None] * len(tr)
    if n <= 0 or len(tr) < n:
        return out
    seed = sum(tr[:n]) / n
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(tr)):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def bollinger(closes, n, k):
    """Bollinger Bands. mid=SMA(n); upper/lower = mid +/- k*stdev(n, ddof=0)."""
    mid = sma(closes, n)
    sd = stdev(closes, n, ddof=0)
    upper = [None] * len(closes)
    lower = [None] * len(closes)
    for i in range(len(closes)):
        if mid[i] is not None and sd[i] is not None:
            upper[i] = mid[i] + k * sd[i]
            lower[i] = mid[i] - k * sd[i]
    return mid, upper, lower


def keltner(highs, lows, closes, n, mult):
    """Keltner Channel. mid=EMA(n); upper/lower = mid +/- mult*ATR(n).

    Note: per the spec Keltner uses EMA(n) and ATR(n) with n = the BB period
    (i.e. ATR20 for the v1 default), which is distinct from the ATR period
    (atr_n=14) used for stops.
    """
    mid = ema(closes, n)
    a = atr(highs, lows, closes, n)
    upper = [None] * len(closes)
    lower = [None] * len(closes)
    for i in range(len(closes)):
        if mid[i] is not None and a[i] is not None:
            upper[i] = mid[i] + mult * a[i]
            lower[i] = mid[i] - mult * a[i]
    return mid, upper, lower


def bandwidth(upper, lower, mid):
    """BandWidth = (upper - lower) / mid. ``None`` where undefined or mid==0."""
    out = [None] * len(upper)
    for i in range(len(upper)):
        if upper[i] is not None and lower[i] is not None and mid[i] not in (None, 0):
            out[i] = (upper[i] - lower[i]) / mid[i]
    return out


def rolling_max(values, n):
    """Highest value over the trailing ``n`` bars (inclusive). None during warm-up."""
    out = [None] * len(values)
    if n <= 0:
        return out
    for i in range(len(values)):
        if i >= n - 1:
            window = values[i - n + 1 : i + 1]
            if any(w is None for w in window):
                continue
            out[i] = max(window)
    return out


def rolling_min(values, n):
    """Lowest value over the trailing ``n`` bars (inclusive). None during warm-up."""
    out = [None] * len(values)
    if n <= 0:
        return out
    for i in range(len(values)):
        if i >= n - 1:
            window = values[i - n + 1 : i + 1]
            if any(w is None for w in window):
                continue
            out[i] = min(window)
    return out


def linreg_endpoint(values, n):
    """Value of the least-squares fitted line AT the last bar of each n-window.

    For each index t (t >= n-1) fit y = a + b*x on x=0..n-1 over
    values[t-n+1..t] and return the fitted value at x = n-1 (the window's
    endpoint). ``None`` where the window is not full or contains ``None``.
    """
    out = [None] * len(values)
    if n <= 0:
        return out
    mean_x = (n - 1) / 2.0
    sxx = sum((x - mean_x) ** 2 for x in range(n))
    for i in range(len(values)):
        if i < n - 1:
            continue
        window = values[i - n + 1 : i + 1]
        if any(w is None for w in window):
            continue
        mean_y = sum(window) / n
        sxy = sum((j - mean_x) * (window[j] - mean_y) for j in range(n))
        b = sxy / sxx if sxx != 0 else 0.0
        a = mean_y - b * mean_x
        out[i] = a + b * (n - 1)
    return out


def daily_ema_ffill(bars, n):
    """Daily (UTC) EMA(n) of closes, forward-filled onto the intraday bars.

    The value used at bar ``t`` comes from the EMA computed through the last UTC
    day that fully closed *before* bar ``t``'s own day (i.e. day D-1 for a bar on
    day D) -- so there is no look-ahead: a bar never sees an EMA value that
    incorporates any close from its own (still-open) day. ``None`` until the
    prior completed day's EMA is available.
    """
    dates = []
    daily_close = []
    cur = None
    for b in bars:
        d = b.ts.date()
        if cur is None or d != cur:
            dates.append(d)
            daily_close.append(b.close)
            cur = d
        else:
            daily_close[-1] = b.close
    dema = ema(daily_close, n)
    pos = {d: i for i, d in enumerate(dates)}
    out = [None] * len(bars)
    for k, b in enumerate(bars):
        p = pos[b.ts.date()]
        if p - 1 >= 0:
            out[k] = dema[p - 1]
    return out


def bw_percentile(bw, lookback):
    """Percentile rank of BandWidth within the trailing ``lookback`` bars.

    The window includes the current bar. Rank is defined as
    100 * (# window values <= current) / window_size, so the minimum in the
    window maps to a small positive value and the maximum to 100. Only defined
    once a full window of non-None BandWidth values is available.
    """
    out = [None] * len(bw)
    L = lookback
    if L <= 0:
        return out
    for i in range(len(bw)):
        if i < L - 1:
            continue
        window = bw[i - L + 1 : i + 1]
        if any(w is None for w in window):
            continue
        cur = bw[i]
        cnt = sum(1 for w in window if w <= cur)
        out[i] = 100.0 * cnt / L
    return out
