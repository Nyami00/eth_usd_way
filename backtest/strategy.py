"""Squeeze-breakout signal generation.

``generate_signals`` returns a list the same length as ``bars`` whose entries
are ``None``, ``"long"`` or ``"short"``. A signal at index ``t`` means the
setup fired at the close of bar ``t``; the engine fills it at bar ``t+1``.

All indicator values used at bar ``t`` are known at the close of bar ``t``
(no look-ahead). Signals are emitted for the entire series (including the 2025
warm-up) so that squeeze state and setup consumption are tracked continuously;
the engine is responsible for suppressing entries whose fill would occur before
the evaluation start.
"""

from . import indicators as ind


def compute_indicators(bars, params):
    """Compute every indicator series needed for signals and stops.

    Returns a dict of parallel lists (same length as ``bars``).
    """
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    bb_n = params["bb_n"]
    bb_k = params["bb_k"]
    bb_mid, bb_up, bb_lo = ind.bollinger(closes, bb_n, bb_k)
    bw = ind.bandwidth(bb_up, bb_lo, bb_mid)

    out = {
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "bb_lo": bb_lo,
        "bw": bw,
    }

    mode = params.get("squeeze_mode", "pctile")
    if mode == "pctile":
        out["bw_pct"] = ind.bw_percentile(bw, params["bw_lookback"])
    elif mode == "ttm":
        # Keltner uses EMA(bb_n) and ATR(bb_n) -- i.e. ATR20 for the default.
        _, kc_up, kc_lo = ind.keltner(highs, lows, closes, bb_n, params["kc_mult"])
        out["kc_up"] = kc_up
        out["kc_lo"] = kc_lo
    else:
        raise ValueError("unknown squeeze_mode: %r" % (mode,))

    trend_ema = params.get("trend_ema", 0)
    out["ema_trend"] = ind.ema(closes, trend_ema) if trend_ema and trend_ema > 0 else None

    # ATR used for stops / sizing (atr_n, typically 14).
    out["atr_stop"] = ind.atr(highs, lows, closes, params["atr_n"])
    return out


def _squeeze_state(idx, ind_data, params):
    """Whether bar ``idx`` is in a squeeze (False if any input is undefined)."""
    mode = params.get("squeeze_mode", "pctile")
    if mode == "pctile":
        pct = ind_data["bw_pct"][idx]
        if pct is None:
            return False
        return pct < params["bw_q"]
    else:  # ttm: BB fully inside Keltner Channel
        bu = ind_data["bb_up"][idx]
        bl = ind_data["bb_lo"][idx]
        ku = ind_data["kc_up"][idx]
        kl = ind_data["kc_lo"][idx]
        if None in (bu, bl, ku, kl):
            return False
        return bu < ku and bl > kl


def generate_signals(bars, params, ind_data=None):
    """Return (signals, ind_data). ``signals[t]`` in {None,'long','short'}."""
    if ind_data is None:
        ind_data = compute_indicators(bars, params)

    n = len(bars)
    signals = [None] * n

    allow_long = params.get("allow_long", True)
    allow_short = params.get("allow_short", True)
    min_squeeze = params["min_squeeze_bars"]
    release_window = params["release_window"]
    trend_ema = params.get("trend_ema", 0)
    ema_trend = ind_data["ema_trend"]
    bb_up = ind_data["bb_up"]
    bb_lo = ind_data["bb_lo"]

    prev_squeeze = False
    run_len = 0
    # active setup: dict with release_bar and consumed flag, or None
    setup = None

    for i in range(n):
        sq = _squeeze_state(i, ind_data, params)

        if sq:
            run_len += 1
        else:
            # transition squeeze -> not squeeze : arm a setup if the run was
            # long enough. bar i is the first (release) bar.
            if prev_squeeze and run_len >= min_squeeze:
                setup = {"release_bar": i, "consumed": False}
            run_len = 0

        # expire the setup once the release window has fully elapsed
        if setup is not None and (i - setup["release_bar"]) >= release_window:
            setup = None

        sig = None
        if setup is not None and not setup["consumed"] and not sq:
            c = bars[i].close
            up = bb_up[i]
            lo = bb_lo[i]
            if up is not None and lo is not None:
                trend_ok_long = (
                    trend_ema == 0
                    or (ema_trend is not None and ema_trend[i] is not None and c > ema_trend[i])
                )
                trend_ok_short = (
                    trend_ema == 0
                    or (ema_trend is not None and ema_trend[i] is not None and c < ema_trend[i])
                )
                if allow_long and c > up and trend_ok_long:
                    sig = "long"
                    setup["consumed"] = True
                elif allow_short and c < lo and trend_ok_short:
                    sig = "short"
                    setup["consumed"] = True

        signals[i] = sig
        prev_squeeze = sq

    return signals, ind_data
