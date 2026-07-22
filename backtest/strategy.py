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

    # --- v2 filters (computed only when enabled) ---
    # F1: higher-timeframe daily-EMA trend filter (always on in v2).
    if params.get("htf_trend"):
        out["htf_ema"] = ind.daily_ema_ffill(bars, params.get("htf_ema_n", 50))
    else:
        out["htf_ema"] = None

    # F2: TTM momentum oscillator = linreg endpoint of (close - midline),
    # midline = ((HH_m + LL_m)/2 + EMA_m)/2 over an m-bar window.
    if params.get("momentum_filter"):
        m = params.get("mom_n", 20)
        hh = ind.rolling_max(highs, m)
        ll = ind.rolling_min(lows, m)
        ema_m = ind.ema(closes, m)
        delta = [None] * len(closes)
        for i in range(len(closes)):
            if hh[i] is not None and ll[i] is not None and ema_m[i] is not None:
                midline = ((hh[i] + ll[i]) / 2.0 + ema_m[i]) / 2.0
                delta[i] = closes[i] - midline
        out["osc"] = ind.linreg_endpoint(delta, m)
    else:
        out["osc"] = None

    # F3: volume confirmation vs SMA(volume, 20).
    if params.get("volume_filter"):
        volumes = [b.volume for b in bars]
        out["sma_vol"] = ind.sma(volumes, params.get("vol_sma_n", 20))
    else:
        out["sma_vol"] = None

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

    htf_trend = bool(params.get("htf_trend"))
    htf_ema = ind_data.get("htf_ema")
    momentum_filter = bool(params.get("momentum_filter"))
    osc = ind_data.get("osc")
    volume_filter = bool(params.get("volume_filter"))
    sma_vol = ind_data.get("sma_vol")
    vol_mult = params.get("vol_mult", 1.3)
    # confirm_bars: 0 = emit on the breakout bar (fill next bar). 1 = require the
    # bar after the breakout to also close beyond the same-direction band
    # (recomputed at that bar); on confirmation emit there (engine fills at the
    # bar after). The setup is consumed on the breakout bar either way.
    confirm_bars = int(params.get("confirm_bars", 0))
    # ema_slope_filter: require the daily EMA to slope in the trade direction at
    # signal time (long: EMA[D-1] > EMA[D-2]; short: EMA[D-1] < EMA[D-2]).
    ema_slope_filter = bool(params.get("ema_slope_filter"))
    htf_slope = ind_data.get("htf_ema_slope")

    prev_squeeze = False
    run_len = 0
    # active setup: dict with release_bar and consumed flag, or None
    setup = None
    # pending confirmation (confirm_bars>=1): {"dir":.., "confirm_at": bar index}
    pending = None
    # episode tracking (v4b): the most recent qualifying squeeze-release bar
    # as of bar i (a new qualifying release replaces the episode).
    episode_release = [None] * n
    current_release = None

    for i in range(n):
        sq = _squeeze_state(i, ind_data, params)

        if sq:
            run_len += 1
        else:
            # transition squeeze -> not squeeze : arm a setup if the run was
            # long enough. bar i is the first (release) bar.
            if prev_squeeze and run_len >= min_squeeze:
                setup = {"release_bar": i, "consumed": False}
                current_release = i  # new episode starts here
            run_len = 0

        episode_release[i] = current_release

        # expire the setup once the release window has fully elapsed
        if setup is not None and (i - setup["release_bar"]) >= release_window:
            setup = None

        sig = None

        # (1) Resolve a pending confirmation from the previous breakout bar.
        if pending is not None and pending["confirm_at"] == i:
            up_c = bb_up[i]
            lo_c = bb_lo[i]
            cc = bars[i].close
            if pending["dir"] == "long" and up_c is not None and cc > up_c:
                sig = "long"
            elif pending["dir"] == "short" and lo_c is not None and cc < lo_c:
                sig = "short"
            pending = None  # consumed whether or not confirmation succeeded

        # (2) Otherwise detect a fresh breakout on this bar.
        if sig is None and pending is None and setup is not None and not setup["consumed"] and not sq:
            c = bars[i].close
            up = bb_up[i]
            lo = bb_lo[i]
            if up is not None and lo is not None:
                # v1 same-timeframe EMA trend filter (disabled when trend_ema==0)
                trend_ok_long = (
                    trend_ema == 0
                    or (ema_trend is not None and ema_trend[i] is not None and c > ema_trend[i])
                )
                trend_ok_short = (
                    trend_ema == 0
                    or (ema_trend is not None and ema_trend[i] is not None and c < ema_trend[i])
                )
                # F1: higher-timeframe daily-EMA trend filter
                if htf_trend:
                    he = htf_ema[i] if htf_ema is not None else None
                    htf_ok_long = he is not None and c > he
                    htf_ok_short = he is not None and c < he
                else:
                    htf_ok_long = htf_ok_short = True
                # F2: momentum oscillator
                if momentum_filter:
                    o = osc[i] if osc is not None else None
                    mom_ok_long = o is not None and o > 0
                    mom_ok_short = o is not None and o < 0
                else:
                    mom_ok_long = mom_ok_short = True
                # F3: volume confirmation on the breakout bar
                if volume_filter:
                    sv = sma_vol[i] if sma_vol is not None else None
                    vol_ok = sv is not None and bars[i].volume >= vol_mult * sv
                else:
                    vol_ok = True
                # daily-EMA slope filter
                if ema_slope_filter:
                    sl = htf_slope[i] if htf_slope is not None else None
                    slope_ok_long = sl is not None and sl > 0
                    slope_ok_short = sl is not None and sl < 0
                else:
                    slope_ok_long = slope_ok_short = True

                long_break = (
                    allow_long and c > up and trend_ok_long and htf_ok_long
                    and mom_ok_long and vol_ok and slope_ok_long
                )
                short_break = (
                    allow_short and c < lo and trend_ok_short and htf_ok_short
                    and mom_ok_short and vol_ok and slope_ok_short
                )
                if long_break:
                    setup["consumed"] = True
                    if confirm_bars >= 1:
                        pending = {"dir": "long", "confirm_at": i + 1}
                    else:
                        sig = "long"
                elif short_break:
                    setup["consumed"] = True
                    if confirm_bars >= 1:
                        pending = {"dir": "short", "confirm_at": i + 1}
                    else:
                        sig = "short"

        signals[i] = sig
        prev_squeeze = sq

    # Pull-back short re-entry trigger (v4b): close crosses down through BBmid
    # while the HTF filter still permits shorts (close < daily EMA50).
    closes = [b.close for b in bars]
    bb_mid = ind_data["bb_mid"]
    pullback_short = [False] * n
    for i in range(1, n):
        m = bb_mid[i]
        mp = bb_mid[i - 1]
        if m is None or mp is None:
            continue
        he = htf_ema[i] if htf_ema is not None else None
        if closes[i] < m and closes[i - 1] >= mp and he is not None and closes[i] < he:
            pullback_short[i] = True

    ind_data["episode_release"] = episode_release
    ind_data["pullback_short"] = pullback_short

    return signals, ind_data
