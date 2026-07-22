"""Unit tests for the v6 confirm_bars breakout-confirmation logic.

A hand-built ind_data cache drives a controlled squeeze -> release -> breakout
so the confirmation branch can be checked deterministically.

Run: python3 -m unittest discover tests -v
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import Bar
from backtest import strategy as strat

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make(closes):
    return [
        Bar(ts=T0 + timedelta(hours=i), open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]


def cache(n, bb_up, bb_lo, bw_pct):
    return {
        "bb_up": [bb_up] * n,
        "bb_lo": [bb_lo] * n,
        "bb_mid": [(bb_up + bb_lo) / 2.0] * n,
        "bw_pct": bw_pct,
        "ema_trend": None,
        "htf_ema": None,
        "osc": None,
        "sma_vol": None,
    }


def base_params(confirm):
    return {
        "squeeze_mode": "pctile",
        "bw_q": 50.0,
        "bw_lookback": 120,
        "min_squeeze_bars": 4,
        "release_window": 3,
        "trend_ema": 0,
        "htf_trend": False,
        "momentum_filter": False,
        "volume_filter": False,
        "allow_long": True,
        "allow_short": True,
        "confirm_bars": confirm,
    }


# Squeeze on bars 0..3 (bw_pct 10 < 50), released bar 4+. Bands 90/100.
BW = [10, 10, 10, 10, 90, 90, 90, 90]


class TestConfirmBars(unittest.TestCase):
    def test_confirm0_signal_on_breakout_bar(self):
        closes = [95, 95, 95, 95, 101, 99, 95, 95]  # bar4 close 101 > 100
        bars = make(closes)
        sig, _ = strat.generate_signals(bars, base_params(0), ind_data=cache(8, 100, 90, BW))
        self.assertEqual(sig[4], "long")
        self.assertTrue(all(sig[i] is None for i in range(8) if i != 4))

    def test_confirm1_success_signals_on_confirm_bar(self):
        closes = [95, 95, 95, 95, 101, 102, 95, 95]  # bar5 close 102 > 100 -> confirms
        bars = make(closes)
        sig, _ = strat.generate_signals(bars, base_params(1), ind_data=cache(8, 100, 90, BW))
        self.assertIsNone(sig[4])       # no fill-signal on the breakout bar
        self.assertEqual(sig[5], "long")  # emitted on the confirmation bar
        self.assertTrue(all(sig[i] is None for i in range(8) if i != 5))

    def test_confirm1_failure_no_signal(self):
        closes = [95, 95, 95, 95, 101, 99, 95, 95]  # bar5 close 99 < 100 -> fails
        bars = make(closes)
        sig, _ = strat.generate_signals(bars, base_params(1), ind_data=cache(8, 100, 90, BW))
        self.assertTrue(all(s is None for s in sig))  # setup consumed, no signal


class TestEmaSlopeFilter(unittest.TestCase):
    def _run(self, slope_val):
        # confirm_bars=0 long breakout at bar 4; slope filter gates it.
        closes = [95, 95, 95, 95, 101, 99, 95, 95]
        bars = make(closes)
        c = cache(8, 100, 90, BW)
        c["htf_ema_slope"] = [slope_val] * 8
        p = base_params(0)
        p["ema_slope_filter"] = True
        return strat.generate_signals(bars, p, ind_data=c)[0]

    def test_slope_up_allows_long(self):
        self.assertEqual(self._run(0.5)[4], "long")  # slope>0 permits long

    def test_slope_down_blocks_long(self):
        self.assertTrue(all(s is None for s in self._run(-0.5)))  # slope<0 blocks long

    def test_slope_none_blocks(self):
        self.assertTrue(all(s is None for s in self._run(None)))


if __name__ == "__main__":
    unittest.main()
