"""Unit tests for v5 engine additions: entry cutoff + window-end force-close,
and the leverage-cap flag. Zero-cost synthetic bars with hand-supplied ATR.

Run: python3 -m unittest discover tests -v
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import Bar
from backtest import engine as eng

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def build(rows):
    return [
        Bar(ts=T0 + timedelta(hours=i), open=o, high=h, low=l, close=c, volume=1.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]


def base_params(**over):
    p = {
        "fee": 0.0,
        "slip": 0.0,
        "stop_atr": 2.0,
        "trail_atr": 3.0,
        "scale_out": True,
        "time_stop_bars": 12,
        "allow_long": False,
        "allow_short": True,
    }
    p.update(over)
    return p


class TestEntryEndWindowClose(unittest.TestCase):
    def test_window_end_force_close_and_entry_cutoff(self):
        rows = [
            (100, 100, 100, 100),   # 0 signal -> short
            (100, 102, 99, 101),    # 1 entry fill 100, stop 110
            (101, 103, 100, 102),   # 2
            (102, 104, 101, 103),   # 3 window_end_idx -> force close @103
            (103, 104, 102, 103),   # 4 signal -> would fill bar 5 (past cutoff)
            (103, 104, 102, 103),   # 5 suppressed (ts > entry_end)
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = [None] * len(bars)
        signals[0] = "short"
        signals[4] = "short"
        entry_end = bars[3].ts  # last allowed fill bar is index 3
        r = eng.run_backtest(bars, signals, atr, base_params(), T0, entry_end=entry_end)
        # Only one trade: the bar-4 signal's fill would be at bar 5 (> cutoff).
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertEqual(t["exit_reason"], "window_end")
        self.assertTrue(t["forced_close"])
        self.assertEqual(t["side"], "short")
        # short 300 units @ (100 - 103) = -900 ; r = -900 / (300*10) = -0.3
        self.assertAlmostEqual(t["pnl"], -900.0)
        self.assertAlmostEqual(t["r_multiple"], -0.3)

    def test_no_window_close_when_flat_at_boundary(self):
        # Position exits (stop) before the window end -> no extra window_end trade.
        rows = [
            (100, 100, 100, 100),   # 0 signal
            (100, 100, 100, 100),   # 1 entry fill 100 stop 110
            (105, 111, 104, 108),   # 2 stop hit -> exit
            (100, 100, 100, 100),   # 3 window_end (already flat)
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["short"] + [None] * 3
        r = eng.run_backtest(bars, signals, atr, base_params(), T0, entry_end=bars[3].ts)
        self.assertEqual(len(r["trades"]), 1)
        self.assertEqual(r["trades"][0]["exit_reason"], "stop")


class TestLeverageCapFlag(unittest.TestCase):
    def test_leverage_capped_true(self):
        # stop_dist = 2*0.2 = 0.4 -> qty = 3000/0.4 = 7500 > cap 5000 -> capped.
        rows = [
            (100, 100, 100, 100),        # 0 signal
            (100, 100.2, 99.9, 100.1),   # 1 entry fill 100, stop 100.4
            (100.5, 101, 100.4, 100.6),  # 2 high>=100.4 -> stop exit
            (100, 100, 100, 100),        # 3
        ]
        bars = build(rows)
        atr = [0.2] * len(bars)
        signals = ["short"] + [None] * 3
        r = eng.run_backtest(bars, signals, atr, base_params(), T0)
        self.assertEqual(len(r["trades"]), 1)
        self.assertTrue(r["trades"][0]["leverage_capped"])

    def test_leverage_capped_false(self):
        rows = [
            (100, 100, 100, 100),   # 0 signal
            (100, 100, 100, 100),   # 1 entry fill 100 stop 110 (dist 10 -> qty 300, not capped)
            (105, 111, 104, 108),   # 2 stop exit
            (100, 100, 100, 100),   # 3
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["short"] + [None] * 3
        r = eng.run_backtest(bars, signals, atr, base_params(), T0)
        self.assertEqual(len(r["trades"]), 1)
        self.assertFalse(r["trades"][0]["leverage_capped"])


if __name__ == "__main__":
    unittest.main()
