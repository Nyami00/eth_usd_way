"""Unit tests for v4 engine mechanics: scale-out, breakeven stop, time stop.

Uses tiny synthetic bar series with zero fees/slippage so the arithmetic is
exact, and a hand-supplied ATR array so the stop distance is a round number.

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


def mk(o, h, l, c):
    return (o, h, l, c)


def build(rows):
    """rows: list of (o,h,l,c). Returns list[Bar] with hourly timestamps."""
    bars = []
    for i, (o, h, l, c) in enumerate(rows):
        bars.append(Bar(ts=T0 + timedelta(hours=i), open=o, high=h, low=l, close=c, volume=1.0))
    return bars


# Zero-cost params; stop_atr=2, trail_atr=3, scale-out on, time stop at 12 bars.
def params(allow_long, allow_short):
    return {
        "fee": 0.0,
        "slip": 0.0,
        "stop_atr": 2.0,
        "trail_atr": 3.0,
        "scale_out": True,
        "time_stop_bars": 12,
        "allow_long": allow_long,
        "allow_short": allow_short,
    }


class TestShortScaleOut(unittest.TestCase):
    def test_short_scaleout_then_breakeven_stop(self):
        # atr=5 -> stop_dist=10. Short entry fill 100; initial stop 110;
        # +1.5R level = 100 - 15 = 85. qty = 100000*0.03/10 = 300.
        rows = [
            mk(100, 100, 100, 100),  # 0 signal bar
            mk(100, 100, 100, 100),  # 1 entry (fill=100)
            mk(90, 90, 85, 86),      # 2 low<=85 -> scale out 150 @ 85; stop->BE 100
            mk(95, 101, 95, 100),    # 3 high>=100 (BE stop) -> exit remainder @ 100
            mk(100, 100, 100, 100),  # 4 padding
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["short"] + [None] * (len(bars) - 1)
        r = eng.run_backtest(bars, signals, atr, params(False, True), T0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertTrue(t["scaled_out"])
        self.assertEqual(t["exit_reason"], "stop")
        # 150 units @ +15 = 2250 realized on scale; remainder exits at breakeven -> 0.
        self.assertAlmostEqual(t["pnl"], 2250.0)
        self.assertAlmostEqual(t["r_multiple"], 0.75)  # 2250 / (300*10)
        self.assertEqual(t["side"], "short")


class TestLongScaleOut(unittest.TestCase):
    def test_long_scaleout_then_trail_stop(self):
        # Long entry fill 100; stop_dist 10; +1.5R level = 115; qty 300.
        rows = [
            mk(100, 100, 100, 100),   # 0 signal
            mk(100, 100, 100, 100),   # 1 entry
            mk(110, 120, 110, 118),   # 2 high>=115 -> scale 150 @115; trail-> 120-15=105
            mk(108, 108, 104, 105),   # 3 low<=105 -> exit remainder @105
            mk(100, 100, 100, 100),   # 4 padding
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["long"] + [None] * (len(bars) - 1)
        r = eng.run_backtest(bars, signals, atr, params(True, False), T0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertTrue(t["scaled_out"])
        self.assertEqual(t["exit_reason"], "stop")
        # scale: 150 @ +15 = 2250 ; remainder: 150 @ +5 = 750 ; total 3000.
        self.assertAlmostEqual(t["pnl"], 3000.0)
        self.assertAlmostEqual(t["r_multiple"], 1.0)  # 3000 / (300*10)


class TestShortTimeStop(unittest.TestCase):
    def test_short_time_stop(self):
        # Short stays underwater (price ~103), never scales/stops; time stop at
        # bar 13 (12 bars after entry@1) -> exit at bar 14 open (104).
        rows = [mk(100, 100, 100, 100)]  # 0 signal
        rows.append(mk(100, 105, 100, 103))  # 1 entry fill=100
        for _ in range(2, 13):  # bars 2..12
            rows.append(mk(102, 105, 100, 103))
        rows.append(mk(102, 105, 100, 103))  # 13: c-entry=12 -> R=(100-103)/10<0 -> pending
        rows.append(mk(104, 106, 103, 105))  # 14: exit at open 104
        rows.append(mk(104, 104, 104, 104))  # 15 padding
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["short"] + [None] * (len(bars) - 1)
        r = eng.run_backtest(bars, signals, atr, params(False, True), T0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertFalse(t["scaled_out"])
        self.assertEqual(t["exit_reason"], "time_stop")
        self.assertEqual(t["bars_held"], 13)  # entry bar 1 -> exit bar 14
        # 300 units short @ (100-104) = -1200.
        self.assertAlmostEqual(t["pnl"], -1200.0)
        self.assertAlmostEqual(t["r_multiple"], -0.4)


class TestLongTimeStop(unittest.TestCase):
    def test_long_time_stop(self):
        rows = [mk(100, 100, 100, 100)]  # 0 signal
        rows.append(mk(100, 100, 95, 97))  # 1 entry fill=100
        for _ in range(2, 13):
            rows.append(mk(98, 99, 95, 97))
        rows.append(mk(98, 99, 95, 97))    # 13: R=(97-100)/10<0 -> pending
        rows.append(mk(96, 97, 94, 95))    # 14: exit at open 96
        rows.append(mk(96, 96, 96, 96))    # 15 padding
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["long"] + [None] * (len(bars) - 1)
        r = eng.run_backtest(bars, signals, atr, params(True, False), T0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertFalse(t["scaled_out"])
        self.assertEqual(t["exit_reason"], "time_stop")
        self.assertEqual(t["bars_held"], 13)
        self.assertAlmostEqual(t["pnl"], -1200.0)  # 300 @ (96-100)
        self.assertAlmostEqual(t["r_multiple"], -0.4)


class TestStopBeforeScaleConservative(unittest.TestCase):
    def test_stop_wins_when_both_trigger(self):
        # A wide bar where high hits the stop AND low hits the +1.5R level:
        # stop must win (conservative), no scale-out recorded.
        rows = [
            mk(100, 100, 100, 100),   # 0 signal
            mk(100, 100, 100, 100),   # 1 entry short fill 100, stop 110
            mk(105, 111, 84, 108),    # 2 high>=110 (stop) and low<=85 (scale)
            mk(100, 100, 100, 100),   # 3 padding
        ]
        bars = build(rows)
        atr = [5.0] * len(bars)
        signals = ["short"] + [None] * (len(bars) - 1)
        r = eng.run_backtest(bars, signals, atr, params(False, True), T0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertFalse(t["scaled_out"])
        self.assertEqual(t["exit_reason"], "stop")
        # exit at max(open=105, stop=110) = 110 ; short pnl = 300*(100-110) = -3000
        self.assertAlmostEqual(t["pnl"], -3000.0)
        self.assertAlmostEqual(t["r_multiple"], -1.0)


if __name__ == "__main__":
    unittest.main()
