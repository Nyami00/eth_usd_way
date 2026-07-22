"""Unit tests for the v2 indicator additions (hand-computed cases).

Run: python3 -m unittest discover tests -v
"""

import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import indicators as ind
from backtest.data import Bar
from backtest.data import resample


class TestRollingMinMax(unittest.TestCase):
    def test_rolling(self):
        vals = [3, 1, 2, 5, 4]
        mx = ind.rolling_max(vals, 3)
        mn = ind.rolling_min(vals, 3)
        self.assertIsNone(mx[0])
        self.assertIsNone(mx[1])
        self.assertEqual(mx[2], 3)  # max(3,1,2)
        self.assertEqual(mx[3], 5)  # max(1,2,5)
        self.assertEqual(mx[4], 5)  # max(2,5,4)
        self.assertEqual(mn[2], 1)  # min(3,1,2)
        self.assertEqual(mn[3], 1)  # min(1,2,5)
        self.assertEqual(mn[4], 2)  # min(2,5,4)


class TestLinregEndpoint(unittest.TestCase):
    def test_perfect_line(self):
        # Each trailing 3-window of [1,2,3,4,5] is linear slope 1 -> endpoint
        # equals the last value of the window.
        vals = [1, 2, 3, 4, 5]
        out = ind.linreg_endpoint(vals, 3)
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 3.0)
        self.assertAlmostEqual(out[3], 4.0)
        self.assertAlmostEqual(out[4], 5.0)

    def test_non_perfect(self):
        # window [0,0,3]: x=0,1,2 ; slope b=1.5, intercept a=-0.5
        # endpoint at x=2 = -0.5 + 1.5*2 = 2.5
        out = ind.linreg_endpoint([0, 0, 3], 3)
        self.assertAlmostEqual(out[2], 2.5)

    def test_none_in_window(self):
        out = ind.linreg_endpoint([None, 1, 2], 3)
        self.assertIsNone(out[2])


class TestDailyEmaFfill(unittest.TestCase):
    def _bar(self, iso, close):
        ts = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)

    def test_ffill_no_lookahead(self):
        # 3 UTC days, 2 hourly bars each; daily closes = 10, 20, 30.
        bars = [
            self._bar("2025-01-01T00:00:00", 5),
            self._bar("2025-01-01T01:00:00", 10),  # day0 close = 10
            self._bar("2025-01-02T00:00:00", 15),
            self._bar("2025-01-02T01:00:00", 20),  # day1 close = 20
            self._bar("2025-01-03T00:00:00", 25),
            self._bar("2025-01-03T01:00:00", 30),  # day2 close = 30
        ]
        out = ind.daily_ema_ffill(bars, 2)
        # daily EMA(2): ema[0]=None, ema[1]=SMA(10,20)=15,
        #   ema[2]=(2/3)*30 + (1/3)*15 = 25
        # ffill: bar on day D uses ema of day D-1.
        #   day0 bars -> D-1 = none -> None
        #   day1 bars -> ema[day0] = None
        #   day2 bars -> ema[day1] = 15
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertIsNone(out[2])
        self.assertIsNone(out[3])
        self.assertAlmostEqual(out[4], 15.0)
        self.assertAlmostEqual(out[5], 15.0)


class TestResample(unittest.TestCase):
    def _bar(self, iso, o, h, l, c, v):
        ts = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)

    def test_k2_basic(self):
        bars = [
            self._bar("2025-01-01T00:00:00", 10, 12, 9, 11, 100),
            self._bar("2025-01-01T01:00:00", 11, 15, 10, 14, 200),
            self._bar("2025-01-01T02:00:00", 14, 16, 13, 15, 50),
            self._bar("2025-01-01T03:00:00", 15, 17, 12, 13, 70),
            self._bar("2025-01-01T04:00:00", 13, 13, 11, 12, 40),  # trailing partial -> dropped
        ]
        out = resample(bars, 2)
        self.assertEqual(len(out), 2)
        g0 = out[0]
        self.assertEqual(g0.ts, bars[0].ts)
        self.assertEqual(g0.open, 10)
        self.assertEqual(g0.high, 15)  # max(12,15)
        self.assertEqual(g0.low, 9)  # min(9,10)
        self.assertEqual(g0.close, 14)
        self.assertEqual(g0.volume, 300)
        g1 = out[1]
        self.assertEqual(g1.open, 14)
        self.assertEqual(g1.high, 17)
        self.assertEqual(g1.low, 12)
        self.assertEqual(g1.close, 13)
        self.assertEqual(g1.volume, 120)

    def test_odd_start_skipped(self):
        # First bar at an odd hour is skipped to align groups to even hours.
        bars = [
            self._bar("2025-01-01T01:00:00", 1, 1, 1, 1, 1),  # odd -> skipped
            self._bar("2025-01-01T02:00:00", 2, 5, 2, 4, 10),
            self._bar("2025-01-01T03:00:00", 4, 6, 3, 5, 20),
        ]
        out = resample(bars, 2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ts, bars[1].ts)
        self.assertEqual(out[0].open, 2)
        self.assertEqual(out[0].close, 5)
        self.assertEqual(out[0].volume, 30)


if __name__ == "__main__":
    unittest.main()
