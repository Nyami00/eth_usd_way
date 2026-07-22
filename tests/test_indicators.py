"""Unit tests for indicators, verified against hand-computed synthetic series.

Run: python3 -m unittest discover tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import indicators as ind


class TestSMA(unittest.TestCase):
    def test_sma3(self):
        vals = [1, 2, 3, 4, 5, 6]
        out = ind.sma(vals, 3)
        # first two undefined, then trailing means
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)
        self.assertAlmostEqual(out[5], 5.0)


class TestStdev(unittest.TestCase):
    def test_stdev3_pop(self):
        vals = [1, 2, 3, 4, 5, 6]
        out = ind.stdev(vals, 3, ddof=0)
        # window [1,2,3]: mean 2, var = (1+0+1)/3 = 2/3
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], math.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(out[3], math.sqrt(2.0 / 3.0))


class TestEMA(unittest.TestCase):
    def test_ema3(self):
        vals = [1, 2, 3, 4, 5, 6]
        out = ind.ema(vals, 3)
        # seed = SMA(1,2,3) = 2 at index 2; alpha = 2/4 = 0.5
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 0.5 * 4 + 0.5 * 2.0)  # 3.0
        self.assertAlmostEqual(out[4], 0.5 * 5 + 0.5 * 3.0)  # 4.0
        self.assertAlmostEqual(out[5], 0.5 * 6 + 0.5 * 4.0)  # 5.0


class TestBollinger(unittest.TestCase):
    def test_bb3(self):
        vals = [1, 2, 3, 4, 5, 6]
        mid, up, lo = ind.bollinger(vals, 3, 2.0)
        sd = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(mid[2], 2.0)
        self.assertAlmostEqual(up[2], 2.0 + 2.0 * sd)
        self.assertAlmostEqual(lo[2], 2.0 - 2.0 * sd)
        # bandwidth = (up-lo)/mid = 4*sd/mid
        bw = ind.bandwidth(up, lo, mid)
        self.assertAlmostEqual(bw[2], (up[2] - lo[2]) / mid[2])
        self.assertAlmostEqual(bw[2], 4.0 * sd / 2.0)


class TestATR(unittest.TestCase):
    def test_atr2(self):
        # Hand-computed TR series:
        #  i0: H10 L8  C9    TR = 10-8 = 2
        #  i1: H12 L9  C11   TR = max(3, |12-9|=3, |9-9|=0) = 3
        #  i2: H13 L11 C12   TR = max(2, |13-11|=2, |11-11|=0) = 2
        #  i3: H12 L10 C10.5 TR = max(2, |12-12|=0, |10-12|=2) = 2
        highs = [10, 12, 13, 12]
        lows = [8, 9, 11, 10]
        closes = [9, 11, 12, 10.5]
        tr = ind.true_range(highs, lows, closes)
        self.assertAlmostEqual(tr[0], 2.0)
        self.assertAlmostEqual(tr[1], 3.0)
        self.assertAlmostEqual(tr[2], 2.0)
        self.assertAlmostEqual(tr[3], 2.0)
        atr = ind.atr(highs, lows, closes, 2)
        # seed at i1 = mean(2,3) = 2.5
        self.assertIsNone(atr[0])
        self.assertAlmostEqual(atr[1], 2.5)
        # i2 = (2.5*1 + 2)/2 = 2.25
        self.assertAlmostEqual(atr[2], 2.25)
        # i3 = (2.25*1 + 2)/2 = 2.125
        self.assertAlmostEqual(atr[3], 2.125)


class TestKeltner(unittest.TestCase):
    def test_kc_uses_ema_and_atr(self):
        highs = [10, 12, 13, 12, 14, 13]
        lows = [8, 9, 11, 10, 11, 11]
        closes = [9, 11, 12, 10.5, 13, 12]
        mid, up, lo = ind.keltner(highs, lows, closes, 3, 1.5)
        ema = ind.ema(closes, 3)
        atr = ind.atr(highs, lows, closes, 3)
        for i in range(len(closes)):
            if ema[i] is not None and atr[i] is not None:
                self.assertAlmostEqual(mid[i], ema[i])
                self.assertAlmostEqual(up[i], ema[i] + 1.5 * atr[i])
                self.assertAlmostEqual(lo[i], ema[i] - 1.5 * atr[i])


class TestBandWidthPercentile(unittest.TestCase):
    def test_percentile_rank(self):
        # BandWidth-like series; lookback 4.
        bw = [None, 0.5, 0.4, 0.3, 0.2, 0.6]
        out = ind.bw_percentile(bw, 4)
        # index < 3 -> None (need full window of 4 non-None)
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertIsNone(out[2])
        # index 3 not defined: window [0.5,0.4,0.3] has a None at idx0? window is
        # bw[0:4] = [None,0.5,0.4,0.3] -> contains None -> None
        self.assertIsNone(out[3])
        # index 4: window bw[1:5] = [0.5,0.4,0.3,0.2]; current 0.2 is min
        # rank = 100 * (#<=0.2) / 4 = 100 * 1/4 = 25
        self.assertAlmostEqual(out[4], 25.0)
        # index 5: window bw[2:6] = [0.4,0.3,0.2,0.6]; current 0.6 is max
        # rank = 100 * 4/4 = 100
        self.assertAlmostEqual(out[5], 100.0)


if __name__ == "__main__":
    unittest.main()
