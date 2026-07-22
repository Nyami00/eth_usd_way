"""CSV data loading for the prototype backtest.

The CSV schema is: timestamp,open,high,low,close,volume
Timestamps are ISO8601 with a trailing 'Z' (UTC) and ascending order.
Bars are taken as-is (no gap filling); indicator windows use consecutive rows.
"""

import csv
from collections import namedtuple
from datetime import datetime, timezone

Bar = namedtuple("Bar", ["ts", "open", "high", "low", "close", "volume"])


def _parse_ts(raw):
    """Parse an ISO8601Z timestamp into a timezone-aware UTC datetime."""
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_bars(path):
    """Load bars from a CSV file into a list of Bar namedtuples (ascending)."""
    bars = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(
                Bar(
                    ts=_parse_ts(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return bars


def resample(bars, k):
    """Aggregate ``k`` consecutive bars into one.

    open = first, high = max, low = min, close = last, volume = sum,
    timestamp = first bar's timestamp. Trailing partial groups are dropped.

    Groups are aligned so each starts at a UTC hour that is a multiple of ``k``
    (e.g. even hours for k=2). The start hour is re-checked at every group
    boundary, so a bar that would break alignment (e.g. an orphan bar left by a
    data gap) is skipped rather than merged across the gap.
    """
    if k <= 1:
        return list(bars)
    out = []
    n = len(bars)
    i = 0
    while i < n:
        if bars[i].ts.hour % k != 0:
            i += 1  # skip until aligned to an even (mult-of-k) start hour
            continue
        if i + k > n:
            break  # trailing partial group
        group = bars[i : i + k]
        out.append(
            Bar(
                ts=group[0].ts,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            )
        )
        i += k
    return out
