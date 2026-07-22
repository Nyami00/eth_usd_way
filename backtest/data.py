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
