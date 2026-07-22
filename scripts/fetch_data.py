#!/usr/bin/env python3
"""Fetch ETH/USD OHLCV candles from Coinbase (primary) and CryptoCompare (fallback).

Pure Python 3.11 standard library only. No third-party packages.

Outputs:
  data/eth_usd_15m.csv
  data/eth_usd_1h.csv
  data/eth_usd_1d.csv
  data/validation.json

Designed to run on a GitHub Actions runner with unrestricted internet.
"""

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

START = datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

# Granularity in seconds -> output filename
GRANULARITIES = {
    900: "eth_usd_15m.csv",   # 15 minutes
    3600: "eth_usd_1h.csv",   # 1 hour
    86400: "eth_usd_1d.csv",  # 1 day
}

GRAN_LABEL = {900: "15m", 3600: "1h", 86400: "1d"}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)

COINBASE_PRODUCT = "ETH-USD"
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_MAX_CANDLES = 300  # Coinbase returns at most 300 candles per request

USER_AGENT = "eth-usd-way-fetcher/1.0 (+https://github.com; python-stdlib)"

REQUEST_SLEEP = 0.2          # seconds between successful requests
MAX_RETRIES = 5              # attempts per request
BACKOFF = [1, 2, 4, 8, 16]  # exponential backoff seconds (index = retry number)

HEADER = ["timestamp", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #

def iso_z(dt):
    """Format a timezone-aware UTC datetime as ISO8601 with a trailing Z."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_to_iso(epoch_sec):
    """Convert an integer epoch (seconds, UTC) to ISO8601 Z string."""
    dt = datetime.fromtimestamp(int(epoch_sec), tz=timezone.utc)
    return iso_z(dt)


def iso_to_epoch(iso_str):
    """Convert one of our ISO8601 Z strings back to an integer epoch (seconds)."""
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def http_get_json(url):
    """GET a URL and parse JSON, with retry/backoff on 429/5xx/URLError.

    Returns the parsed JSON object on success. Raises the last exception on
    persistent failure.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # Retry on rate limiting and server errors; give up on other 4xx.
            if exc.code == 429 or 500 <= exc.code < 600:
                _backoff_sleep(attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            _backoff_sleep(attempt)
            continue
        except json.JSONDecodeError as exc:
            # Occasionally an edge/proxy returns a partial or HTML body; retry.
            last_exc = exc
            _backoff_sleep(attempt)
            continue
    # Exhausted retries.
    raise last_exc if last_exc is not None else RuntimeError("request failed")


def _backoff_sleep(attempt):
    idx = attempt if attempt < len(BACKOFF) else len(BACKOFF) - 1
    time.sleep(BACKOFF[idx])


# --------------------------------------------------------------------------- #
# Coinbase source
# --------------------------------------------------------------------------- #

def fetch_coinbase(granularity, start, end):
    """Fetch candles from Coinbase Exchange for [start, end].

    Returns a list of rows: [iso_ts, open, high, low, close, volume].
    Raises on total failure so the caller can decide to fall back.

    Coinbase returns each candle as:
        [time_epoch_sec, low, high, open, close, volume]
    Note the order: low, high, open (open comes AFTER high).
    """
    rows = []
    window = timedelta(seconds=granularity * COINBASE_MAX_CANDLES)

    cur = start
    while cur < end:
        win_end = cur + window
        if win_end > end:
            win_end = end

        params = "?granularity={g}&start={s}&end={e}".format(
            g=granularity, s=iso_z(cur), e=iso_z(win_end)
        )
        url = COINBASE_URL.format(product=COINBASE_PRODUCT) + params

        data = http_get_json(url)
        if not isinstance(data, list):
            # Coinbase returns {"message": "..."} on some errors.
            raise RuntimeError("Coinbase unexpected response: {!r}".format(data)[:200])

        for candle in data:
            # candle = [time, low, high, open, close, volume]
            t = int(candle[0])
            low = float(candle[1])
            high = float(candle[2])
            open_ = float(candle[3])
            close = float(candle[4])
            volume = float(candle[5])
            rows.append([
                epoch_to_iso(t),
                open_, high, low, close, volume,
            ])

        time.sleep(REQUEST_SLEEP)

        # Advance forward. Overlap by one bar is fine (we dedup); the key point
        # is that cur strictly increases so the loop always terminates.
        cur = win_end
        if window.total_seconds() <= 0:
            break  # defensive: never possible with our granularities

    return rows


# --------------------------------------------------------------------------- #
# CryptoCompare fallback source
# --------------------------------------------------------------------------- #

CC_ENDPOINT = {
    3600: "https://min-api.cryptocompare.com/data/v2/histohour",
    86400: "https://min-api.cryptocompare.com/data/v2/histoday",
    # 15m (900s) has no free multi-week minute endpoint -> handled as unavailable.
}
CC_LIMIT = 2000  # max candles per CryptoCompare request


def fetch_cryptocompare(granularity, start, end):
    """Fetch candles from CryptoCompare for [start, end], paginating backwards.

    Returns a list of rows: [iso_ts, open, high, low, close, volume].
    Raises RuntimeError for granularities with no suitable free endpoint (15m).
    """
    endpoint = CC_ENDPOINT.get(granularity)
    if endpoint is None:
        raise RuntimeError(
            "CryptoCompare fallback unavailable for granularity {}s".format(granularity)
        )

    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())

    rows = []
    seen = set()
    to_ts = end_epoch

    # Paginate backwards until we cover the start, or the API stops giving data.
    while to_ts >= start_epoch:
        url = "{ep}?fsym=ETH&tsym=USD&limit={lim}&toTs={ts}".format(
            ep=endpoint, lim=CC_LIMIT, ts=to_ts
        )
        payload = http_get_json(url)

        if not isinstance(payload, dict) or payload.get("Response") == "Error":
            msg = payload.get("Message") if isinstance(payload, dict) else str(payload)
            raise RuntimeError("CryptoCompare error: {}".format(msg))

        data = (payload.get("Data") or {}).get("Data") or []
        if not data:
            break

        oldest = None
        added = False
        for bar in data:
            t = int(bar["time"])
            if oldest is None or t < oldest:
                oldest = t
            if t < start_epoch or t > end_epoch:
                continue
            if t in seen:
                continue
            # CryptoCompare pads with zero-value bars outside real history.
            o = float(bar["open"])
            h = float(bar["high"])
            low = float(bar["low"])
            c = float(bar["close"])
            v = float(bar.get("volumefrom", 0.0))
            if o <= 0 and h <= 0 and low <= 0 and c <= 0:
                continue
            seen.add(t)
            rows.append([epoch_to_iso(t), o, h, low, c, v])
            added = True

        time.sleep(REQUEST_SLEEP)

        if oldest is None:
            break
        # Step to just before the oldest bar returned to walk further back.
        next_to_ts = oldest - granularity
        if next_to_ts >= to_ts:
            # No progress -> stop to avoid an infinite loop.
            break
        to_ts = next_to_ts
        if oldest <= start_epoch and not added:
            break

    return rows


# --------------------------------------------------------------------------- #
# Cleaning / dedup
# --------------------------------------------------------------------------- #

def clean_rows(rows):
    """Sort ascending by timestamp, drop dupes (keep first), drop non-positive prices."""
    # Keep first occurrence per timestamp.
    by_ts = {}
    order = []
    for r in rows:
        ts = r[0]
        if ts not in by_ts:
            by_ts[ts] = r
            order.append(ts)

    cleaned = []
    for ts in sorted(by_ts.keys()):
        r = by_ts[ts]
        o, h, low, c = r[1], r[2], r[3], r[4]
        if o <= 0 or h <= 0 or low <= 0 or c <= 0:
            continue
        cleaned.append(r)
    return cleaned


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for r in rows:
            # Format numbers without forcing scientific notation.
            writer.writerow([
                r[0],
                _fmt_num(r[1]),
                _fmt_num(r[2]),
                _fmt_num(r[3]),
                _fmt_num(r[4]),
                _fmt_num(r[5]),
            ])


def _fmt_num(x):
    # Use repr-ish formatting that round-trips floats without trailing noise.
    if x == int(x):
        # Preserve a plain integer look for whole numbers (e.g. volume 0).
        return "{:.1f}".format(x) if abs(x) < 1e15 else repr(x)
    return repr(x)


# --------------------------------------------------------------------------- #
# Validation / gap analysis
# --------------------------------------------------------------------------- #

def analyze_gaps(rows, granularity):
    """Count gaps where the delta between consecutive bars != granularity.

    Returns (num_gaps, largest_gap_hours).
    """
    if len(rows) < 2:
        return 0, 0.0
    num_gaps = 0
    largest = 0
    prev = iso_to_epoch(rows[0][0])
    for r in rows[1:]:
        cur = iso_to_epoch(r[0])
        delta = cur - prev
        if delta != granularity:
            num_gaps += 1
            missing = delta - granularity
            if missing > largest:
                largest = missing
        prev = cur
    return num_gaps, round(largest / 3600.0, 4)


def compare_daily_closes(coinbase_daily, crypto_daily):
    """Compare closing prices on overlapping days between the two sources.

    Inputs are lists of [iso_ts, o, h, l, c, v]. Returns a dict.
    """
    cb = {r[0][:10]: r[4] for r in coinbase_daily}  # date -> close
    cc = {r[0][:10]: r[4] for r in crypto_daily}
    common = sorted(set(cb.keys()) & set(cc.keys()))
    max_rel = 0.0
    worst_day = None
    for day in common:
        a = cb[day]
        b = cc[day]
        denom = abs(a) if a != 0 else 1.0
        rel = abs(a - b) / denom
        if rel > max_rel:
            max_rel = rel
            worst_day = day
    return {
        "days_compared": len(common),
        "max_relative_close_diff": round(max_rel, 6),
        "worst_day": worst_day,
        "note": "Different venues; recorded for reference only, not a failure condition.",
    }


# --------------------------------------------------------------------------- #
# Per-granularity orchestration
# --------------------------------------------------------------------------- #

def acquire_granularity(granularity, end, keep_cc_daily=None):
    """Fetch one granularity, preferring Coinbase and falling back to CryptoCompare.

    Returns a dict describing the result:
        {
          "rows": [...cleaned rows...],
          "source": "coinbase" | "cryptocompare" | None,
          "note": str or None,
        }
    keep_cc_daily: if a dict is passed and this is the daily granularity, the
    raw CryptoCompare daily rows are stored under key "rows" for cross-checks.
    """
    label = GRAN_LABEL[granularity]
    result = {"rows": [], "source": None, "note": None}

    # 1) Primary: Coinbase.
    try:
        print("[{}] fetching from Coinbase...".format(label))
        cb_rows = fetch_coinbase(granularity, START, end)
        cb_rows = clean_rows(cb_rows)
        if cb_rows:
            result["rows"] = cb_rows
            result["source"] = "coinbase"
            print("[{}] Coinbase returned {} rows".format(label, len(cb_rows)))
        else:
            print("[{}] Coinbase returned 0 rows; will try fallback".format(label))
    except Exception as exc:  # noqa: BLE001 - we intentionally fall back on anything
        print("[{}] Coinbase failed: {}".format(label, exc))

    # 2) Fallback: CryptoCompare (also used to seed daily cross-check).
    if result["source"] is None:
        try:
            print("[{}] fetching from CryptoCompare (fallback)...".format(label))
            cc_rows = fetch_cryptocompare(granularity, START, end)
            cc_rows = clean_rows(cc_rows)
            if cc_rows:
                result["rows"] = cc_rows
                result["source"] = "cryptocompare"
                print("[{}] CryptoCompare returned {} rows".format(label, len(cc_rows)))
                if keep_cc_daily is not None:
                    keep_cc_daily["rows"] = cc_rows
            else:
                result["note"] = "No data from Coinbase or CryptoCompare."
                print("[{}] CryptoCompare returned 0 rows".format(label))
        except Exception as exc:  # noqa: BLE001
            result["note"] = "Coinbase failed and fallback unavailable: {}".format(exc)
            print("[{}] CryptoCompare fallback failed: {}".format(label, exc))

    return result


def fetch_cc_daily_for_crosscheck(end):
    """Best-effort fetch of CryptoCompare daily candles for the cross-source check."""
    try:
        rows = fetch_cryptocompare(86400, START, end)
        return clean_rows(rows)
    except Exception as exc:  # noqa: BLE001
        print("[crosscheck] CryptoCompare daily fetch failed: {}".format(exc))
        return []


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_ts = datetime.now(timezone.utc)
    end = fetch_ts

    print("ETH/USD OHLCV fetch")
    print("  range : {} -> {}".format(iso_z(START), iso_z(end)))
    print("  output: {}".format(DATA_DIR))
    print("")

    results = {}
    cc_daily_from_fallback = {}  # populated only if daily fell back to CryptoCompare

    for granularity in (3600, 900, 86400):  # do 1h first (it's the success gate)
        keep = cc_daily_from_fallback if granularity == 86400 else None
        results[granularity] = acquire_granularity(granularity, end, keep_cc_daily=keep)

    # Write CSVs for every granularity that produced rows.
    for granularity, fname in GRANULARITIES.items():
        res = results[granularity]
        path = os.path.join(DATA_DIR, fname)
        if res["rows"]:
            write_csv(path, res["rows"])
            print("[{}] wrote {} rows -> {}".format(GRAN_LABEL[granularity], len(res["rows"]), path))
        else:
            print("[{}] no data; CSV not written".format(GRAN_LABEL[granularity]))

    # ---- Cross-source daily close comparison ----------------------------- #
    daily_res = results[86400]
    coinbase_daily = daily_res["rows"] if daily_res["source"] == "coinbase" else []

    if daily_res["source"] == "cryptocompare":
        crypto_daily = cc_daily_from_fallback.get("rows", [])
        # We have no independent Coinbase daily to compare against in this case.
    else:
        crypto_daily = fetch_cc_daily_for_crosscheck(end)

    if coinbase_daily and crypto_daily:
        cross = compare_daily_closes(coinbase_daily, crypto_daily)
    else:
        cross = {
            "days_compared": 0,
            "max_relative_close_diff": None,
            "worst_day": None,
            "note": "Could not obtain daily candles from both venues for comparison.",
        }

    # ---- Build validation.json ------------------------------------------- #
    validation = {
        "fetch_utc": iso_z(fetch_ts),
        "range_start": iso_z(START),
        "range_end": iso_z(end),
        "cross_source_daily_close_check": cross,
        "granularities": {},
    }

    for granularity in (900, 3600, 86400):
        res = results[granularity]
        rows = res["rows"]
        label = GRAN_LABEL[granularity]
        if rows:
            num_gaps, largest_gap_hours = analyze_gaps(rows, granularity)
            validation["granularities"][label] = {
                "granularity_seconds": granularity,
                "source": res["source"],
                "row_count": len(rows),
                "first_timestamp": rows[0][0],
                "last_timestamp": rows[-1][0],
                "num_gaps": num_gaps,
                "largest_gap_hours": largest_gap_hours,
                "note": res["note"],
            }
        else:
            validation["granularities"][label] = {
                "granularity_seconds": granularity,
                "source": None,
                "row_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "num_gaps": None,
                "largest_gap_hours": None,
                "note": res["note"] or "No data acquired.",
            }

    vpath = os.path.join(DATA_DIR, "validation.json")
    with open(vpath, "w") as fh:
        json.dump(validation, fh, indent=2, sort_keys=True)
    print("wrote validation -> {}".format(vpath))

    # ---- Human-readable summary ------------------------------------------ #
    print("")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("fetch_utc : {}".format(validation["fetch_utc"]))
    print("range     : {} -> {}".format(validation["range_start"], validation["range_end"]))
    for label in ("15m", "1h", "1d"):
        g = validation["granularities"][label]
        if g["row_count"]:
            print(
                "  {:<4} src={:<13} rows={:<6} {} .. {} gaps={} maxgap={}h".format(
                    label,
                    str(g["source"]),
                    g["row_count"],
                    g["first_timestamp"],
                    g["last_timestamp"],
                    g["num_gaps"],
                    g["largest_gap_hours"],
                )
            )
        else:
            print("  {:<4} NO DATA ({})".format(label, g["note"]))
    print("cross-source daily close check:")
    print("  days_compared          = {}".format(cross.get("days_compared")))
    print("  max_relative_close_diff= {}".format(cross.get("max_relative_close_diff")))
    print("=" * 60)

    # ---- Exit policy ----------------------------------------------------- #
    # Success requires at least the 1h granularity.
    if results[3600]["rows"]:
        print("OK: 1h data present -> exit 0")
        return 0
    print("FAIL: 1h data missing from all sources -> exit 1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
