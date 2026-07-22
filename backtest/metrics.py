"""Performance metrics computed from the equity curve and trade list.

The Sharpe ratio (primary metric) is computed from UTC-daily-resampled equity
returns, annualised by sqrt(365).
"""

import math

SQRT_365 = math.sqrt(365.0)


def daily_closes(equity):
    """Resample an [(ts, equity), ...] series to UTC daily closes.

    Returns a list of equity values, one per calendar date, taking the last
    bar's equity within each date (bars are assumed ascending).
    """
    out = []
    cur_date = None
    cur_val = None
    for ts, val in equity:
        d = ts.date()
        if cur_date is None:
            cur_date = d
            cur_val = val
        elif d != cur_date:
            out.append(cur_val)
            cur_date = d
            cur_val = val
        else:
            cur_val = val
    if cur_val is not None:
        out.append(cur_val)
    return out


def daily_returns(daily_eq):
    """Simple returns from a list of daily equity levels."""
    out = []
    for i in range(1, len(daily_eq)):
        prev = daily_eq[i - 1]
        if prev != 0:
            out.append(daily_eq[i] / prev - 1.0)
        else:
            out.append(0.0)
    return out


def sharpe_from_daily(daily_eq):
    """Annualised Sharpe from a list of daily equity levels (rf=0)."""
    rets = daily_returns(daily_eq)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)  # ddof=1
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mean / sd * SQRT_365


def max_drawdown_pct(equity):
    """Maximum drawdown (%) over the hourly equity series, as a magnitude."""
    peak = None
    max_dd = 0.0
    for _ts, val in equity:
        if peak is None or val > peak:
            peak = val
        if peak > 0:
            dd = val / peak - 1.0
            if dd < max_dd:
                max_dd = dd
    return abs(max_dd) * 100.0


def compute_metrics(equity, trades, exposure):
    """Compute the full metric dict from the eval-window equity and trades.

    ``equity`` is the [(ts, equity), ...] series restricted to the evaluation
    window. ``exposure`` is the fraction of eval bars holding a position.
    """
    result = {
        "trades": len(trades),
        "net_return_pct": None,
        "sharpe": 0.0,
        "max_dd_pct": 0.0,
        "win_rate": None,
        "avg_r": None,
        "profit_factor": None,
        "exposure": exposure,
        "cagr_pct": None,
        "avg_bars_held": None,
    }

    if equity:
        eq_start = equity[0][1]
        eq_end = equity[-1][1]
        if eq_start != 0:
            result["net_return_pct"] = (eq_end / eq_start - 1.0) * 100.0
        d_eq = daily_closes(equity)
        result["sharpe"] = sharpe_from_daily(d_eq)
        result["max_dd_pct"] = max_drawdown_pct(equity)

        days = (equity[-1][0] - equity[0][0]).total_seconds() / 86400.0
        if days > 0 and eq_start > 0 and eq_end > 0:
            result["cagr_pct"] = ((eq_end / eq_start) ** (365.0 / days) - 1.0) * 100.0

    if trades:
        n = len(trades)
        pnls = [t["pnl"] for t in trades]
        rs = [t["r_multiple"] for t in trades]
        wins = [p for p in pnls if p > 0]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = -sum(p for p in pnls if p < 0)
        result["win_rate"] = len(wins) / n * 100.0
        result["avg_r"] = sum(rs) / n
        result["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else None
        result["avg_bars_held"] = sum(t["bars_held"] for t in trades) / n

    return result
