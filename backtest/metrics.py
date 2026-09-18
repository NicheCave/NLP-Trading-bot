# backtest/metrics.py — scoring a backtest run.
# ─────────────────────────────────────────────────────────────
# Win rate is reported prominently because it's the stated goal, but it is
# deliberately shown NEXT TO its own break-even line. A 70% win rate at
# 0.4R average win and 1R average loss loses money; a 38% win rate at 3R
# makes it. The pair is the information — either number alone is noise.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult, Trade

BARS_PER_YEAR = 24 * 365      # 1H bars


@dataclass
class Metrics:
    n_trades:        int = 0
    win_rate:        float = 0.0
    breakeven_wr:    float = 0.0     # win rate this payoff needs to break even
    expectancy_r:    float = 0.0     # mean R per trade — the headline number
    avg_win_r:       float = 0.0
    avg_loss_r:      float = 0.0
    profit_factor:   float = 0.0
    total_return:    float = 0.0
    cagr:            float = 0.0
    sharpe:          float = 0.0
    sortino:         float = 0.0
    max_drawdown:    float = 0.0
    calmar:          float = 0.0
    avg_bars_held:   float = 0.0
    total_fees:      float = 0.0
    total_funding:   float = 0.0
    gross_pnl:       float = 0.0
    net_pnl:         float = 0.0
    # Trading cost expressed in R — the diagnostic that matters most at this
    # frequency. A strategy can have a genuine gross edge and still lose if
    # each round trip costs a large fraction of what it risks.
    cost_r:          float = 0.0
    gross_expectancy_r: float = 0.0
    # Is the edge distinguishable from zero, or just this sample's luck?
    # With R-multiples having std ≈ 1, a few hundred trades can only resolve
    # an edge of roughly ±0.09R. Reporting expectancy without this invites
    # trading a number that is indistinguishable from noise.
    expectancy_se:   float = 0.0
    expectancy_t:    float = 0.0
    best_trade_r:    float = 0.0
    worst_trade_r:   float = 0.0
    avg_mae_r:       float = 0.0
    avg_mfe_r:       float = 0.0
    long_trades:     int = 0
    short_trades:    int = 0
    long_win_rate:   float = 0.0
    short_win_rate:  float = 0.0
    exit_reasons:    dict[str, int] = field(default_factory=dict)
    per_symbol:      dict[str, dict] = field(default_factory=dict)
    skipped:         dict[str, int] = field(default_factory=dict)


def _safe(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _win_rate(trades: list[Trade]) -> float:
    return _safe(sum(1 for t in trades if t.is_win), len(trades)) * 100


def compute_metrics(result: BacktestResult) -> Metrics:
    """Score a completed backtest."""
    trades = result.trades
    m = Metrics(n_trades=len(trades), skipped=dict(result.skipped))
    if not trades:
        return m

    rs = np.array([t.r_multiple for t in trades], dtype=float)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    m.win_rate = _win_rate(trades)
    m.expectancy_r = float(rs.mean())

    if len(rs) > 1:
        std = float(rs.std(ddof=1))
        m.expectancy_se = std / np.sqrt(len(rs)) if std > 0 else 0.0
        m.expectancy_t = m.expectancy_r / m.expectancy_se if m.expectancy_se > 0 else 0.0

    m.avg_win_r = float(np.mean([t.r_multiple for t in wins])) if wins else 0.0
    m.avg_loss_r = float(np.mean([t.r_multiple for t in losses])) if losses else 0.0

    # Break-even win rate implied by this payoff ratio:
    #   wr × avg_win = (1 − wr) × |avg_loss|
    if wins and losses and (m.avg_win_r + abs(m.avg_loss_r)) > 0:
        m.breakeven_wr = abs(m.avg_loss_r) / (m.avg_win_r + abs(m.avg_loss_r)) * 100

    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    m.profit_factor = _safe(gross_win, gross_loss, float('inf') if gross_win else 0.0)

    m.gross_pnl = sum(t.gross_pnl for t in trades)
    m.total_fees = sum(t.fees for t in trades)
    m.total_funding = sum(t.funding for t in trades)
    m.net_pnl = sum(t.net_pnl for t in trades)
    costs_r = [(t.fees + t.funding) / t.risk_amount
               for t in trades if t.risk_amount > 0]
    m.cost_r = float(np.mean(costs_r)) if costs_r else 0.0
    m.gross_expectancy_r = m.expectancy_r + m.cost_r

    m.best_trade_r = float(rs.max())
    m.worst_trade_r = float(rs.min())
    m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))
    m.avg_mae_r = float(np.mean([t.mae for t in trades]))
    m.avg_mfe_r = float(np.mean([t.mfe for t in trades]))

    longs = [t for t in trades if t.side == 'long']
    shorts = [t for t in trades if t.side == 'short']
    m.long_trades, m.short_trades = len(longs), len(shorts)
    m.long_win_rate = _win_rate(longs)
    m.short_win_rate = _win_rate(shorts)

    for t in trades:
        m.exit_reasons[t.exit_reason] = m.exit_reasons.get(t.exit_reason, 0) + 1

    for symbol in sorted({t.symbol for t in trades}):
        sub = [t for t in trades if t.symbol == symbol]
        m.per_symbol[symbol] = {
            'n': len(sub),
            'win_rate': _win_rate(sub),
            'expectancy_r': float(np.mean([t.r_multiple for t in sub])),
            'net_pnl': sum(t.net_pnl for t in sub),
        }

    # ── equity-curve statistics ───────────────────────────────
    curve = result.equity_curve
    if len(curve) > 1:
        start, end = float(curve.iloc[0]), float(curve.iloc[-1])
        m.total_return = (end / start - 1) * 100 if start else 0.0

        years = len(curve) / BARS_PER_YEAR
        if years > 0 and start > 0 and end > 0:
            m.cagr = ((end / start) ** (1 / years) - 1) * 100

        returns = curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            m.sharpe = float(returns.mean() / returns.std() * np.sqrt(BARS_PER_YEAR))
            downside = returns[returns < 0]
            if len(downside) > 1 and downside.std() > 0:
                m.sortino = float(returns.mean() / downside.std() * np.sqrt(BARS_PER_YEAR))

        running_max = curve.cummax()
        drawdown = (curve - running_max) / running_max
        m.max_drawdown = float(drawdown.min()) * 100
        m.calmar = _safe(m.cagr, abs(m.max_drawdown))

    return m


def format_report(m: Metrics, title: str = "BACKTEST") -> str:
    """Render metrics as a readable block for the log."""
    w = 68

    def row(label: str, value: str) -> str:
        return f"│ {label:<30}{value:>{w - 34}} │"

    lines = [
        "┌" + "─" * (w - 2) + "┐",
        f"│ {title:<{w - 4}} │",
        "├" + "─" * (w - 2) + "┤",
        row("Trades", f"{m.n_trades}"),
        row("Win rate", f"{m.win_rate:.1f}%"),
        row("  break-even win rate needed", f"{m.breakeven_wr:.1f}%"),
        row("  edge over break-even", f"{m.win_rate - m.breakeven_wr:+.1f} pts"),
        "├" + "─" * (w - 2) + "┤",
        row("Expectancy (net)", f"{m.expectancy_r:+.3f} R/trade"),
        row("  ± standard error", f"{m.expectancy_se:.3f}  (t = {m.expectancy_t:+.2f})"),
        row("  significant?", "yes" if abs(m.expectancy_t) >= 2 else "NO — within noise"),
        row("  gross, before costs", f"{m.gross_expectancy_r:+.3f} R/trade"),
        row("  cost per round trip", f"{m.cost_r:.3f} R"),
        row("Avg win / Avg loss", f"{m.avg_win_r:+.2f}R / {m.avg_loss_r:+.2f}R"),
        row("Profit factor", f"{m.profit_factor:.2f}"),
        "├" + "─" * (w - 2) + "┤",
        row("Total return", f"{m.total_return:+.1f}%"),
        row("CAGR", f"{m.cagr:+.1f}%"),
        row("Max drawdown", f"{m.max_drawdown:.1f}%"),
        row("Sharpe / Sortino", f"{m.sharpe:.2f} / {m.sortino:.2f}"),
        row("Calmar", f"{m.calmar:.2f}"),
        "├" + "─" * (w - 2) + "┤",
        row("Gross PnL (before costs)", f"${m.gross_pnl:,.0f}"),
        row("  fees", f"-${m.total_fees:,.0f}"),
        row("  funding", f"-${m.total_funding:,.0f}"),
        row("Net PnL", f"${m.net_pnl:,.0f}"),
        "├" + "─" * (w - 2) + "┤",
        row("Long / Short trades", f"{m.long_trades} / {m.short_trades}"),
        row("Long / Short win rate", f"{m.long_win_rate:.1f}% / {m.short_win_rate:.1f}%"),
        row("Avg bars held", f"{m.avg_bars_held:.1f}"),
        row("Avg MAE / MFE", f"{m.avg_mae_r:.2f}R / {m.avg_mfe_r:.2f}R"),
    ]

    if m.exit_reasons:
        lines.append("├" + "─" * (w - 2) + "┤")
        lines.append(f"│ {'Exits':<{w - 4}} │")
        for reason, count in sorted(m.exit_reasons.items(), key=lambda kv: -kv[1]):
            pct = count / m.n_trades * 100 if m.n_trades else 0
            lines.append(row(f"  {reason}", f"{count} ({pct:.0f}%)"))

    if m.per_symbol:
        lines.append("├" + "─" * (w - 2) + "┤")
        lines.append(f"│ {'Per symbol':<{w - 4}} │")
        for symbol, s in m.per_symbol.items():
            lines.append(row(f"  {symbol}",
                             f"{s['n']}t {s['win_rate']:.0f}%wr {s['expectancy_r']:+.2f}R"))

    if m.skipped:
        lines.append("├" + "─" * (w - 2) + "┤")
        lines.append(f"│ {'Signals skipped':<{w - 4}} │")
        for why, count in sorted(m.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(row(f"  {why}", f"{count}"))

    lines.append("└" + "─" * (w - 2) + "┘")
    return "\n".join(lines)


def session_breakdown(result: BacktestResult) -> str:
    """
    Performance by UTC hour of entry — the kill-zone hypothesis, measured.

    config.KILL_ZONES asserts that institutional activity concentrates in the
    London/NY session windows and penalises signals outside them. That is FX
    and equities logic applied to a market that never closes, and it was
    never tested. This prints the evidence.

    Read it sceptically: with a few hundred trades spread over 24 buckets,
    most of the variation here is noise. It is only worth acting on if a
    contiguous block of hours stands out, not a single spiky hour.
    """
    trades = result.trades
    if len(trades) < 48:
        return "Session breakdown: too few trades to be meaningful"

    import config

    buckets: dict[int, list] = {}
    for trade in trades:
        buckets.setdefault(trade.entry_ts.hour, []).append(trade)

    lines = [
        "",
        "  PERFORMANCE BY UTC HOUR OF ENTRY",
        "  " + "─" * 60,
        f"  {'hour':<8}{'zone':<10}{'trades':>8}{'win %':>9}{'expect. R':>12}",
        "  " + "─" * 60,
    ]

    def zone_for(hour: int) -> str:
        for label, start, end in config.KILL_ZONES:
            if start <= hour < end:
                return label
        return '—'

    in_zone, out_zone = [], []
    for hour in range(24):
        sub = buckets.get(hour, [])
        if not sub:
            continue
        label = zone_for(hour)
        (in_zone if label != '—' else out_zone).extend(sub)
        expectancy = float(np.mean([t.r_multiple for t in sub]))
        lines.append(f"  {hour:02d}:00   {label:<10}{len(sub):>8}"
                     f"{_win_rate(sub):>9.1f}{expectancy:>+12.3f}")

    lines.append("  " + "─" * 60)
    for label, group in (('inside kill zones', in_zone), ('outside', out_zone)):
        if group:
            expectancy = float(np.mean([t.r_multiple for t in group]))
            lines.append(f"  {label:<26}{len(group):>8}"
                         f"{_win_rate(group):>9.1f}{expectancy:>+12.3f}")

    if in_zone and out_zone:
        delta = (float(np.mean([t.r_multiple for t in in_zone]))
                 - float(np.mean([t.r_multiple for t in out_zone])))
        verdict = ("supports the kill-zone penalty" if delta > 0.05
                   else "does NOT support the kill-zone penalty" if delta < -0.05
                   else "shows no meaningful difference")
        lines.append("")
        lines.append(f"  Difference: {delta:+.3f} R — {verdict}.")
    lines.append("")
    return "\n".join(lines)


def compare(baseline: Metrics, candidate: Metrics,
            baseline_name: str = "baseline",
            candidate_name: str = "candidate") -> str:
    """Side-by-side table — used to judge every change against the baseline."""
    fields = [
        ('Trades',          'n_trades',     '{:.0f}'),
        ('Win rate %',      'win_rate',     '{:.1f}'),
        ('Break-even %',    'breakeven_wr', '{:.1f}'),
        ('Gross expect. R', 'gross_expectancy_r', '{:+.3f}'),
        ('Cost R/trade',    'cost_r',       '{:.3f}'),
        ('Net expect. R',   'expectancy_r', '{:+.3f}'),
        ('Profit factor',   'profit_factor', '{:.2f}'),
        ('Total return %',  'total_return', '{:+.1f}'),
        ('Max DD %',        'max_drawdown', '{:.1f}'),
        ('Sharpe',          'sharpe',       '{:.2f}'),
    ]
    lines = [
        f"{'Metric':<18}{baseline_name:>14}{candidate_name:>14}{'Δ':>12}",
        "─" * 58,
    ]
    for label, attr, fmt in fields:
        b, c = getattr(baseline, attr), getattr(candidate, attr)
        delta = c - b
        lines.append(f"{label:<18}{fmt.format(b):>14}{fmt.format(c):>14}{delta:>+12.2f}")
    return "\n".join(lines)
