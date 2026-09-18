# edge_tracker.py — which ideas actually pay? (Steenbarger edge log)
# ─────────────────────────────────────────────────────────────
# Carried over from v5 because the concept was genuinely good: record the
# feature tags present on every trade, then report win rate per tag so the
# stack can be judged by evidence rather than by how sophisticated it feels.
#
# Two corrections to the original:
#
#   1. A "win" is now NET of fees and funding. The old version counted any
#      pnl_pct > 0 as a win, which on Bybit means a trade can be logged as a
#      win while having lost money: a round trip costs ~0.11% in taker fees
#      plus up to six funding settlements on a 48h hold. Over 815 baseline
#      trades that misclassification is worth several percentage points of
#      reported win rate — flattering exactly the marginal trades that
#      should be pruned.
#
#   2. Win rate is reported ALONGSIDE its break-even line. A tag winning 65%
#      of the time at 0.4R per win is losing money; one winning 40% at 3R is
#      not. Either number alone invites the wrong conclusion.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

import config

EDGE_LOG_PATH = "logs/edge_log.json"


@dataclass
class ClosedTrade:
    symbol:      str
    side:        str
    entry_price: float
    exit_price:  float
    entry_time:  str
    exit_time:   str
    pnl_pct:     float          # gross move, as a fraction of entry
    net_pnl_pct: float          # after estimated fees and funding
    r_multiple:  float
    confidence:  float
    outcome:     str            # 'win' | 'loss' — decided on NET
    tags:        list[str] = field(default_factory=list)
    reason:      str = ''


class EdgeTracker:
    """Per-tag performance attribution, persisted across restarts."""

    def __init__(self, path: str = EDGE_LOG_PATH):
        self.path = path
        self.closed: list[ClosedTrade] = []
        self._load()

    # ── recording ─────────────────────────────────────────────
    def record_close(self, state, exit_price: float, pnl_pct: float,
                     outcome: str, closed_at: Optional[datetime] = None) -> None:
        """
        Log a completed trade.

        `outcome` from the executor is gross; it is recomputed here on a net
        basis, which is the number that decides win/loss.
        """
        # Round-trip taker fees, plus funding for however long it was held.
        fee_cost = config.TAKER_FEE * 2
        closed_at = closed_at or datetime.now(timezone.utc)
        settlements = max(0, int(state.hours_held(closed_at) // 8))
        funding_cost = settlements * 0.0001      # ~1bp per settlement, typical
        net_pnl_pct = pnl_pct - fee_cost - funding_cost

        initial_stop = getattr(state, 'initial_stop', None) or state.stop_price
        risk_pct = abs(state.entry_price - initial_stop) / state.entry_price
        r_multiple = net_pnl_pct / risk_pct if risk_pct > 0 else 0.0

        trade = ClosedTrade(
            symbol=state.symbol, side=state.side,
            entry_price=state.entry_price, exit_price=exit_price,
            entry_time=state.entry_time,
            exit_time=closed_at.isoformat(),
            pnl_pct=pnl_pct, net_pnl_pct=net_pnl_pct, r_multiple=r_multiple,
            confidence=state.confidence,
            outcome='win' if net_pnl_pct > 0 else 'loss',
            tags=list(state.tags), reason=state.reason[:200],
        )
        self.closed.append(trade)
        self._save()

        if trade.outcome != outcome:
            logger.info(f"  ↳ {state.symbol} was a gross {outcome} but a net "
                        f"{trade.outcome} once costs are counted "
                        f"({pnl_pct:+.2%} → {net_pnl_pct:+.2%})")

    # ── reporting ─────────────────────────────────────────────
    def report(self) -> str:
        if len(self.closed) < 5:
            return f"EdgeTracker: {len(self.closed)} closed trades — need 5 for a report"

        total = len(self.closed)
        wins = [t for t in self.closed if t.outcome == 'win']
        losses = [t for t in self.closed if t.outcome == 'loss']
        win_rate = len(wins) / total * 100

        avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0
        breakeven = (abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
                     if (avg_win + abs(avg_loss)) > 0 else 0.0)
        expectancy = sum(t.r_multiple for t in self.closed) / total

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            f"║  EDGE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            f"{'':>29}║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Trades {total:<6}  win rate {win_rate:5.1f}%   "
            f"break-even {breakeven:5.1f}%      ║",
            f"║  Expectancy {expectancy:+.3f}R    avg win {avg_win:+.2f}R / "
            f"avg loss {avg_loss:+.2f}R  ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║  Per tag (net of costs)                                      ║",
        ]

        stats: dict[str, list] = {}
        for trade in self.closed:
            for tag in trade.tags:
                bucket = stats.setdefault(tag, [0, 0, 0.0])
                bucket[1] += 1
                bucket[2] += trade.r_multiple
                if trade.outcome == 'win':
                    bucket[0] += 1

        ranked = sorted(stats.items(), key=lambda kv: kv[1][2] / kv[1][1], reverse=True)
        for tag, (tag_wins, tag_total, total_r) in ranked:
            if tag_total < 3:
                continue     # too few samples to read anything into
            rate = tag_wins / tag_total * 100
            edge = total_r / tag_total
            bar = '█' * int(rate / 10) + '░' * (10 - int(rate / 10))
            lines.append(f"║   {tag:<22} {bar} {rate:5.1f}% {edge:+.2f}R "
                         f"({tag_total:3d})  ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────
    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump({'closed': [asdict(t) for t in self.closed[-1000:]]}, f, indent=2)
        except Exception as e:
            logger.warning(f"EdgeTracker: could not save: {e}")

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.closed = [ClosedTrade(**t) for t in data.get('closed', [])]
            logger.info(f"EdgeTracker: loaded {len(self.closed)} historical trades")
        except Exception as e:
            logger.warning(f"EdgeTracker: could not load history: {e}")
