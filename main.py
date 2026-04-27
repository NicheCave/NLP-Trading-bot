# main.py — TA + SMC + Advanced TA crypto bot orchestrator (v3)
# ─────────────────────────────────────────────────────────────
# Changes from v2 (Steenbarger edge logging):
#
#   - EdgeTracker: new class that logs win-rate per signal TYPE.
#     Tracks every executed BUY/SELL + its eventual outcome (profit/loss)
#     broken down by which signal "reason tags" were present.
#     Prints an edge summary every EDGE_LOG_INTERVAL minutes showing:
#       - Win rate % per signal category (SMC, AVWAP, Alligator, etc.)
#       - Average confidence vs actual win rate (calibration check)
#       - Trap flip accuracy
#     This lets you see WHICH indicators actually add edge over time.
#
#   - edge_cycle() added to the schedule loop.
#   - analysis_cycle() now passes signal metadata to EdgeTracker.
#   - Startup log updated to v3.
# ─────────────────────────────────────────────────────────────
import schedule
import time
import sys
import json
import os
from loguru import logger
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import config
from signal_generator import generate_signal, Signal
from executor import execute_signal, check_exits, sync_daily_counter, get_open_positions

# ── Logging setup ─────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", rotation="1 day",
           retention="7 days", level="DEBUG")

os.makedirs("logs", exist_ok=True)

# ── Signal cache ──────────────────────────────────────────────
_cached_signals: dict[str, Signal] = {}


# ══════════════════════════════════════════════════════════════
# STEENBARGER EDGE TRACKER
# Tracks win-rate per signal tag to identify which indicators
# genuinely add edge. Persisted to logs/edge_log.json.
# ══════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    ticker: str
    action: str
    confidence: float
    entry_price: float
    signal_tags: list[str]  # which features were active (e.g. 'CHoCH', 'AVWAP', 'trap_flip')
    wave5_caution: bool
    trap_flip: bool
    timestamp: str
    outcome: Optional[str] = None  # 'win' | 'loss' | None (open)
    pnl_pct: Optional[float] = None


class EdgeTracker:
    """
    Steenbarger-style edge log.

    Records every executed trade with its signal tags, then when positions
    close (via check_exits or SELL signal), updates the outcome.
    Periodically prints a summary showing win rate per signal category.

    This answers: "Does my AVWAP breakout signal actually add edge?
    Does the trap flip actually work? Is the Alligator veto saving me
    from bad trades?"
    """

    EDGE_LOG_PATH = "logs/edge_log.json"

    def __init__(self):
        self.open_trades: dict[str, TradeRecord] = {}  # ticker → open record
        self.closed_trades: list[TradeRecord] = []
        self._load()

    def _load(self):
        """Load persisted trade history on startup."""
        if os.path.exists(self.EDGE_LOG_PATH):
            try:
                with open(self.EDGE_LOG_PATH, 'r') as f:
                    data = json.load(f)
                    self.closed_trades = [TradeRecord(**r) for r in data.get('closed', [])]
                    logger.info(f"EdgeTracker: loaded {len(self.closed_trades)} historical trades")
            except Exception as e:
                logger.warning(f"EdgeTracker: could not load history: {e}")

    def _save(self):
        """Persist closed trades to disk."""
        try:
            with open(self.EDGE_LOG_PATH, 'w') as f:
                json.dump({
                    'closed': [r.__dict__ for r in self.closed_trades[-500:]]  # keep last 500
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"EdgeTracker: could not save: {e}")

    @staticmethod
    def _extract_tags(signal: Signal) -> list[str]:
        """Extract meaningful feature tags from a signal's reason string."""
        tags = []
        reason = signal.reason.lower()

        # SMC tags
        if 'choch' in reason: tags.append('CHoCH')
        if 'liquidity sweep' in reason: tags.append('sweep')
        if 'order block' in reason: tags.append('OB')
        if 'fvg' in reason or 'fair value' in reason: tags.append('FVG')
        if 'bos' in reason: tags.append('BOS')

        # v3 advanced tags
        if 'avwap' in reason: tags.append('AVWAP')
        if 'alligator' in reason: tags.append('Alligator')
        if 'ao zero-line' in reason: tags.append('AO_cross')
        if 'fractal' in reason: tags.append('Fractal')
        if 'trap flip' in reason or signal.trap_flip: tags.append('TrapFlip')
        if 'wave 5' in reason or signal.wave5_caution: tags.append('W5_caution')
        if 'wave 3' in reason: tags.append('W3_confirmed')
        if 'smc signals' in reason: tags.append('multi_SMC')
        if 'bear market' in reason: tags.append('bear_ctx')

        # Confidence buckets
        if signal.confidence >= 0.8:
            tags.append('conf_high')
        elif signal.confidence >= 0.5:
            tags.append('conf_mid')
        else:
            tags.append('conf_low')

        return tags

    def record_entry(self, signal: Signal, entry_price: float):
        """Call this when a BUY order is successfully placed."""
        tags = self._extract_tags(signal)
        record = TradeRecord(
            ticker=signal.ticker,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=entry_price,
            signal_tags=tags,
            wave5_caution=signal.wave5_caution,
            trap_flip=signal.trap_flip,
            timestamp=datetime.now().isoformat(),
        )
        self.open_trades[signal.ticker] = record
        logger.debug(f"EdgeTracker: recorded entry {signal.ticker} @ ${entry_price:.2f} | tags={tags}")

    def record_exit(self, ticker: str, exit_price: float, reason: str = ''):
        """Call this when a position is closed (take-profit, stop-loss, or SELL signal)."""
        if ticker not in self.open_trades:
            return

        record = self.open_trades.pop(ticker)
        pnl_pct = (exit_price - record.entry_price) / record.entry_price * 100
        if record.action == 'SELL':
            pnl_pct = -pnl_pct  # short trade

        record.outcome = 'win' if pnl_pct > 0 else 'loss'
        record.pnl_pct = round(pnl_pct, 3)

        self.closed_trades.append(record)
        self._save()

        # v5: notify executor so it can update loss-streak / cooldown state
        try:
            from executor import register_trade_outcome
            register_trade_outcome(ticker, record.outcome)
        except Exception as e:
            logger.debug(f"register_trade_outcome failed: {e}")

        emoji = '✅' if record.outcome == 'win' else '❌'
        logger.info(
            f"EdgeTracker: {emoji} {ticker} closed | "
            f"pnl={pnl_pct:+.2f}% | reason={reason} | tags={record.signal_tags}"
        )

    def print_summary(self):
        """
        Print Steenbarger-style edge report: win rate per signal tag.
        Called on EDGE_LOG_INTERVAL schedule.
        """
        if len(self.closed_trades) < 5:
            logger.info("EdgeTracker: fewer than 5 closed trades — no summary yet")
            return

        # Build tag → [win, total] map
        tag_stats: dict[str, list[int]] = {}
        total_wins = 0
        total_trades = len(self.closed_trades)

        for t in self.closed_trades:
            if t.outcome == 'win':
                total_wins += 1
            for tag in t.signal_tags:
                if tag not in tag_stats:
                    tag_stats[tag] = [0, 0]
                tag_stats[tag][1] += 1
                if t.outcome == 'win':
                    tag_stats[tag][0] += 1

        overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        # Average confidence calibration
        conf_wins = [t.confidence for t in self.closed_trades if t.outcome == 'win']
        conf_loss = [t.confidence for t in self.closed_trades if t.outcome == 'loss']
        avg_conf_w = sum(conf_wins) / len(conf_wins) * 100 if conf_wins else 0
        avg_conf_l = sum(conf_loss) / len(conf_loss) * 100 if conf_loss else 0

        lines = [
            f"",
            f"╔══════════════════════════════════════════════════════╗",
            f"║         STEENBARGER EDGE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}      ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║ Overall win rate: {overall_wr:.1f}% ({total_wins}/{total_trades} trades)              ║",
            f"║ Avg confidence on wins:  {avg_conf_w:.1f}%                        ║",
            f"║ Avg confidence on losses:{avg_conf_l:.1f}%                        ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║ Win rate per signal tag:                             ║",
        ]

        # Sort by win rate descending
        sorted_tags = sorted(tag_stats.items(),
                             key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0,
                             reverse=True)
        for tag, (wins, total) in sorted_tags:
            if total < 2:
                continue  # skip tags with too few samples
            wr = wins / total * 100
            bar = '█' * int(wr / 10) + '░' * (10 - int(wr / 10))
            lines.append(f"║  {tag:<18} {bar} {wr:5.1f}% ({wins}/{total})   ║")

        lines.append(f"╚══════════════════════════════════════════════════════╝")
        lines.append("")

        for line in lines:
            logger.info(line)

    def sync_exits_from_positions(self):
        """
        Cross-reference open_trades against current Alpaca positions.
        FIX: Normalizes symbols (strips '/') to avoid mismatch between BTC/USD and BTCUSD.
        """
        if not self.open_trades:
            return
        try:
            positions = get_open_positions()
            # Normalize symbols: 'ETH/USD' becomes 'ETHUSD'
            open_symbols = {p.symbol.replace('/', '') for p in positions}

            for ticker in list(self.open_trades.keys()):
                # Check if the normalized ticker is missing from Alpaca's list
                if ticker.replace('/', '') not in open_symbols:
                    from executor import _get_current_price
                    from config import to_order_symbol
                    order_sym = to_order_symbol(ticker)

                    exit_price = _get_current_price(ticker, order_sym)
                    if exit_price is None:
                        exit_price = self.open_trades[ticker].entry_price

                    logger.info(f"EdgeTracker: External exit detected for {ticker} — sync complete.")
                    self.record_exit(ticker, exit_price, 'auto_exit')

                    from executor import clear_tracking_state
                    clear_tracking_state(ticker)
        except Exception as e:
            logger.warning(f"EdgeTracker sync error: {e}")


# Global EdgeTracker instance
edge_tracker = EdgeTracker()


# ── Patched execute_signal wrapper ───────────────────────────
# Wraps execute_signal to intercept entries and notify EdgeTracker.
def execute_signal_tracked(signal: Signal):
    """
    Wrapper around execute_signal that records entries in the EdgeTracker.

    v5 FIX (CRITICAL): The previous guard blocked ALL BUY signals when a
    position already existed — including pyramid signals. That made the
    Bellafiore scale-in feature dead code. The guard now only blocks NEW
    entries (signal.add_to_winner=False); pyramid signals pass through to
    execute_signal which routes them to scale_in_winner().
    """
    if signal.action == 'HOLD':
        return

    # GUARD: If already in position AND this isn't a pyramid signal, skip.
    # Pyramid signals fall through to execute_signal → scale_in_winner.
    from executor import already_in_position
    if (signal.action == 'BUY'
            and already_in_position(signal.ticker)
            and not signal.add_to_winner):
        return

    # Capture price for the EdgeTracker entry record (only for fresh entries)
    from config import to_order_symbol
    from executor import _get_current_price
    order_sym = to_order_symbol(signal.ticker)
    price = _get_current_price(signal.ticker, order_sym)

    # Place the actual order on Alpaca
    execute_signal(signal)

    # Only record a NEW entry in EdgeTracker when this was a fresh BUY,
    # not a pyramid scale-in (those don't open a new tracked trade).
    if signal.action == 'BUY' and price is not None and not signal.add_to_winner:
        edge_tracker.record_entry(signal, price)


# ══════════════════════════════════════════════════════════════
# SCHEDULE LOOPS
# ══════════════════════════════════════════════════════════════

def analysis_cycle():
    """
    Slow loop: fetch OHLCV bars → run TA + SMC + Advanced TA indicators → cache signals.
    """
    logger.info(f"════ TA+SMC+AdvTA analysis cycle: {datetime.now().strftime('%Y-%m-%d %H:%M')} ════")

    for ticker in config.WATCHLIST:
        try:
            signal = generate_signal(ticker)
            _cached_signals[ticker] = signal
            logger.info(
                f"Signal cached: {ticker} → {signal.action} "
                f"(bull={signal.bull_pct:.0%} bear={signal.bear_pct:.0%} "
                f"net={signal.avg_score:+.3f} conf={signal.confidence:.0%}"
                + (" [W5⚠]" if signal.wave5_caution else "")
                + (" [TRAP↩]" if signal.trap_flip else "")
                + (" [PYRAMID]" if signal.add_to_winner else "")
                + ")"
            )
        except Exception as e:
            logger.error(f"Analysis error for {ticker}: {e}", exc_info=True)

    logger.info("════ Analysis cycle complete ════\n")


def execution_cycle():
    """
    Fast loop: read cached signals → execute if actionable.
    """
    if not _cached_signals:
        logger.debug("No cached signals yet — skipping execution cycle")
        return

    logger.debug(f"── Execution check: {datetime.now().strftime('%H:%M:%S')} ──")

    for ticker, signal in list(_cached_signals.items()):
        try:
            if signal.action != 'HOLD':
                logger.info(f"Acting on cached signal: {ticker} → {signal.action}")

            if signal.action == 'SELL':
                from executor import already_in_position
                execute_signal_tracked(signal)
                if not already_in_position(ticker):
                    _cached_signals[ticker] = Signal(
                        ticker, 'HOLD', 0.0, 0.0, 0.0, 0.0, 0,
                        "Post-sell hold — waiting for next analysis cycle"
                    )
                    logger.info(f"Signal cleared for {ticker} — position closed")
                else:
                    logger.warning(f"SELL for {ticker} may not have filled — keeping signal")
            else:
                execute_signal_tracked(signal)
        except Exception as e:
            logger.error(f"Execution error for {ticker}: {e}", exc_info=True)


def exit_cycle():
    """
    Fastest loop: check open positions for stop-loss / take-profit.
    Also syncs EdgeTracker with current positions.
    """
    try:
        check_exits()
        edge_tracker.sync_exits_from_positions()
    except Exception as e:
        logger.error(f"Exit check error: {e}")


def sync_cycle():
    """Sync the daily trade counter from Alpaca (once per minute)."""
    try:
        sync_daily_counter()
    except Exception as e:
        logger.error(f"Daily sync error: {e}")


def edge_cycle():
    """
    Steenbarger edge report: print win-rate per signal tag.
    Runs every EDGE_LOG_INTERVAL minutes.
    """
    try:
        edge_tracker.print_summary()
    except Exception as e:
        logger.error(f"Edge report error: {e}")


# ── Schedule all loops independently ──────────────────────────
schedule.every(config.CHECK_INTERVAL).minutes.do(analysis_cycle)
schedule.every(config.EXECUTION_INTERVAL).seconds.do(execution_cycle)
schedule.every(config.EXIT_INTERVAL).seconds.do(exit_cycle)
schedule.every(config.DAILY_SYNC_INTERVAL).seconds.do(sync_cycle)
schedule.every(config.EDGE_LOG_INTERVAL).minutes.do(edge_cycle)

logger.info("🤖 Crypto TA + SMC + Advanced TA Bot starting (v5)...")
logger.info(f"   Device:        {config.DEVICE}")
logger.info(f"   Watchlist:     {config.WATCHLIST}")
logger.info(f"   Analysis:      every {config.CHECK_INTERVAL} min  (TA + SMC + AdvTA)")
logger.info(f"   Execution:     every {config.EXECUTION_INTERVAL} s")
logger.info(f"   Exit guard:    every {config.EXIT_INTERVAL} s")
logger.info(f"   Counter sync:  every {config.DAILY_SYNC_INTERVAL} s")
logger.info(f"   Edge report:   every {config.EDGE_LOG_INTERVAL} min")
logger.info(f"   Alpaca:        {config.ALPACA_BASE_URL}")
logger.info(f"   v3 features:   AVWAP pinch | Alligator veto | AO | Fractals")
logger.info(f"                  RVOL filter | Trap flip | Elliott W5 caution")
logger.info(f"                  Bellafiore pyramiding | Steenbarger edge log")
logger.info(f"   v4 features:   Kill zones | MTF align | BTC anchor | Time/Trail stops")
logger.info(f"   v5 features:   Pyramid storm guard | Persisted exits | Vol-confirmed BOS")
logger.info(
    f"                  Confidence floor ({config.CONFIDENCE_FLOOR:.0%}) | Loss-streak cooldown ({config.LOSS_STREAK_THRESHOLD}L → {config.LOSS_STREAK_COOLDOWN_HRS}h)")
logger.info(f"                  Asymmetric R:R ({config.CRYPTO_STOP_LOSS_PCT:.0%}/{config.CRYPTO_TAKE_PROFIT_PCT:.0%})")

# Run immediately to populate cache
analysis_cycle()
exit_cycle()
sync_cycle()
edge_cycle()

while True:
    schedule.run_pending()
    time.sleep(5)