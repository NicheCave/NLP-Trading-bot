# executor.py — risk-managed order placement for Bybit perpetuals (v7).
# ─────────────────────────────────────────────────────────────
# Rewritten against the Broker interface. What changed from the Alpaca
# version, and why each mattered:
#
#   Shorts are real          SELL used to mean "close the long, if any";
#                            half the indicator stack voted into the void.
#   Sizing is risk-based     see risk.py — the old rule risked 1/20th of
#                            what it claimed.
#   Stops live on the venue  attached to the entry order, so a crashed bot
#                            or a gapped candle can't leave a leveraged
#                            position unprotected.
#   Price-based exit maths   Bybit's unrealised % is a percentage of MARGIN;
#                            comparing it to a price-based target fires the
#                            exit leverage-times too early.
#   Symmetric trailing       the old trailing stop only tracked highs.
#
# Local polling is now a SAFETY NET for the things the exchange can't know
# about — time stops, funding-driven exits, and trailing amendments — not
# the primary protection mechanism.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from loguru import logger

import config
import risk
from brokers.base import Broker, BrokerError, BrokerPosition

STATE_FILE = "logs/executor_state.json"


@dataclass
class TradeState:
    """What we know about an open position that the exchange doesn't."""
    symbol:      str
    side:        str
    entry_price: float
    entry_time:  str                    # ISO8601 UTC
    stop_price:  float                  # current stop — moves as it trails
    take_price:  Optional[float] = None
    # The stop as placed at entry. R-multiples must be measured against THIS,
    # never the trailed stop: once a winner trails past entry, entry − stop
    # is locked-in profit, not risk, and dividing by it understates the
    # trade — a full 3.9 R take-profit was once logged as 2.5 R this way.
    initial_stop: Optional[float] = None
    atr:         float = 0.0
    confidence:  float = 0.0
    extreme:     float = 0.0            # best price seen since entry
    reason:      str = ''
    tags:        list[str] = field(default_factory=list)

    @property
    def entered_at(self) -> datetime:
        return datetime.fromisoformat(self.entry_time)

    def hours_held(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.entered_at).total_seconds() / 3600


class Executor:
    """Turns decisions into orders, and manages what happens afterwards."""

    def __init__(self, broker: Broker, on_close=None):
        self.broker = broker
        # Called as on_close(TradeState, exit_price, pnl_pct, outcome) whenever
        # a position ends — including exchange-side stops, which are the normal
        # exit path now that protection lives at the venue.
        self.on_close = on_close
        self.states: dict[str, TradeState] = {}
        self.daily_trades = 0
        self._today = date.today()
        # Epoch-ish default so the first management cycle always polls.
        self._last_poll = datetime.fromtimestamp(0, tz=timezone.utc)

        # Loss-streak cooldown, carried over from v5 — it earned its place.
        self.recent_outcomes: dict[str, list[str]] = {}
        self.cooldown_until: dict[str, datetime] = {}

        self._load()

    # ══════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════

    def open_position(self,
                      symbol: str,
                      direction: str,
                      stop_price: float,
                      take_price: Optional[float] = None,
                      atr: float = 0.0,
                      confidence: float = 0.0,
                      reason: str = '',
                      tags: Optional[list[str]] = None) -> bool:
        """Size, gate and place an entry with exchange-resident protection."""
        self._roll_day()

        if self.daily_trades >= config.MAX_DAILY_TRADES:
            logger.warning(f"Daily trade limit reached — skipping {symbol}")
            return False

        if self.is_in_cooldown(symbol):
            until = self.cooldown_until.get(symbol)
            logger.warning(f"🧊 {symbol} in cooldown until "
                           f"{until.strftime('%H:%M UTC') if until else '?'} — skipping")
            return False

        price = self.broker.get_price(symbol)
        if price is None:
            logger.warning(f"No price for {symbol} — skipping entry")
            return False

        # Sanity: the stop must be on the losing side of the entry.
        if (direction == 'long' and stop_price >= price) or \
           (direction == 'short' and stop_price <= price):
            logger.error(f"{symbol}: stop {stop_price} is on the wrong side of "
                         f"{price} for a {direction} — refusing")
            return False

        # If we can't confirm what we already hold, we must not open anything —
        # the duplicate-entry guard depends on this list being truthful.
        try:
            positions = self.broker.get_positions()
        except BrokerError as e:
            logger.warning(f"Entry for {symbol} aborted — cannot confirm open "
                           f"positions ({e})")
            return False

        decision = risk.evaluate_entry(
            symbol=symbol, side=direction, entry=price, stop=stop_price,
            equity=self.broker.get_equity(), positions=positions,
            instrument=self.broker.get_instrument(symbol),
            available_margin=self.broker.get_available_margin(),
            confidence=confidence,
        )
        if not decision.allowed:
            logger.info(f"Entry blocked for {symbol}: {decision.reason}")
            return False

        sizing = decision.sizing
        result = self.broker.open_position(
            symbol=symbol, side=direction, qty=sizing.qty,
            stop_loss=stop_price, take_profit=take_price,
        )
        if not result.ok:
            logger.error(f"Entry failed for {symbol}: {result.error}")
            return False

        fill = result.filled_price or price
        self.states[symbol] = TradeState(
            symbol=symbol, side=direction, entry_price=fill,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_price=stop_price, take_price=take_price,
            initial_stop=stop_price,
            atr=atr, confidence=confidence, extreme=fill,
            reason=reason, tags=list(tags or []),
        )
        self.daily_trades += 1
        self._save()

        logger.success(
            f"📈 {direction.upper()} {sizing.qty} {symbol} @ ~{fill:.4f} | "
            f"risk={sizing.risk_amount:.2f} ({sizing.risk_amount / max(1e-9, self.broker.get_equity()):.2%}) "
            f"| SL={stop_price:.4f} TP={take_price:.4f} | {reason[:70]}"
        )
        return True

    # ══════════════════════════════════════════════════════════
    # EXIT
    # ══════════════════════════════════════════════════════════

    def close_position(self, symbol: str, reason: str = 'manual') -> bool:
        result = self.broker.close_position(symbol)
        if not result.ok:
            logger.error(f"Close failed for {symbol}: {result.error}")
            return False
        logger.info(f"Closed {symbol} — {reason}")
        self.states.pop(symbol, None)
        self._save()
        return True

    def manage_positions(self, funding_rates: Optional[dict[str, float]] = None) -> None:
        """
        Per-cycle maintenance for things the exchange cannot decide.

        The exchange already owns stop-loss and take-profit. This handles
        trailing amendments, the time stop, and funding-driven exits.

        If the venue can't be reached, the cycle is SKIPPED entirely rather
        than acted on. Reconciling against an unanswered request would read
        as "everything closed" and destroy the tracking state for positions
        that are still open.
        """
        # With nothing open there is nothing to manage, so back the polling
        # right off. The tuned gates keep the bot flat most of the time, and
        # hammering a flaky demo endpoint every EXIT_INTERVAL seconds for an
        # answer we already know just manufactures connection errors.
        if not self.states:
            elapsed = (datetime.now(timezone.utc) - self._last_poll).total_seconds()
            if elapsed < config.FLAT_POLL_INTERVAL:
                return

        try:
            positions = self.broker.get_positions()
            self._last_poll = datetime.now(timezone.utc)
        except BrokerError as e:
            logger.warning(f"Skipping position management — {e}. "
                           f"Exchange-side stops remain in force.")
            return

        self._reconcile(positions)

        for symbol, position in positions.items():
            state = self.states.get(symbol)
            if state is None:
                continue

            price = position.mark_price or self.broker.get_price(symbol)
            if not price:
                continue

            # ── track the best price seen, for the trail ──────
            if position.side == 'long':
                state.extreme = max(state.extreme or price, price)
            else:
                state.extreme = min(state.extreme or price, price)

            # ── time stop ─────────────────────────────────────
            held = state.hours_held()
            if config.TIME_STOP_HOURS and held >= config.TIME_STOP_HOURS:
                logger.warning(f"⏱ Time stop for {symbol} — held {held:.1f}h "
                               f"(P&L {position.pnl_pct_of_price():+.2%})")
                self.close_position(symbol, 'time_stop')
                continue

            # ── funding-driven exit ───────────────────────────
            rate = (funding_rates or {}).get(symbol)
            if rate is not None and risk.funding_argues_for_exit(position.side, rate):
                logger.warning(
                    f"💸 Funding exit for {symbol} — rate {rate:+.4%}/8h against a "
                    f"{position.side} (≈{rate * 3 * 365:+.0%}/yr carry)")
                self.close_position(symbol, 'funding')
                continue

            # ── trailing stop ─────────────────────────────────
            new_stop = risk.trailing_stop_price(position, state.extreme, state.atr)
            if new_stop is not None:
                # Quantise BEFORE deciding whether to send. The exchange
                # stores prices rounded to the symbol's tick, so a candidate
                # fractionally above the current stop that rounds to the SAME
                # tick is a no-op. Bybit rejects those with 34040 "not
                # modified" — once per management cycle, indefinitely, while
                # the real stop sits correctly in place the whole time.
                instrument = self.broker.get_instrument(symbol)
                quantised = instrument.round_price(new_stop)
                current = (instrument.round_price(position.stop_loss)
                           if position.stop_loss is not None else None)

                if current is None or quantised != current:
                    result = self.broker.set_stops(symbol, stop_loss=quantised)
                    if result.ok:
                        logger.info(f"📉 Trail {symbol}: stop {current} → "
                                    f"{quantised} (extreme {state.extreme:.4f})")
                        state.stop_price = quantised

        self._save()

    def _reconcile(self, positions: dict[str, BrokerPosition]) -> None:
        """
        Detect positions the exchange closed behind our back.

        With stops living at the venue this is the NORMAL exit path, not an
        edge case: the position simply disappears between polls. Anything we
        still track but the exchange doesn't must be released.
        """
        for symbol in list(self.states):
            if symbol in positions:
                continue

            state = self.states.pop(symbol)

            # The real fill, not the price at the moment we happened to look.
            exit_info = None
            try:
                exit_info = self.broker.get_last_exit(symbol, state.entered_at)
            except Exception as e:
                logger.debug(f"get_last_exit({symbol}) failed: {e}")
            if exit_info:
                price, closed_at = exit_info
            else:
                price = self.broker.get_price(symbol) or state.entry_price
                closed_at = datetime.now(timezone.utc)
                logger.warning(f"{symbol}: actual exit unavailable — recording "
                               f"current price {price}, which may differ from the fill")

            move = (price - state.entry_price) / state.entry_price
            pnl_pct = move if state.side == 'long' else -move
            outcome = 'win' if pnl_pct > 0 else 'loss'

            logger.info(
                f"{'✅' if outcome == 'win' else '❌'} {symbol} closed at the exchange "
                f"— {pnl_pct:+.2%} ({outcome})")
            self.register_outcome(symbol, outcome)

            if self.on_close is not None:
                try:
                    self.on_close(state, price, pnl_pct, outcome, closed_at=closed_at)
                except Exception as e:
                    logger.warning(f"on_close hook failed for {symbol}: {e}")

        self._save()

    # ══════════════════════════════════════════════════════════
    # LOSS-STREAK COOLDOWN
    # ══════════════════════════════════════════════════════════

    def register_outcome(self, symbol: str, outcome: str) -> None:
        """Track consecutive losses and freeze the symbol after a streak."""
        history = self.recent_outcomes.setdefault(symbol, [])
        history.insert(0, outcome)
        del history[10:]

        if outcome == 'win':
            if self.cooldown_until.pop(symbol, None):
                logger.info(f"❄ Cooldown cleared for {symbol} — a win resets the streak")
            self._save()
            return

        streak = 0
        for entry in history:
            if entry != 'loss':
                break
            streak += 1

        if streak >= config.LOSS_STREAK_THRESHOLD:
            until = datetime.now(timezone.utc) + timedelta(hours=config.LOSS_STREAK_COOLDOWN_HRS)
            self.cooldown_until[symbol] = until
            logger.warning(f"🧊 Cooldown for {symbol}: {streak} consecutive losses "
                           f"→ frozen until {until.strftime('%Y-%m-%d %H:%M UTC')}")
        self._save()

    def is_in_cooldown(self, symbol: str) -> bool:
        until = self.cooldown_until.get(symbol)
        if until is None:
            return False
        if datetime.now(timezone.utc) >= until:
            self.cooldown_until.pop(symbol, None)
            self._save()
            return False
        return True

    # ══════════════════════════════════════════════════════════
    # STATE
    # ══════════════════════════════════════════════════════════

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._today:
            self._today = today
            self.daily_trades = 0
            logger.info("New day — daily trade counter reset")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump({
                    'states': {s: asdict(v) for s, v in self.states.items()},
                    'recent_outcomes': self.recent_outcomes,
                    'cooldown_until': {s: t.isoformat() for s, t in self.cooldown_until.items()},
                    'daily_trades': self.daily_trades,
                    'today': self._today.isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save executor state: {e}")

    def _load(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.states = {s: TradeState(**v) for s, v in data.get('states', {}).items()}
            # State files written before initial_stop existed: the current stop
            # is the best available proxy, and is exact for any position that
            # has not yet trailed.
            for state in self.states.values():
                if state.initial_stop is None:
                    state.initial_stop = state.stop_price
            self.recent_outcomes = data.get('recent_outcomes', {})
            self.cooldown_until = {
                s: datetime.fromisoformat(t)
                for s, t in data.get('cooldown_until', {}).items()
            }
            saved_day = data.get('today')
            if saved_day == date.today().isoformat():
                self.daily_trades = data.get('daily_trades', 0)
            logger.info(f"Restored {len(self.states)} tracked positions, "
                        f"{len(self.cooldown_until)} cooldowns")
        except Exception as e:
            logger.warning(f"Could not load executor state: {e}")
