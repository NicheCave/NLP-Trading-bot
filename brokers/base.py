# brokers/base.py — the interface every execution venue implements.
# ─────────────────────────────────────────────────────────────
# One interface, three implementations: Bybit Demo, Bybit live, and a local
# simulator. The strategy and risk layers only ever see this ABC, which is
# what lets the exact same code be backtested, paper-traded and (if ever
# wanted) traded live without a rewrite.
#
# Design note — the interface is deliberately POSITION-oriented, not
# order-oriented. On a one-way perpetual account the position is the real
# unit of state; orders are just the mechanism that changes it. Modelling
# orders here would leak Bybit's specifics into every caller.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from data import Instrument


class BrokerError(RuntimeError):
    """
    The venue could not be reached, or refused to answer.

    This exists to keep "the exchange says you hold nothing" strictly
    separate from "the exchange did not answer". Collapsing the two into an
    empty result is dangerous: reconciliation would read a dropped request
    as every position having been closed, wipe its own tracking state, and
    then re-enter symbols it still holds.

    Callers should FAIL CLOSED on this — skip the cycle, place no orders.
    """


@dataclass
class BrokerPosition:
    """A live position, normalised across venues."""
    symbol:         str
    side:           str              # 'long' | 'short'
    qty:            float
    entry_price:    float
    mark_price:     float
    unrealised_pnl: float            # in quote currency, NOT a percentage
    leverage:       float = 1.0
    stop_loss:      Optional[float] = None
    take_profit:    Optional[float] = None
    created_at:     Optional[datetime] = None

    @property
    def notional(self) -> float:
        return self.qty * self.mark_price

    def pnl_pct_of_price(self) -> float:
        """
        Move in the underlying, as a fraction of entry price.

        Deliberately NOT the exchange's unrealised-PnL percentage: on a
        leveraged position that figure is a percentage of MARGIN, so a 3×
        position showing "+6%" has only moved 2% in price. Sizing a target
        against the wrong one fires it 3× too early.
        """
        if not self.entry_price:
            return 0.0
        move = (self.mark_price - self.entry_price) / self.entry_price
        return move if self.side == 'long' else -move


@dataclass
class OrderResult:
    ok:       bool
    order_id: Optional[str] = None
    error:    Optional[str] = None
    filled_price: Optional[float] = None

    def __bool__(self) -> bool:
        return self.ok


class Broker(ABC):
    """Execution venue interface."""

    # ── account ───────────────────────────────────────────────
    @abstractmethod
    def get_equity(self) -> float:
        """Total account equity in quote currency, including open PnL."""

    @abstractmethod
    def get_available_margin(self) -> float:
        """Margin free for new positions."""

    # ── positions ─────────────────────────────────────────────
    @abstractmethod
    def get_positions(self) -> dict[str, BrokerPosition]:
        """
        All open positions, keyed by symbol. Empty dict when genuinely flat.

        MUST raise BrokerError rather than return {} if the venue could not
        be reached — an empty dict means "confirmed flat", and callers act
        on that difference.
        """

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return self.get_positions().get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.get_positions()

    # ── trading ───────────────────────────────────────────────
    @abstractmethod
    def open_position(self,
                      symbol: str,
                      side: str,
                      qty: float,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> OrderResult:
        """
        Open a position, attaching protection AT THE EXCHANGE.

        stop_loss/take_profit must be submitted with the entry rather than
        managed by a local polling loop: if the bot dies, the network drops,
        or a candle gaps, exchange-resident stops still fire. A leveraged
        position guarded only by a 30-second poll is unprotected.
        """

    @abstractmethod
    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        """Close all (or part) of a position with a reduce-only market order."""

    @abstractmethod
    def set_stops(self,
                  symbol: str,
                  stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None) -> OrderResult:
        """Amend protection on an open position (used by the trailing stop)."""

    # ── venue metadata ────────────────────────────────────────
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for a symbol. True if set or already correct."""

    @abstractmethod
    def get_instrument(self, symbol: str) -> Instrument:
        """Lot/tick rules, for quantising orders."""

    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Latest traded price."""

    def get_mark_price(self, symbol: str) -> Optional[float]:
        """Mark price; defaults to last price when a venue has no separate mark."""
        return self.get_price(symbol)

    def get_last_exit(self, symbol: str,
                      since: datetime) -> Optional[tuple[float, datetime]]:
        """
        Actual fill price and time of the most recent close since `since`.

        Exchange-side stops close positions while the bot isn't looking, and
        the bot may only notice hours later (e.g. after the machine slept).
        Recording the market price at the moment of NOTICING rather than the
        real fill misstates every such trade — and it is the winners that
        suffer, since those are the ones that trail and close unattended.

        Returns None when the venue can't say; callers then fall back.
        """
        return None

    # ── lifecycle ─────────────────────────────────────────────
    def prepare(self, symbols: list[str]) -> None:
        """One-off setup at boot (leverage, margin mode). Optional."""

    @property
    def name(self) -> str:
        return type(self).__name__
