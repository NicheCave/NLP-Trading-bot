# brokers/sim.py — local paper-trading simulator.
# ─────────────────────────────────────────────────────────────
# Implements the same Broker interface as the Bybit adapter, but fills
# orders internally against real public mainnet prices. Two uses:
#
#   1. Paper trading with no Bybit account at all.
#   2. A deterministic target for unit tests — fills can be driven by an
#      injected price function instead of the network.
#
# Unlike the exchange, this simulator must enforce stops itself, so
# mark_to_market() has to be called regularly by whatever loop owns it.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from loguru import logger

import config
from data import Instrument, get_instrument, get_last_price

from .base import Broker, BrokerPosition, OrderResult

STATE_FILE = "logs/sim_broker_state.json"


class SimBroker(Broker):
    """Simulated perpetual-futures venue."""

    def __init__(self,
                 initial_equity: float = 10_000.0,
                 taker_fee: float = None,
                 slippage_bps: float = None,
                 price_fn: Optional[Callable[[str], Optional[float]]] = None,
                 persist: bool = True):
        self.cash = initial_equity
        self.initial_equity = initial_equity
        self.taker_fee = config.TAKER_FEE if taker_fee is None else taker_fee
        self.slippage_bps = config.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        self._price_fn = price_fn or get_last_price
        self._positions: dict[str, BrokerPosition] = {}
        self.closed_trades: list[dict] = []
        self.persist = persist

        if persist:
            self._load()

    # ── pricing ───────────────────────────────────────────────
    def get_price(self, symbol: str) -> Optional[float]:
        return self._price_fn(symbol)

    def _fill_price(self, symbol: str, side: str, entering: bool) -> Optional[float]:
        """Apply slippage in whichever direction hurts."""
        price = self.get_price(symbol)
        if price is None:
            return None
        bps = self.slippage_bps / 10_000.0
        adverse_up = (side == 'long') == entering
        return price * (1 + bps) if adverse_up else price * (1 - bps)

    # ── account ───────────────────────────────────────────────
    def get_equity(self) -> float:
        equity = self.cash
        for position in self._positions.values():
            price = self.get_price(position.symbol)
            if price is not None:
                move = price - position.entry_price
                equity += move * position.qty if position.side == 'long' else -move * position.qty
        return equity

    def get_available_margin(self) -> float:
        used = sum(p.qty * p.entry_price / config.LEVERAGE for p in self._positions.values())
        return max(0.0, self.get_equity() - used)

    # ── positions ─────────────────────────────────────────────
    def get_positions(self) -> dict[str, BrokerPosition]:
        for position in self._positions.values():
            price = self.get_price(position.symbol)
            if price is not None:
                position.mark_price = price
                move = price - position.entry_price
                position.unrealised_pnl = (move * position.qty if position.side == 'long'
                                           else -move * position.qty)
        return dict(self._positions)

    # ── trading ───────────────────────────────────────────────
    def open_position(self,
                      symbol: str,
                      side: str,
                      qty: float,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> OrderResult:
        if symbol in self._positions:
            return OrderResult(False, error=f"already in position for {symbol}")

        instrument = self.get_instrument(symbol)
        qty = instrument.round_qty(qty)
        if qty <= 0:
            return OrderResult(False, error=f"qty below {symbol} min lot")

        entry = self._fill_price(symbol, side, entering=True)
        if entry is None:
            return OrderResult(False, error=f"no price for {symbol}")
        if not instrument.is_tradeable(qty, entry):
            return OrderResult(False, error="notional below exchange minimum")

        fee = qty * entry * self.taker_fee
        self.cash -= fee

        self._positions[symbol] = BrokerPosition(
            symbol=symbol, side=side, qty=qty,
            entry_price=entry, mark_price=entry, unrealised_pnl=0.0,
            leverage=config.LEVERAGE,
            stop_loss=stop_loss, take_profit=take_profit,
            created_at=datetime.now(timezone.utc),
        )
        self._save()
        logger.success(
            f"[SIM] OPEN {side.upper()} {qty} {symbol} @ {entry:.4f} | "
            f"SL={stop_loss} TP={take_profit} | fee={fee:.2f}"
        )
        return OrderResult(True, order_id=f"sim-{symbol}-{len(self.closed_trades)}",
                           filled_price=entry)

    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        position = self._positions.get(symbol)
        if position is None:
            return OrderResult(False, error=f"no open position for {symbol}")

        exit_price = self._fill_price(symbol, position.side, entering=False)
        if exit_price is None:
            return OrderResult(False, error=f"no price for {symbol}")

        return self._settle(position, exit_price, 'manual')

    def _settle(self, position: BrokerPosition, exit_price: float, reason: str) -> OrderResult:
        """Book a close at a given price and release the position."""
        move = exit_price - position.entry_price
        gross = move * position.qty if position.side == 'long' else -move * position.qty
        fee = position.qty * exit_price * self.taker_fee
        self.cash += gross - fee

        self.closed_trades.append({
            'symbol': position.symbol, 'side': position.side, 'qty': position.qty,
            'entry_price': position.entry_price, 'exit_price': exit_price,
            'gross_pnl': gross, 'fee': fee, 'net_pnl': gross - fee,
            'reason': reason,
            'opened_at': position.created_at.isoformat() if position.created_at else None,
            'closed_at': datetime.now(timezone.utc).isoformat(),
        })
        self._positions.pop(position.symbol, None)
        self._save()

        logger.success(
            f"[SIM] CLOSE {position.side} {position.qty} {position.symbol} "
            f"@ {exit_price:.4f} | {reason} | net={gross - fee:+.2f}"
        )
        return OrderResult(True, filled_price=exit_price)

    def get_last_exit(self, symbol, since):
        for trade in reversed(self.closed_trades):
            if trade['symbol'] != symbol or not trade.get('closed_at'):
                continue
            closed_at = datetime.fromisoformat(trade['closed_at'])
            if closed_at >= since:
                return float(trade['exit_price']), closed_at
        return None

    def set_stops(self,
                  symbol: str,
                  stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None) -> OrderResult:
        position = self._positions.get(symbol)
        if position is None:
            return OrderResult(False, error=f"no open position for {symbol}")
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        self._save()
        return OrderResult(True)

    def mark_to_market(self) -> list[str]:
        """
        Enforce stops locally.

        The real exchange does this itself once a stop is attached to the
        position; the simulator has to be driven. Returns the symbols closed.
        """
        closed: list[str] = []
        for symbol, position in list(self._positions.items()):
            price = self.get_price(symbol)
            if price is None:
                continue

            hit_stop = position.stop_loss is not None and (
                price <= position.stop_loss if position.side == 'long'
                else price >= position.stop_loss)
            hit_take = position.take_profit is not None and (
                price >= position.take_profit if position.side == 'long'
                else price <= position.take_profit)

            if hit_stop:
                self._settle(position, position.stop_loss, 'stop_loss')
                closed.append(symbol)
            elif hit_take:
                self._settle(position, position.take_profit, 'take_profit')
                closed.append(symbol)
        return closed

    # ── venue metadata ────────────────────────────────────────
    def set_leverage(self, symbol: str, leverage: float) -> bool:
        return True

    def get_instrument(self, symbol: str) -> Instrument:
        return get_instrument(symbol)

    def prepare(self, symbols: list[str]) -> None:
        logger.info(f"[SIM] paper broker ready — equity ${self.get_equity():,.2f}")

    # ── persistence ───────────────────────────────────────────
    def _save(self) -> None:
        if not self.persist:
            return
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump({
                    'cash': self.cash,
                    'initial_equity': self.initial_equity,
                    'positions': [
                        {**vars(p),
                         'created_at': p.created_at.isoformat() if p.created_at else None}
                        for p in self._positions.values()
                    ],
                    'closed_trades': self.closed_trades[-500:],
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"[SIM] could not save state: {e}")

    def _load(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.cash = state.get('cash', self.cash)
            self.initial_equity = state.get('initial_equity', self.initial_equity)
            self.closed_trades = state.get('closed_trades', [])
            for entry in state.get('positions', []):
                created = entry.pop('created_at', None)
                position = BrokerPosition(**entry)
                position.created_at = datetime.fromisoformat(created) if created else None
                self._positions[position.symbol] = position
            logger.info(f"[SIM] restored {len(self._positions)} positions, "
                        f"cash ${self.cash:,.2f}")
        except Exception as e:
            logger.warning(f"[SIM] could not load state: {e}")
