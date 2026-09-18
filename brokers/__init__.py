# brokers — execution venues behind one interface.
from __future__ import annotations

from loguru import logger

import config

from .base import Broker, BrokerPosition, OrderResult
from .sim import SimBroker

__all__ = ['Broker', 'BrokerPosition', 'OrderResult', 'SimBroker',
           'BybitFuturesBroker', 'make_broker']


def __getattr__(name: str):
    # Imported lazily so the backtester and the simulator never require
    # pybit to be installed.
    if name == 'BybitFuturesBroker':
        from .bybit import BybitFuturesBroker
        return BybitFuturesBroker
    raise AttributeError(name)


def make_broker(mode: str = 'demo', **kwargs) -> Broker:
    """
    Build the broker for a run mode.

        'demo'  → Bybit Demo Trading  (real prices, fake money, needs keys)
        'live'  → Bybit mainnet       (real money — requires explicit opt-in)
        'paper' → local simulator     (real prices, no account needed)
    """
    mode = mode.lower()

    if mode == 'paper':
        return SimBroker(initial_equity=kwargs.pop('initial_equity',
                                                   config.BACKTEST_INITIAL_EQUITY),
                         **kwargs)

    if mode in ('demo', 'live'):
        from .bybit import BybitFuturesBroker
        if mode == 'live':
            logger.warning("⚠ LIVE mode — orders will use real funds")
        return BybitFuturesBroker(demo=(mode == 'demo'), **kwargs)

    raise ValueError(f"Unknown broker mode {mode!r}; expected paper, demo or live")
