"""
The bot must fail CLOSED when the exchange doesn't answer.

A read timeout against api-demo.bybit.com is routine. The dangerous
response to one is to treat the missing answer as information — which is
what returning an empty position list does. These tests pin the boundary
between "confirmed flat" and "don't know".
"""
from datetime import datetime, timezone

import pytest

import config
from brokers.base import Broker, BrokerError, BrokerPosition, OrderResult
from executor import Executor, TradeState


class FlakyBroker(Broker):
    """A broker whose position feed can be switched on and off."""

    def __init__(self):
        self.available = True
        self.positions: dict[str, BrokerPosition] = {}
        self.closed: list[str] = []
        self.opened: list[tuple] = []
        self.stops_set: list[tuple] = []

    def get_equity(self):
        return 10_000.0

    def get_available_margin(self):
        return 10_000.0

    def get_positions(self):
        if not self.available:
            raise BrokerError("read timed out")
        return dict(self.positions)

    def open_position(self, symbol, side, qty, stop_loss=None, take_profit=None):
        self.opened.append((symbol, side, qty))
        return OrderResult(True, order_id='x', filled_price=100.0)

    def close_position(self, symbol, qty=None):
        self.closed.append(symbol)
        self.positions.pop(symbol, None)
        return OrderResult(True)

    def set_stops(self, symbol, stop_loss=None, take_profit=None):
        self.stops_set.append((symbol, stop_loss, take_profit))
        return OrderResult(True)

    def set_leverage(self, symbol, leverage):
        return True

    def get_instrument(self, symbol):
        from tests.test_risk import make_instrument
        return make_instrument(qty_step='0.000001', min_qty='0.000001')

    def get_price(self, symbol):
        return 100.0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr('executor.STATE_FILE', str(tmp_path / 'state.json'))
    broker = FlakyBroker()
    executor = Executor(broker)
    return broker, executor


def open_state(executor, symbol='BTCUSDT', side='long'):
    executor.states[symbol] = TradeState(
        symbol=symbol, side=side, entry_price=100.0,
        entry_time=datetime.now(timezone.utc).isoformat(),
        stop_price=98.0, take_price=104.0, atr=1.0, extreme=100.0,
    )


def live_position(symbol='BTCUSDT', side='long'):
    return BrokerPosition(
        symbol=symbol, side=side, qty=1.0, entry_price=100.0,
        mark_price=100.0, unrealised_pnl=0.0, stop_loss=98.0,
        created_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════
# THE BUG THIS FILE EXISTS FOR
# ══════════════════════════════════════════════════════════════

def test_outage_does_not_destroy_tracking_state(setup):
    """
    A failed position read must NOT be read as "everything closed".

    Before the fix, get_positions() returned {} on any API failure and
    reconciliation concluded that every tracked position had been closed at
    the exchange: state wiped, phantom outcomes logged, and the symbol freed
    for a duplicate entry into a position still open.
    """
    broker, executor = setup
    open_state(executor)
    broker.positions['BTCUSDT'] = live_position()

    broker.available = False
    executor.manage_positions()

    assert 'BTCUSDT' in executor.states, "tracking state destroyed by an outage"
    assert broker.closed == [], "an outage must not trigger closes"


def test_outage_does_not_log_phantom_outcomes(setup):
    """Phantom losses would also trip the loss-streak cooldown."""
    broker, executor = setup
    open_state(executor)
    broker.positions['BTCUSDT'] = live_position()

    broker.available = False
    executor.manage_positions()

    assert executor.recent_outcomes.get('BTCUSDT', []) == []
    assert not executor.is_in_cooldown('BTCUSDT')


def test_outage_blocks_new_entries(setup):
    """
    Fail closed: if we cannot confirm what we hold, we open nothing.

    The duplicate-position guard reads the live position list, so acting
    while it is unavailable risks doubling exposure.
    """
    broker, executor = setup
    broker.available = False

    assert executor.open_position('BTCUSDT', 'long', stop_price=98.0,
                                  take_price=104.0, atr=1.0) is False
    assert broker.opened == []


def test_recovery_reconciles_normally(setup):
    """Once the venue answers again, a genuine close is processed."""
    broker, executor = setup
    open_state(executor)
    broker.positions['BTCUSDT'] = live_position()

    broker.available = False
    executor.manage_positions()
    assert 'BTCUSDT' in executor.states

    # Exchange stop fired while we were blind; now the venue answers.
    broker.available = True
    broker.positions.clear()
    executor.manage_positions()

    assert 'BTCUSDT' not in executor.states
    assert executor.recent_outcomes.get('BTCUSDT')


def test_confirmed_flat_still_reconciles(setup):
    """
    An EMPTY list from a healthy venue is real information and must be acted
    on — otherwise closed positions would be tracked forever.
    """
    broker, executor = setup
    open_state(executor)
    broker.available = True          # answering, and genuinely flat

    executor.manage_positions()

    assert 'BTCUSDT' not in executor.states
    assert executor.recent_outcomes.get('BTCUSDT')


def test_polling_backs_off_while_flat(setup, monkeypatch):
    """
    Holding nothing, there is nothing to manage — so don't hammer the API.

    The tuned gates keep the bot flat most of the time, and polling a flaky
    demo endpoint every 30 s for an answer we already know just manufactures
    connection resets.
    """
    broker, executor = setup
    calls = []
    original = broker.get_positions
    monkeypatch.setattr(broker, 'get_positions',
                        lambda: (calls.append(1), original())[1])

    executor.manage_positions()          # first call always polls
    assert len(calls) == 1

    for _ in range(5):                   # subsequent calls throttled
        executor.manage_positions()
    assert len(calls) == 1


def test_polling_is_never_throttled_while_holding(setup, monkeypatch):
    """An open position must be checked every cycle, without exception."""
    broker, executor = setup
    open_state(executor)
    broker.positions['BTCUSDT'] = live_position()

    calls = []
    original = broker.get_positions
    monkeypatch.setattr(broker, 'get_positions',
                        lambda: (calls.append(1), original())[1])

    for _ in range(4):
        executor.manage_positions()
    assert len(calls) == 4


def test_throttle_expires(setup, monkeypatch):
    from datetime import timedelta
    broker, executor = setup

    executor.manage_positions()
    calls_before = len(broker.closed)     # any call proves it ran

    # Rewind the clock past the throttle window.
    executor._last_poll -= timedelta(seconds=config.FLAT_POLL_INTERVAL + 1)

    calls = []
    original = broker.get_positions
    monkeypatch.setattr(broker, 'get_positions',
                        lambda: (calls.append(1), original())[1])
    executor.manage_positions()
    assert len(calls) == 1


def test_trailing_stop_is_not_amended_during_an_outage(setup):
    broker, executor = setup
    open_state(executor)
    broker.positions['BTCUSDT'] = live_position()

    broker.available = False
    executor.manage_positions()

    assert broker.stops_set == []


# ══════════════════════════════════════════════════════════════
# THE BOT MUST OUTLIVE A NETWORK OUTAGE
# ══════════════════════════════════════════════════════════════

def test_a_failing_cycle_never_kills_the_scheduler():
    """
    Regression guard for a crash seen in the wild.

    When get_positions() was changed to RAISE on failure instead of
    returning {}, main.execution_cycle still called it unguarded. A dropped
    DNS lookup on an idle machine therefore propagated out through
    schedule.run_pending() and terminated the process — leaving live
    positions with nothing watching them.
    """
    import main

    class Boom:
        __name__ = 'exploding_cycle'

        def __call__(self):
            raise BrokerError("getaddrinfo failed")

    bot = main.TradingBot.__new__(main.TradingBot)   # no broker needed
    bot._safe(Boom())                                # must not raise

    def unexpected():
        raise ValueError("something nobody predicted")
    unexpected.__name__ = 'unexpected'

    bot._safe(unexpected)                            # must not raise either


def test_execution_cycle_opens_nothing_when_positions_are_unreadable(setup, monkeypatch):
    """Fail closed: no position list, no new orders."""
    import main

    broker, executor = setup
    broker.available = False

    bot = main.TradingBot.__new__(main.TradingBot)
    bot.broker = broker
    bot.executor = executor
    bot.decisions = {'BTCUSDT': type('D', (), {
        'direction': 'long', 'confidence': 0.9, 'reason': '', 'tags': []})()}

    bot.execution_cycle()          # must return quietly
    assert broker.opened == []


# ══════════════════════════════════════════════════════════════
# NO-OP STOP AMENDMENTS
# ══════════════════════════════════════════════════════════════

def test_trail_does_not_resend_an_unchanged_stop(setup, monkeypatch):
    """
    Regression guard for an endless 34040 "not modified" loop.

    trailing_stop_price compares the RAW candidate against the stored stop,
    but the exchange keeps prices rounded to the symbol's tick. A candidate
    fractionally above the current stop that rounds to the SAME tick is a
    no-op, and Bybit rejects it — once per management cycle, indefinitely,
    while the correct stop sat in place the whole time.
    """
    broker, executor = setup
    open_state(executor, side='long')
    position = live_position(side='long')
    position.stop_loss = 99.0
    broker.positions['BTCUSDT'] = position
    executor.states['BTCUSDT'].extreme = 120.0
    executor.states['BTCUSDT'].atr = 1.0

    # Candidate rounds to the tick already stored → must not be sent.
    monkeypatch.setattr('risk.trailing_stop_price',
                        lambda *a, **k: 99.000004)
    executor.manage_positions()
    assert broker.stops_set == [], "re-sent a stop the exchange already holds"

    # A candidate that clears a whole tick IS sent.
    monkeypatch.setattr('risk.trailing_stop_price', lambda *a, **k: 101.5)
    executor.manage_positions()
    assert len(broker.stops_set) == 1
    assert broker.stops_set[0][1] == pytest.approx(101.5)


def test_already_set_codes_are_not_errors():
    """34040 means 'the stop is already there' — success, not failure."""
    from brokers.bybit import _STOPS_UNCHANGED
    assert 34040 in _STOPS_UNCHANGED


# ══════════════════════════════════════════════════════════════
# RECORDING CLOSED TRADES HONESTLY
# ══════════════════════════════════════════════════════════════

def test_r_multiple_uses_the_original_stop_not_the_trailed_one(tmp_path):
    """
    Regression guard: a NEARUSDT long hit its full 3.89 R take-profit but was
    logged as +2.46 R, because R was divided by entry − CURRENT stop. After
    trailing, that distance is locked-in profit, not risk.
    """
    from edge_tracker import EdgeTracker
    tracker = EdgeTracker(path=str(tmp_path / 'edge.json'))

    entry, initial, trailed, exit_ = 2.786, 2.6679, 3.064, 3.247
    state = TradeState(symbol='NEARUSDT', side='long', entry_price=entry,
                       entry_time=datetime.now(timezone.utc).isoformat(),
                       stop_price=trailed, initial_stop=initial)
    move = (exit_ - entry) / entry
    tracker.record_close(state, exit_, move, 'win',
                         closed_at=datetime.now(timezone.utc))

    r = tracker.closed[-1].r_multiple
    assert r == pytest.approx(3.87, abs=0.05), f"logged {r:+.2f} R"


def test_reconcile_records_the_real_fill_not_the_price_when_noticed(setup, monkeypatch):
    """
    Regression guard: a take-profit fired at 3.247 while the machine slept;
    the bot noticed hours later at 3.473 and recorded THAT as the exit.
    """
    broker, executor = setup
    open_state(executor)
    recorded = {}
    executor.on_close = lambda state, price, pnl, outcome, closed_at=None: \
        recorded.update(price=price, closed_at=closed_at)

    real_close = datetime.now(timezone.utc)
    monkeypatch.setattr(broker, 'get_last_exit', lambda s, since: (104.0, real_close))
    monkeypatch.setattr(broker, 'get_price', lambda s: 150.0)   # market ran on

    broker.positions.clear()            # closed at the exchange
    executor.manage_positions()

    assert recorded['price'] == pytest.approx(104.0)
    assert recorded['closed_at'] == real_close


def test_reconcile_falls_back_when_the_venue_cannot_say(setup, monkeypatch):
    broker, executor = setup
    open_state(executor)
    recorded = {}
    executor.on_close = lambda state, price, pnl, outcome, closed_at=None: \
        recorded.update(price=price)

    broker.positions.clear()
    executor.manage_positions()         # FlakyBroker has no get_last_exit override
    assert recorded['price'] == pytest.approx(100.0)


def test_old_state_files_backfill_initial_stop(tmp_path, monkeypatch):
    """Positions saved before initial_stop existed must still load."""
    import json
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'states': {'ETHUSDT': {
        'symbol': 'ETHUSDT', 'side': 'long', 'entry_price': 2454.52,
        'entry_time': datetime.now(timezone.utc).isoformat(),
        'stop_price': 2417.18}}}))
    monkeypatch.setattr('executor.STATE_FILE', str(path))

    executor = Executor(FlakyBroker())
    assert executor.states['ETHUSDT'].initial_stop == pytest.approx(2417.18)
