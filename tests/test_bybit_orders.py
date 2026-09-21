"""
Tests for how the Bybit broker reads the exchange's answers.

The failure these defend: Bybit rejected every SUIUSDT order (above the
symbol's 66,000 market-order maximum) with retCode 10001, and the broker
logged each one as a fill, because 10001 sat in a list of "already set"
codes that was substring-matched against every call's error text. The
executor then tracked positions that never existed and recorded their
"exits" as trades.
"""
from dataclasses import replace
from decimal import Decimal

import pytest
from pybit.exceptions import InvalidRequestError

import risk
from brokers.base import BrokerPosition
from brokers.bybit import BybitFuturesBroker, _ret_code
from tests.test_risk import make_instrument

# The request dump pybit appends to its errors can contain any digits. These
# two are "already set" codes, planted to prove only the real retCode counts.
REQUEST = 'POST /v5/order/create: {"qty": "34040", "stopLoss": "110043"}'


def bybit_error(code: int, message: str = 'rejected') -> InvalidRequestError:
    return InvalidRequestError(request=REQUEST, message=message,
                               status_code=code, time='00:00:00', resp_headers={})


class FakeSession:
    """Stands in for pybit's HTTP session: records calls, replies as told."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else {
            'retCode': 0, 'result': {'orderId': 'abc123'}}
        self.error = error
        self.calls = []

    def _reply(self, name, kwargs):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return self.response

    def place_order(self, **kwargs):
        return self._reply('place_order', kwargs)

    def set_trading_stop(self, **kwargs):
        return self._reply('set_trading_stop', kwargs)

    def set_leverage(self, **kwargs):
        return self._reply('set_leverage', kwargs)


def make_broker(session, instrument=None, price=100.0):
    """A broker wired to a fake session, with no network and no API keys."""
    broker = BybitFuturesBroker.__new__(BybitFuturesBroker)
    broker.session = session
    inst = instrument or make_instrument()
    broker.get_instrument = lambda symbol: inst
    broker.get_price = lambda symbol: price
    return broker


def sui_instrument():
    """SUIUSDT's real rules: whole lots of 10, 0.0001 ticks, 66,000 per market order."""
    return replace(make_instrument(qty_step='10', min_qty='10', tick='0.00010',
                                   max_qty='10000000'),
                   max_mkt_qty=Decimal('66000'))


# ══════════════════════════════════════════════════════════════
# REJECTIONS ARE NEVER FILLS
# ══════════════════════════════════════════════════════════════

def test_rejected_order_is_not_a_fill():
    error = bybit_error(10001, 'The number of contracts exceeds maximum limit allowed')
    broker = make_broker(FakeSession(error=error))
    result = broker.open_position('SUIUSDT', 'long', 1.0, stop_loss=95.0, take_profit=110.0)
    assert not result.ok


def test_rejected_close_is_not_a_close():
    broker = make_broker(FakeSession(error=bybit_error(10001)))
    broker.get_position = lambda symbol: BrokerPosition(
        symbol=symbol, side='long', qty=1.0, entry_price=100.0,
        mark_price=100.0, unrealised_pnl=0.0)
    assert not broker.close_position('TESTUSDT').ok


def test_order_without_an_id_is_not_a_fill():
    broker = make_broker(FakeSession(response={'retCode': 0, 'result': {}}))
    assert not broker.open_position('TESTUSDT', 'long', 1.0).ok


def test_accepted_order_is_a_fill():
    result = make_broker(FakeSession()).open_position('TESTUSDT', 'long', 1.0)
    assert result.ok
    assert result.order_id == 'abc123'


# ══════════════════════════════════════════════════════════════
# "ALREADY SET" IS SUCCESS — BUT ONLY WHERE IT MEANS THAT
# ══════════════════════════════════════════════════════════════

def test_unchanged_stop_is_success():
    broker = make_broker(FakeSession(error=bybit_error(34040)))
    assert broker.set_stops('TESTUSDT', stop_loss=95.0).ok


def test_unchanged_leverage_is_success():
    broker = make_broker(FakeSession(error=bybit_error(110043)))
    assert broker.set_leverage('TESTUSDT', 3) is True


@pytest.mark.parametrize('call', ['stops', 'leverage'])
def test_generic_param_error_is_failure_even_when_codes_appear_in_the_text(call):
    broker = make_broker(FakeSession(error=bybit_error(10001)))
    if call == 'stops':
        assert not broker.set_stops('TESTUSDT', stop_loss=95.0).ok
    else:
        assert broker.set_leverage('TESTUSDT', 3) is False


def test_ret_code_falls_back_to_the_message():
    assert _ret_code(Exception('leverage not modified (ErrCode: 110043) (ErrTime: 1)')) == 110043
    assert _ret_code(Exception('connection reset')) is None


# ══════════════════════════════════════════════════════════════
# QUANTITIES AND PRICES THE EXCHANGE WILL ACCEPT
# ══════════════════════════════════════════════════════════════

def test_qty_is_a_whole_number_of_steps():
    assert sui_instrument().round_qty(130225.73) == 66000.0   # capped
    assert sui_instrument().round_qty(12345.6) == 12340.0     # whole lots of 10


def test_price_is_a_whole_number_of_ticks():
    # tick '0.00010' means 0.0001 steps, not five decimal places
    assert sui_instrument().round_price(0.83042) == pytest.approx(0.8304)
    assert make_instrument(tick='0.10').round_price(83998.48) == pytest.approx(83998.5)


def test_open_never_sends_more_than_the_market_order_maximum():
    session = FakeSession()
    broker = make_broker(session, instrument=sui_instrument(), price=0.8546)
    assert broker.open_position('SUIUSDT', 'long', 130225.73).ok
    sent = float(session.calls[0][1]['qty'])
    assert sent == 66000.0


def test_sizing_caps_at_the_market_maximum_and_reports_the_lower_risk():
    equity, entry, stop = 300_000.0, 0.8546, 0.83042
    sized = risk.size_position(equity, entry, stop, sui_instrument())
    assert sized.ok
    assert sized.qty == 66000.0
    assert sized.risk_amount == pytest.approx(66000 * (entry - stop))
    assert sized.risk_amount < equity * 0.01


def test_sizing_below_the_maximum_is_unchanged():
    sized = risk.size_position(10_000.0, 0.8546, 0.83042, sui_instrument())
    assert sized.ok
    assert sized.qty % 10 == 0
    assert sized.risk_amount == pytest.approx(10_000 * 0.01, rel=0.01)
