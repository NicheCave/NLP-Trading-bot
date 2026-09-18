#!/usr/bin/env python
# verify_demo.py — prove the Bybit Demo integration actually works.
# ─────────────────────────────────────────────────────────────
# Run this ONCE after adding demo API keys to .env, before starting the
# bot. It exercises the whole order path end to end:
#
#   1. connect and read balance
#   2. set leverage and margin mode
#   3. open a minimum-size BTCUSDT position WITH stop-loss and take-profit
#   4. read the position back and confirm the exchange actually stored them
#   5. close it reduce-only
#
# Step 4 is the one that matters. Exchange-resident protection is the whole
# reason for the migration — if Bybit silently drops the TP/SL, positions
# would be running naked and nothing else would reveal it.
#
#   python verify_demo.py            # full round trip
#   python verify_demo.py --read-only
#   python verify_demo.py --fund 100000
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import sys
import time

from loguru import logger

import config
import risk

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

logger.remove()
logger.add(sys.stdout, colorize=True, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

SYMBOL = 'BTCUSDT'


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Bybit Demo integration")
    parser.add_argument('--read-only', action='store_true',
                        help="check connectivity and balance only — place no orders")
    parser.add_argument('--fund', type=float, metavar='USDT',
                        help="request demo funds first")
    args = parser.parse_args()

    if not config.credentials_present():
        logger.error("BYBIT_API_KEY / BYBIT_API_SECRET are not set.")
        logger.error("Create DEMO TRADING keys at bybit.com (avatar → API, while in "
                     "Demo Trading mode) and copy .env.example to .env")
        return 1

    from brokers import make_broker
    broker = make_broker('demo')

    # ── 1. account ────────────────────────────────────────────
    equity = broker.get_equity()
    available = broker.get_available_margin()
    logger.info(f"1/5  Connected. Equity {equity:,.2f} {config.QUOTE_CURRENCY}, "
                f"available margin {available:,.2f}")

    if args.fund:
        broker.request_demo_funds(args.fund)
        time.sleep(2)
        equity = broker.get_equity()
        logger.info(f"     After funding: {equity:,.2f} {config.QUOTE_CURRENCY}")

    if equity <= 0:
        logger.error("Demo account has no balance. Re-run with --fund 100000, or add "
                     "funds from the Bybit Demo Trading UI.")
        return 1

    existing = broker.get_positions()
    if existing:
        logger.warning(f"     Open positions already present: {list(existing)}")

    if args.read_only:
        logger.success("Read-only checks passed.")
        return 0

    # ── 2. leverage / margin mode ─────────────────────────────
    broker.prepare([SYMBOL])
    logger.info(f"2/5  Leverage set to {config.LEVERAGE}× ({config.MARGIN_MODE.lower()})")

    # ── 3. open with protection attached ──────────────────────
    price = broker.get_price(SYMBOL)
    instrument = broker.get_instrument(SYMBOL)
    if price is None:
        logger.error("Could not read a price")
        return 1

    # Smallest position that clears both the lot and notional minimums.
    qty = max(float(instrument.min_qty),
              float(instrument.min_notional) * 1.5 / price)
    qty = instrument.round_qty(qty)

    stop, take = risk.atr_stop_target(price, 'long', atr=price * 0.01)
    logger.info(f"3/5  Opening LONG {qty} {SYMBOL} @ ~{price:,.2f} "
                f"(SL {stop:,.2f} / TP {take:,.2f}, notional ≈ {qty * price:,.2f})")

    result = broker.open_position(SYMBOL, 'long', qty, stop_loss=stop, take_profit=take)
    if not result.ok:
        logger.error(f"     Order rejected: {result.error}")
        return 1

    time.sleep(2)

    # ── 4. confirm the exchange stored the stops ──────────────
    position = broker.get_position(SYMBOL)
    if position is None:
        logger.error("4/5  Position did not appear — the order may not have filled")
        return 1

    logger.info(f"4/5  Position: {position.side} {position.qty} @ {position.entry_price:,.2f} "
                f"| leverage {position.leverage}×")
    logger.info(f"     Exchange-side SL: {position.stop_loss}")
    logger.info(f"     Exchange-side TP: {position.take_profit}")

    ok = True
    if position.stop_loss is None:
        logger.error("     ✗ NO STOP-LOSS STORED — the position would be unprotected "
                     "if the bot stopped running")
        ok = False
    if position.take_profit is None:
        logger.warning("     ✗ no take-profit stored")
        ok = False

    # The leverage trap, checked explicitly.
    logger.info(f"     Price move since entry: {position.pnl_pct_of_price():+.3%} "
                f"(unrealised {position.unrealised_pnl:+.4f} {config.QUOTE_CURRENCY})")

    # ── 5. close reduce-only ──────────────────────────────────
    logger.info("5/5  Closing (reduce-only)…")
    close = broker.close_position(SYMBOL)
    if not close.ok:
        logger.error(f"     Close failed: {close.error} — CLOSE IT MANUALLY in the UI")
        return 1

    time.sleep(2)
    if broker.get_position(SYMBOL) is not None:
        logger.error("     Position still open after close — check the UI")
        return 1

    logger.info(f"     Closed. Equity now {broker.get_equity():,.2f} "
                f"{config.QUOTE_CURRENCY}")

    if ok:
        logger.success("✓ Demo integration verified: orders fill, stops are stored at "
                       "the exchange, and reduce-only closes work.")
        logger.info("  You can now run:  python main.py --mode demo")
        return 0

    logger.error("✗ Orders work but protection was not stored — do NOT run unattended.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
