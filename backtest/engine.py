# backtest/engine.py — event-driven portfolio simulator for USDT perps.
# ─────────────────────────────────────────────────────────────
# The point of this module is to make every later change *measurable*.
# Nothing in the strategy should be tuned until it can be scored here.
#
# Timing model (this is where most backtests quietly cheat):
#
#   bar i covers [t_i, t_i + interval).
#   At each bar i the engine does, strictly in this order:
#
#     1. FILL   orders decided at bar i-1's close, at bar i's OPEN
#     2. EXIT   check bar i's high/low against SL/TP of open positions
#     3. FUND   charge funding for any settlement inside bar i
#     4. DECIDE run the strategy on bars 0..i (bar i is now closed)
#               → orders queued for bar i+1
#
#   A decision at step 4 can therefore never fill earlier than the next
#   bar's open, so no signal can act on a price it could not have seen.
#
# Costs modelled: taker fee both ways, slippage, and 8-hourly funding.
# Ignoring funding on a 48h-hold perp strategy overstates returns by
# several percent a year, so it is on by default.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════
# TYPES
# ══════════════════════════════════════════════════════════════

@dataclass
class Intent:
    """
    What the strategy wants to do at one bar.

    The strategy decides direction and *where the stop belongs* (that is a
    market-structure judgement). The engine decides size and whether the
    trade is permitted — that is a portfolio judgement.
    """
    direction:  str                       # 'long' | 'short' | 'flat'
    confidence: float = 0.0               # 0..1
    stop_price: Optional[float] = None
    take_price: Optional[float] = None
    atr:        Optional[float] = None    # ATR at decision time, for trailing
    reason:     str = ''
    tags:       list[str] = field(default_factory=list)


@dataclass
class Trade:
    """One completed round trip, with everything needed to score it."""
    symbol:      str
    side:        str          # 'long' | 'short'
    entry_ts:    pd.Timestamp
    entry_price: float
    exit_ts:     pd.Timestamp
    exit_price:  float
    qty:         float
    gross_pnl:   float
    fees:        float
    funding:     float
    net_pnl:     float
    r_multiple:  float        # net PnL / initial risk — the unit that matters
    risk_amount: float
    exit_reason: str
    bars_held:   int
    mae:         float        # max adverse excursion, in R
    mfe:         float        # max favourable excursion, in R
    confidence:  float
    tags:        list[str] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        """A win is NET of fees and funding — not gross."""
        return self.net_pnl > 0


@dataclass
class BacktestConfig:
    initial_equity:  float = 10_000.0
    taker_fee:       float = 0.00055   # Bybit taker, 0.055%
    slippage_bps:    float = 2.0       # each side, on top of the fee
    risk_per_trade:  float = 0.01      # 1% of equity at the stop
    max_positions:   int   = 4
    max_leverage:    float = 3.0
    funding_enabled: bool  = True
    # Cap on a single position's notional as a multiple of equity. Stops a
    # very tight stop from demanding an absurd position.
    max_notional_mult: float = 1.0

    # ── exit policy ───────────────────────────────────────────
    # Time stop: perps bleed funding on a schedule, so a position that has
    # gone nowhere is costing money to hold. 0 disables.
    time_stop_bars:     int   = 48
    # ATR trailing stop, armed only once the trade is genuinely working.
    use_trailing:       bool  = True
    trailing_atr_mult:  float = 2.0
    trailing_activate_r: float = 1.0


@dataclass
class Position:
    symbol:       str
    side:         str
    qty:          float
    entry_price:  float
    entry_ts:     pd.Timestamp
    stop_price:   float
    take_price:   Optional[float]
    risk_amount:  float
    confidence:   float
    tags:         list[str]
    entry_fee:    float
    atr:          Optional[float] = None
    trail_stop:   Optional[float] = None   # armed only after +activate_R
    funding_paid: float = 0.0
    bars_held:    int = 0
    mae_price:    float = 0.0    # worst price seen
    mfe_price:    float = 0.0    # best price seen

    def unrealised(self, price: float) -> float:
        d = price - self.entry_price
        return d * self.qty if self.side == 'long' else -d * self.qty

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class BacktestResult:
    trades:       list[Trade]
    equity_curve: pd.Series
    config:       BacktestConfig
    bars:         int = 0
    skipped:      dict[str, int] = field(default_factory=dict)


# Strategy signature: (symbol, hist_1h, hist_1d, ts) -> Intent | None
StrategyFn = Callable[[str, pd.DataFrame, pd.DataFrame, pd.Timestamp], Optional[Intent]]


# ══════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Multi-symbol portfolio simulator.

    Intents can either be produced live by a strategy callable, or supplied
    pre-computed as {symbol: DataFrame indexed by bar ts}. The second form
    matters because signal generation is the slow part: compute it once,
    then re-run the portfolio simulation instantly across many different
    risk settings.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg = config or BacktestConfig()

    # ── sizing ────────────────────────────────────────────────
    def _size(self, equity: float, entry: float, stop: float) -> tuple[float, float]:
        """
        Risk-based position size.

            qty = (equity × risk_per_trade) / |entry − stop|

        This is the correction to the live bot's current sizing, which
        allocates a fixed fraction of equity as *notional* and therefore
        risks only risk_pct × stop_pct — about 0.05% per trade instead of 1%.

        Returns (qty, risk_amount). qty is 0 when the trade can't be sized.
        """
        stop_distance = abs(entry - stop)
        if stop_distance <= 0 or entry <= 0:
            return 0.0, 0.0

        risk_amount = equity * self.cfg.risk_per_trade
        qty = risk_amount / stop_distance

        # Respect leverage and the per-position notional cap.
        max_notional = min(
            equity * self.cfg.max_leverage,
            equity * self.cfg.max_notional_mult * self.cfg.max_leverage,
        )
        if qty * entry > max_notional:
            qty = max_notional / entry
            risk_amount = qty * stop_distance   # the real risk after clamping

        return qty, risk_amount

    def _slip(self, price: float, side: str, entering: bool) -> float:
        """Apply slippage in whichever direction hurts."""
        bps = self.cfg.slippage_bps / 10_000.0
        adverse_up = (side == 'long') == entering
        return price * (1 + bps) if adverse_up else price * (1 - bps)

    # ── main loop ─────────────────────────────────────────────
    def run(self,
            data: dict[str, dict[str, pd.DataFrame]],
            strategy: Optional[StrategyFn] = None,
            intents: Optional[dict[str, pd.DataFrame]] = None,
            warmup: int = 300) -> BacktestResult:
        """
        Simulate the portfolio.

        Args:
            data:     {symbol: {'1h': df, '1d': df, 'funding': df}}
            strategy: callable invoked at each closed bar (slow path)
            intents:  pre-computed intents per symbol (fast path)
            warmup:   bars skipped at the start so indicators are warm
        """
        if strategy is None and intents is None:
            raise ValueError("Provide either a strategy callable or precomputed intents")

        symbols = list(data.keys())

        # Master clock: the union of all symbols' 1H bar timestamps.
        timeline = sorted(set().union(*(set(data[s]['1h'].index) for s in symbols)))
        timeline = pd.DatetimeIndex(timeline)

        equity = self.cfg.initial_equity
        positions: dict[str, Position] = {}
        pending: dict[str, Intent] = {}          # decided at bar i-1, fills at bar i
        trades: list[Trade] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        skipped: dict[str, int] = {}

        def skip(why: str) -> None:
            skipped[why] = skipped.get(why, 0) + 1

        # Pre-index funding for O(1) lookup per bar.
        funding_maps = {
            s: data[s].get('funding', pd.DataFrame()) for s in symbols
        }

        for bar_i, ts in enumerate(timeline):
            bar_end = ts + pd.Timedelta(hours=1)

            # ── 1. FILL pending orders at this bar's OPEN ─────
            # Highest conviction first. Once the universe is larger than
            # max_positions, several symbols routinely signal on the same
            # bar and only some can be taken — filling in dict order would
            # hand the slots to whichever symbol happens to sit earliest in
            # the watchlist, regardless of signal strength.
            queued = sorted(pending.items(),
                            key=lambda kv: -kv[1].confidence)
            for symbol, intent in queued:
                pending.pop(symbol, None)
                df = data[symbol]['1h']
                if ts not in df.index or symbol in positions:
                    continue
                if len(positions) >= self.cfg.max_positions:
                    skip('max_positions')
                    continue
                if intent.stop_price is None:
                    skip('no_stop')
                    continue

                raw_open = float(df.at[ts, 'open'])
                side = 'long' if intent.direction == 'long' else 'short'

                # A gap through the stop means the trade is already invalid.
                if (side == 'long' and raw_open <= intent.stop_price) or \
                   (side == 'short' and raw_open >= intent.stop_price):
                    skip('gapped_through_stop')
                    continue

                entry = self._slip(raw_open, side, entering=True)
                qty, risk_amount = self._size(equity, entry, intent.stop_price)
                if qty <= 0 or risk_amount <= 0:
                    skip('unsizeable')
                    continue

                fee = qty * entry * self.cfg.taker_fee
                equity -= fee
                positions[symbol] = Position(
                    symbol=symbol, side=side, qty=qty,
                    entry_price=entry, entry_ts=ts,
                    stop_price=intent.stop_price, take_price=intent.take_price,
                    risk_amount=risk_amount, confidence=intent.confidence,
                    tags=list(intent.tags), entry_fee=fee, atr=intent.atr,
                    mae_price=entry, mfe_price=entry,
                )

            # ── 2. EXIT checks against this bar's range ───────
            for symbol, pos in list(positions.items()):
                df = data[symbol]['1h']
                if ts not in df.index:
                    continue

                high = float(df.at[ts, 'high'])
                low = float(df.at[ts, 'low'])
                close = float(df.at[ts, 'close'])
                pos.bars_held += 1

                # The trailing level in play during THIS bar is the one
                # derived from previous bars. Updating it from this bar's
                # own extreme before testing it would let the stop chase a
                # high the position could not yet have seen.
                active_trail = pos.trail_stop

                if pos.side == 'long':
                    hit_stop = low <= pos.stop_price
                    hit_trail = active_trail is not None and low <= active_trail
                    hit_take = pos.take_price is not None and high >= pos.take_price
                else:
                    hit_stop = high >= pos.stop_price
                    hit_trail = active_trail is not None and high >= active_trail
                    hit_take = pos.take_price is not None and low <= pos.take_price

                exit_price: Optional[float] = None
                reason = ''
                # When several levels sit inside one bar we cannot know which
                # came first from OHLC alone. Resolve to the worst outcome —
                # stop, then trail, then target — rather than flatter the run.
                if hit_stop:
                    exit_price, reason = pos.stop_price, 'stop'
                elif hit_trail:
                    exit_price, reason = active_trail, 'trailing_stop'
                elif hit_take:
                    exit_price, reason = pos.take_price, 'target'
                elif self.cfg.time_stop_bars and pos.bars_held >= self.cfg.time_stop_bars:
                    exit_price, reason = close, 'time_stop'

                if exit_price is not None:
                    equity = self._close(pos, exit_price, ts, reason, equity, trades)
                    positions.pop(symbol, None)
                    continue

                # Position survived the bar — now fold this bar into the
                # excursion record and re-arm the trail for the NEXT bar.
                if pos.side == 'long':
                    pos.mae_price = min(pos.mae_price, low)
                    pos.mfe_price = max(pos.mfe_price, high)
                else:
                    pos.mae_price = max(pos.mae_price, high)
                    pos.mfe_price = min(pos.mfe_price, low)

                self._update_trail(pos)

            # ── 3. FUNDING for settlements inside this bar ────
            if self.cfg.funding_enabled:
                for symbol, pos in positions.items():
                    fdf = funding_maps.get(symbol)
                    if fdf is None or fdf.empty:
                        continue
                    due = fdf.loc[(fdf.index >= ts) & (fdf.index < bar_end)]
                    if due.empty:
                        continue
                    df = data[symbol]['1h']
                    price = float(df.at[ts, 'close']) if ts in df.index else pos.entry_price
                    for rate in due['funding_rate']:
                        # Longs pay a positive rate; shorts receive it.
                        cost = pos.qty * price * float(rate)
                        if pos.side == 'short':
                            cost = -cost
                        pos.funding_paid += cost
                        equity -= cost

            # ── 4. DECIDE using bars up to and including this one ──
            if bar_i >= warmup and bar_i < len(timeline) - 1:
                for symbol in symbols:
                    df = data[symbol]['1h']
                    if ts not in df.index:
                        continue

                    intent = self._get_intent(symbol, data, intents, strategy, ts)
                    if intent is None or intent.direction == 'flat':
                        continue

                    if symbol in positions:
                        # Already positioned; a reversal closes at next open,
                        # which the current strategy does not request. Skip.
                        continue
                    if len(positions) >= self.cfg.max_positions:
                        skip('max_positions')
                        continue
                    pending[symbol] = intent

            # Mark equity to market once per bar.
            marked = equity
            for symbol, pos in positions.items():
                df = data[symbol]['1h']
                if ts in df.index:
                    marked += pos.unrealised(float(df.at[ts, 'close']))
            equity_points.append((ts, marked))

        # Force-close anything still open at the end of the sample.
        if positions:
            last_ts = timeline[-1]
            for symbol, pos in list(positions.items()):
                df = data[symbol]['1h']
                price = float(df['close'].iloc[-1])
                equity = self._close(pos, price, last_ts, 'end_of_data', equity, trades)

        curve = pd.Series(dict(equity_points)).sort_index()
        return BacktestResult(trades=trades, equity_curve=curve,
                              config=self.cfg, bars=len(timeline), skipped=skipped)

    # ── helpers ───────────────────────────────────────────────
    def _update_trail(self, pos: Position) -> None:
        """
        Move the trailing stop up (long) or down (short), never backwards.

        Armed only once the position has run trailing_activate_r in our
        favour. Trailing from the first bar converts ordinary noise into
        stop-outs and is one of the quickest ways to turn a positive-edge
        system negative.
        """
        if not self.cfg.use_trailing or not pos.atr or pos.atr <= 0:
            return

        per_unit = pos.risk_amount / pos.qty if pos.qty else 0.0
        if per_unit <= 0:
            return

        if pos.side == 'long':
            excursion_r = (pos.mfe_price - pos.entry_price) / per_unit
            if excursion_r < self.cfg.trailing_activate_r:
                return
            level = pos.mfe_price - self.cfg.trailing_atr_mult * pos.atr
            pos.trail_stop = level if pos.trail_stop is None else max(pos.trail_stop, level)
        else:
            excursion_r = (pos.entry_price - pos.mfe_price) / per_unit
            if excursion_r < self.cfg.trailing_activate_r:
                return
            level = pos.mfe_price + self.cfg.trailing_atr_mult * pos.atr
            pos.trail_stop = level if pos.trail_stop is None else min(pos.trail_stop, level)

    def _get_intent(self, symbol, data, intents, strategy, ts) -> Optional[Intent]:
        """Read a precomputed intent, or invoke the strategy callable."""
        if intents is not None:
            idf = intents.get(symbol)
            if idf is None or ts not in idf.index:
                return None
            row = idf.loc[ts]
            if row['direction'] == 'flat':
                return None
            def opt(col):
                val = row.get(col)
                return None if val is None or pd.isna(val) else float(val)

            # Parquet round-trips a list column back as a numpy array, and
            # `array or []` raises on truthiness. Convert explicitly.
            raw_tags = row.get('tags')
            tags = list(raw_tags) if raw_tags is not None and len(raw_tags) else []

            return Intent(
                direction=row['direction'],
                confidence=float(row.get('confidence', 0.0)),
                stop_price=opt('stop_price'),
                take_price=opt('take_price'),
                atr=opt('atr'),
                reason=str(row.get('reason', '')),
                tags=tags,
            )

        df_1h = data[symbol]['1h']
        hist_1h = df_1h.loc[:ts]
        df_1d = data[symbol].get('1d')
        # Only daily bars that have actually CLOSED by now.
        hist_1d = (df_1d[df_1d.index <= ts - pd.Timedelta(days=1)]
                   if df_1d is not None and not df_1d.empty else pd.DataFrame())
        return strategy(symbol, hist_1h, hist_1d, ts)

    def _close(self, pos: Position, raw_price: float, ts: pd.Timestamp,
               reason: str, equity: float, trades: list[Trade]) -> float:
        """Close a position, book the costs, and record the trade."""
        exit_price = self._slip(raw_price, pos.side, entering=False)
        gross = pos.unrealised(exit_price)
        exit_fee = pos.qty * exit_price * self.cfg.taker_fee
        fees = pos.entry_fee + exit_fee

        # Both legs count against the trade. The entry fee was already taken
        # out of running equity when the position opened, so it must NOT be
        # deducted from equity again here — but it absolutely belongs in the
        # trade's own P&L, and therefore in the win/loss test. Omitting it
        # flatters every marginal trade into a "win".
        net = gross - fees - pos.funding_paid

        equity += gross - exit_fee

        risk = pos.risk_amount if pos.risk_amount > 0 else float('nan')
        per_unit = pos.risk_amount / pos.qty if pos.qty else float('nan')
        if pos.side == 'long':
            mae = (pos.entry_price - pos.mae_price) / per_unit
            mfe = (pos.mfe_price - pos.entry_price) / per_unit
        else:
            mae = (pos.mae_price - pos.entry_price) / per_unit
            mfe = (pos.entry_price - pos.mfe_price) / per_unit

        trades.append(Trade(
            symbol=pos.symbol, side=pos.side,
            entry_ts=pos.entry_ts, entry_price=pos.entry_price,
            exit_ts=ts, exit_price=exit_price, qty=pos.qty,
            gross_pnl=gross, fees=fees, funding=pos.funding_paid,
            net_pnl=net, r_multiple=net / risk if risk == risk else 0.0,
            risk_amount=pos.risk_amount, exit_reason=reason,
            bars_held=pos.bars_held, mae=mae, mfe=mfe,
            confidence=pos.confidence, tags=pos.tags,
        ))
        return equity
