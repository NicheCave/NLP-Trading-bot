# Bybit Perpetual Futures Bot (v7)

Migrated from Alpaca crypto **spot, long-only** to Bybit **USDT perpetual futures**,
trading both directions on **Demo Trading** (real mainnet prices, fake money).

---

## Quick start

```bash
pip install -r requirements.txt
```

Backtest — needs no API keys at all (Bybit's market endpoints are public):

```bash
python run_backtest.py --days 730 --variants baseline,shorts,v7
```

Paper trade against real live prices, still no keys:

```bash
python main.py --mode paper
```

Trade on Bybit Demo — copy `.env.example` to `.env` and add **Demo Trading** keys
(created from a mainnet account: avatar → API, while switched into Demo Trading;
testnet keys will not work), then verify the integration before running unattended:

```bash
python verify_demo.py --fund 100000
```

```bash
python main.py --mode demo
```

---

## What changed, and why

### 1. It can trade both directions

Previously `SELL` only meant "close the long, if any" — spot cannot short. Every
bearish vote the indicator stack produced (bear traps, bearish CHoCH, bear sweeps)
was computed and discarded. On perps `SELL` opens a short.

### 2. Position sizing now does what the config claimed

The old rule was:

```python
dollars = portfolio * RISK_PER_TRADE * confidence
qty     = dollars / price
```

That allocates 1% of equity as **notional**. With a 5% stop the actual loss at the
stop was 1% × 5% = **0.05% of equity — 20× smaller than intended**, and unrelated to
where the stop sat. Now:

```python
qty = (equity * RISK_PER_TRADE) / abs(entry - stop)
```

Every trade loses the same fraction of equity when it is wrong, which is what makes
R-multiples comparable and expectancy meaningful. Leverage caps notional; it is not
the risk lever. Pinned by `tests/test_risk.py`.

### 3. Stops live at the exchange

`stopLoss` / `takeProfit` are attached to the entry order and trigger on **mark
price** (index-anchored, much harder to wick than last price). If the bot crashes or
the network drops, positions stay protected. The old design polled every 30 s and
sent market orders — survivable on unleveraged spot, not on leveraged perps.

### 4. Volatility-scaled stops

Flat 5% / 7% applied the same distance to BTC and SOL. Stops are now ATR multiples,
clamped to a sane band.

### 5. Correlated indicators no longer vote five times

`ema_cross`, `macd`, `williams_ao`, `williams_alligator` and `mtf_daily_align` all
measure roughly one thing: whether price has been rising. The old aggregation summed
all 24 votes as independent evidence, so those five contributed ~8.5 of ~30 total
weight and could clear the entry threshold on momentum alone — and they agree most
strongly when a move is already extended. That is a structural late-entry bias, not a
tuning problem.

Now votes aggregate **within** a factor family first (trend / momentum /
mean-reversion / structure / volume / reversal / context), then across families.
See `features/blocks.py`.

### 6. Regime routing

`features/regime.py` classifies each bar using three orthogonal measurements — ADX
(is there a trend?), Kaufman Efficiency Ratio (is price going somewhere, or just
moving?) and realised-volatility percentile (is this survivable?). The regime then
sets how much each factor family is trusted: trend evidence in trends,
mean-reversion in ranges, and a stand-down in volatile chop.

This replaces the old single binary Alligator "sleeping" veto.

### 7. Futures-only signals

`features/derivatives.py` reads funding rate and open interest — direct measurements
of leveraged positioning with no spot equivalent:

| price | open interest | reading |
|---|---|---|
| ↑ | ↑ | new longs — genuine demand |
| ↑ | ↓ | short covering — fuel exhausting |
| ↓ | ↑ | new shorts — genuine supply |
| ↓ | ↓ | long liquidation — capitulation |

Funding is also a real cost: the BTCUSDT mean over the last two years is +0.0047%
per 8 h, about **5.2%/yr just to hold longs**. It is charged in the backtest.

### 8. One bounded confidence calculation

The old score started at `net × 2.5` then applied up to seven boosts (×1.20, ×1.15,
×1.10, ×1.08, ×1.10, ×1.07, ×1.05) and four penalties (×0.70, ×0.80, ×0.60, ×0.65),
clipping at 1.0 throughout — so all strong signals collapsed to the same value, and
the order of multiplication changed the answer. It is now a single logistic in
signal strength and cross-family agreement, monotonic and fittable.

### 9. Everything is measurable

There was no backtest. Every threshold was hand-tuned by feel (`config.py` was full
of comments like `v6: raised from 1.5`). There is now an event-driven backtester with
fees, slippage and funding, plus walk-forward tuning.

---

## Architecture

```
data/            Bybit public market data (klines, funding, OI) + parquet cache
brokers/         base.py (ABC) · bybit.py (Demo/live) · sim.py (local paper)
features/        regime.py · derivatives.py · blocks.py
indicators.py    the original TA/SMC stack + ADX, efficiency ratio, squeeze
market_analyzer.py  indicator votes  (analyze_frames is pure — no I/O, no clock)
strategy_v7.py   regime routing + block aggregation + gates → Intent
risk.py          ATR sizing, portfolio heat, correlation cap, trailing, funding exit
executor.py      order placement against the Broker interface
backtest/        engine.py · metrics.py · signals.py · feature_stream.py · walkforward.py
main.py          live orchestrator (--mode paper|demo|live)
```

The same strategy code runs in backtest, paper and demo because everything talks to
the `Broker` interface.

### No-lookahead guarantee

The backtest is explicit about timing, since this is where most backtests quietly
cheat. Per bar, strictly in order:

1. **Fill** orders decided at bar *i−1*'s close, at bar *i*'s **open**
2. **Exit** — test bar *i*'s high/low against stops (trailing levels are those
   derived from *earlier* bars only)
3. **Fund** — charge any funding settlement inside bar *i*
4. **Decide** — run the strategy on bars 0..*i*, queueing orders for *i+1*

When several exit levels sit inside one bar, OHLC cannot say which came first, so the
engine resolves to the **worst** outcome. `run_backtest.py` asserts the property
directly, and `tests/test_engine.py` pins it.

---

## Measured results

Two years of 1H data, BTCUSDT/ETHUSDT/SOLUSDT, 1% risk per trade, Bybit taker fees
(0.055%), 2 bps slippage, funding charged.

| | baseline (as-was) | + shorts | v7 (untuned) |
|---|---|---|---|
| Trades | 816 | 1,201 | 3,018 |
| Win rate | 50.0% | 48.0% | 42.7% |
| Break-even win rate | 50.6% | 48.1% | 46.8% |
| **Gross** PnL (pre-cost) | **+$1,154** | **+$1,777** | **+$1,223** |
| Fees | −$1,886 | −$2,427 | −$10,238 |
| Funding | −$409 | −$157 | −$410 |
| **Net** PnL | **−$1,142** | **−$807** | **−$9,425** |
| Expectancy (net) | −0.009 R | −0.002 R | −0.082 R |
| Total return | −11.4% | −8.0% | −94.2% |
| Max drawdown | −31.6% | −25.4% | −95.4% |

### The finding that matters

Splitting expectancy into its gross and cost halves makes the whole picture legible:

| per trade | baseline | + shorts | v7 untuned |
|---|---|---|---|
| Gross expectancy | **+0.018 R** | **+0.022 R** | −0.007 R |
| Cost per round trip | −0.026 R | −0.023 R | **−0.075 R** |
| **Net expectancy** | −0.009 R | −0.002 R | −0.082 R |
| t-statistic | −0.31 | −0.07 | −3.94 |

**The legacy signals have a small but real gross edge of ~0.02 R per trade, and it
costs ~0.025 R to collect it.** The edge is not imaginary — it is just smaller than
the toll. That is a completely different problem from "the indicators don't work",
and it is not fixed by adding more indicators.

There is a clean formula behind it. Because size is set by `risk / stop_distance`,
the notional — and therefore the fee — scales inversely with how wide the stop is:

```
cost_in_R  ≈  round_trip_fee_%  /  stop_%
```

which the measurements confirm almost exactly:

| | stop width | predicted | measured |
|---|---|---|---|
| baseline | 5% | 0.11 / 5 = 0.022 R | 0.026 R |
| v7 | ~1.5% (1.8×ATR) | 0.11 / 1.5 = 0.073 R | 0.075 R |

So the levers, in order of leverage:

1. **Wider stops.** Doubling stop width halves cost in R. This is the biggest single
   lever and it is free.
2. **Maker instead of taker fills.** Bybit charges 0.020% maker vs 0.055% taker —
   PostOnly limit entries would cut the numerator ~2.75×. Not implemented, because
   modelling limit-order fill probability honestly is its own piece of work and an
   unmodelled assumption here would flatter the backtest.
3. **Selectivity.** Fewer, higher-conviction entries; fewer tolls paid.

**Untuned v7 got this exactly backwards.** Its ATR stops were ~3× tighter than the
baseline's flat 5%, which tripled cost per trade, and it fired on 36% of bars. Note
its gross expectancy is −0.007 R — its signals are, per trade, slightly *worse* than
the baseline's, not better. The similar gross dollar figure came from taking 3.7× as
many trades, not from better prediction. It is the one variant whose loss is
statistically unambiguous (t = −3.94).

`run_backtest.py --fee-sweep` re-scores the **identical signals** across cost
assumptions, and the result is the most useful table in this repo:

| scenario | trades | win rate | expectancy | 2-yr return |
|---|---|---|---|---|
| zero cost (signal only) | 817 | 51.3% | +0.022 R | **+13.9%** |
| maker 0.020% | 817 | 50.8% | +0.010 R | **+3.3%** |
| taker 0.055% (actual) | 816 | 50.0% | −0.009 R | **−11.4%** |
| taker + wide slippage | 814 | 49.0% | −0.021 R | **−20.0%** |

Same entries, same exits, same everything — a 34-point swing in return, entirely
from execution cost. And note the **win rate itself moves with cost**: 51.3% → 49.0%,
because each marginal winner that has to clear a bigger toll becomes a loser.

The practical reading: at maker fees your *existing, untouched* strategy is already
mildly profitable. Execution is a bigger lever here than prediction.

**Enabling shorts is roughly neutral on its own** (−0.009 R → −0.002 R). It doubles
the opportunity set, which is worth having, but it does not manufacture an edge that
was not there.

### Walk-forward result

`run_walkforward.py` searched 80 parameter sets across 8 rolling windows
(180 days in-sample → 60 days out-of-sample), fitting only on data the scoring
never saw.

| | in-sample | out-of-sample |
|---|---|---|
| Mean expectancy | +0.170 R | **+0.032 R** |
| Pooled expectancy | — | **+0.069 R** (368 trades) |
| Win rate | — | **50.5%** (break-even 46.9%) |
| Profit factor | — | 1.16 |
| Max drawdown | — | −14.9% |

Tuning moved v7 from **−0.082 R to +0.069 R out-of-sample**, with win rate 3.6 points
clear of its own break-even line — on data the search never touched, and better than
the −0.009 R baseline on every measure including drawdown.

The selected configuration is far more selective than the hand-picked defaults:
`|net| ≥ 0.50` (vs 0.18), block agreement `≥ 90%` (vs 55%), a 1.8×ATR stop against a
7.0×ATR target — a 3.9 R payoff — and session-restricted entries. Fewer, higher-
conviction trades that each pay the toll once and aim further: precisely the shape
the cost analysis above predicts should work.

**But read the two warnings, not just the headline.**

1. **The overfit gap is +0.105 R.** In-sample expectancy averaged +0.126 R and only
   +0.021 R survived out-of-sample, so roughly 80% of the apparent in-sample skill
   was noise the search fitted. That is normal and it is exactly why walk-forward
   exists — but it means the in-sample number is worthless as a forecast.

2. **+0.069 R is still not statistically significant.** The full figure is
   **+0.069 ± 0.060 R, t = +1.15** over 368 trades — a little over one standard error
   from zero. Confirming an edge this small would take roughly **1,100 trades, about
   3× the sample available**. Four of the eight windows were negative out-of-sample
   (−0.205, −0.140, −0.006, −0.132 R). Every report now prints expectancy with its
   standard error and t-statistic so this cannot be glossed over.

The honest summary: **a plausible small edge, not a proven one.** It is a hypothesis
worth paper-trading on Demo, not a system to trust with size.

The parameters chosen were also unstable across windows — `min_net` between 0.20 and
0.50, stop between 1.8× and 3.5× ATR — which is itself evidence that the surface is
mostly noise. `params_store.py` therefore saves the *consensus* across windows
(most-frequently-chosen value, ties broken toward the middle) together with a
per-knob **stability score**: the fraction of windows that agreed. Read those scores
before trusting any individual setting — `min_net` and `min_agreement` held at 62%
of windows, while the stop and target multipliers never settled at all.

The consensus deliberately snaps to a value some window actually selected. An earlier
version took a plain median, which over an even number of grid choices interpolated
*between* tested values (`atr_stop_mult = 2.65` from a grid of {1.8, 2.5, 3.5}) — a
configuration the search had never scored, about to be handed to the live bot. That
is pinned by `test_consensus_never_invents_an_untested_value`.

### The kill zones, finally measured

`config.KILL_ZONES` asserted that institutional activity concentrates in the
London/NY session windows, and penalised signals outside them. That is FX and
equities logic applied to a market that never closes, and it had never been tested.
v7 initially dropped it for that reason.

Measuring it against the baseline's 815 trades (`run_backtest.py --sessions`)
contradicted that call:

| | trades | win rate | expectancy |
|---|---|---|---|
| Inside kill zones | 294 | 52.4% | **+0.039 R** |
| Outside | 522 | 48.7% | **−0.036 R** |

A +0.075 R gap in the hypothesis's favour — but that is roughly one standard error,
and the hour-by-hour detail is plainly noise (09:00 London −0.251 R, 10:00 London
+0.237 R, about 30 trades each). Suggestive, not established.

So it became `V7Params.use_kill_zones`, defaulting **off** (matching what is actually
validated) and included in the walk-forward search space, rather than a judgement
call either way. `tests/test_path_consistency.py` checks it behaves identically in
the live and vectorised paths.

**The search then selected it in 62% of windows**, and adding it as a searchable
dimension lifted out-of-sample expectancy from +0.035 R to +0.069 R and win rate from
47.7% to 50.5%. So the original config was onto something the redesign nearly threw
away — which is the argument for measuring assumptions rather than reasoning about
whether they sound principled.

---

## Win rate, honestly

Win rate on its own is trivially gameable: widen the stop and tighten the target and
you get 90% winners with negative expectancy. Every report therefore prints win rate
**next to the break-even win rate its own payoff ratio implies**. The pair is the
information; either number alone invites the wrong conclusion.

The lever that raises win rate without wrecking expectancy is entry selectivity and
regime routing — not moving the target closer.

---

## Commands

```bash
python run_backtest.py --variants baseline,shorts,v7   # compare strategies
```

```bash
python run_walkforward.py --samples 80                 # out-of-sample tuning
```

```bash
python verify_demo.py                                  # prove the Demo path works
```

```bash
python -m pytest tests/ -q                             # 94 tests
```

`tests/test_path_consistency.py` is worth knowing about: it asserts on real market
data that the vectorised scoring path used by the tuner produces exactly the same
directions, stops and confidences as the bar-by-bar path used live. Without it the
tuner could quietly optimise a strategy that never trades.

Cached market data and computed signals live in `cache/`; delete it to force a
refetch. Signal caches are keyed by strategy variant, and a cache that does not cover
the requested date range is extended rather than silently reused.
