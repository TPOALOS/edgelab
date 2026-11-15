# edgelab

[![tests](https://github.com/TPOALOS/edgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/TPOALOS/edgelab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

**Is this trading edge real?** Most backtest tools print a return. `edgelab` keeps a **ledger of every variant you try** and then refuses to
call the best one an edge until it survives four honest questions:

1. Is the dev Sharpe still significant **after deflating for the number of trials** you ran?
2. Is the holdout significant **after counting how many strategies have already looked at it**?
3. Does the holdout keep at least half of the dev Sharpe, and beat buy-and-hold?
4. Does the edge survive **twice the cost** you assumed?

Pure Python standard library: no pandas, no numpy, no install step. Works with returns from any backtester.

## Quick start

Your CSV needs one row per period, in time order, with the gross strategy return as a fraction. `turnover` (fraction of capital traded per
period) and `bench` (buy-and-hold returns) are optional but unlock the cost and benchmark checks.

```bash
python3 examples/make_sample.py                 # writes noise.csv and real_edge.csv
python3 -m edgelab run examples/noise.csv --name sma_cross_20_50 --turnover-col turnover --bench-col bench
```

```
sma_cross_20_50: 1500 observations, 1050 dev / 450 holdout (split 70%), cost 5.0 bps

  Sharpe (annualised)        dev      holdout
    gross                   -0.14      0.88
    net of costs            -0.53      0.47
    buy-and-hold                       -0.36
  Breakeven cost             2.0 bps
  Trials counted             1  (1 recorded + 0 extra), 1 look(s) at this holdout

  [FAIL] Dev Sharpe survives the number of trials (deflated >= 95%)
  [FAIL] Holdout is significant after 1 look(s) at it
  [FAIL] Holdout keeps >= 50% of the dev Sharpe
  [PASS] Holdout beats buy-and-hold
  [FAIL] Edge survives 2x the assumed cost

Verdict: NOT PROVEN
```

Every `run` appends to `edgelab.jsonl`. Try a second and third variant on the same file and the trial count, the luck level and the holdout
threshold all move against you, which is the point. Re-running the *same* name, data, split and columns does not count as a new trial.

## Commands

```bash
python3 -m edgelab run FILE --name NAME [--turnover-col C] [--bench-col C] [--split 0.7] [--cost-bps 5]
                        [--periods-per-year 252] [--extra-trials N] [--ledger edgelab.jsonl] [--no-log] [--format json]
python3 -m edgelab ledger [--ledger edgelab.jsonl] [--format json]
```

- `--extra-trials N`: variants you tried in a notebook and never recorded. Be honest here; this is the knob people forget.
- `--no-log`: evaluate without recording (a dry run still counts itself in the numbers it prints).
- Exit code `0` = SUPPORTED, `1` = NOT PROVEN, `2` = bad input, so it can gate a CI job.

## How it decides

| Check | Rule |
| --- | --- |
| Deflated dev Sharpe | Bailey & Lopez de Prado deflated Sharpe on the dev segment, with N = recorded trials on this data + extra trials, luck level from the spread of recorded trial Sharpes; needs >= 95% |
| Holdout significance | one-sided p-value of the holdout net Sharpe <= 0.05 / (number of distinct strategies that have looked at this holdout) |
| Holdout retention | holdout net Sharpe >= 50% of dev net Sharpe (dev must be positive) |
| Buy-and-hold | holdout net Sharpe above the benchmark's (skipped without `--bench-col`) |
| Cost stress | closed-form breakeven cost (mean return / mean turnover) >= 2x the assumed cost (skipped without `--turnover-col`) |

Net return = gross return - turnover x cost. A trial is identified by (name, data file hash, split, columns), so editing the data starts a new pool.

## Limits

- SUPPORTED means these tests did not reject the edge, not that it will keep working.
- The normal approximation for Sharpe p-values assumes roughly independent returns; autocorrelated returns make it optimistic.
- The ledger is a file you control. It cannot know about variants you never record; `--extra-trials` is on the honour system.
- Not financial advice.

## Tests

```bash
python3 -m unittest -v
```

MIT licensed.
