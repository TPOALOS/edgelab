# Changelog

## 1.0.0

First release.

- `run`: evaluates a strategy CSV on a chronological dev/holdout split and records it in an append-only JSON-lines trial ledger.
- Counts every distinct variant tried on the same data (plus `--extra-trials` for ones you did not record) and deflates the dev Sharpe for them.
- Tracks how many strategies have looked at the same holdout and tightens the holdout significance threshold accordingly.
- Cost stress: net Sharpe at an assumed cost, closed-form breakeven cost, and a 2x cost check.
- Buy-and-hold comparison on the holdout.
- `ledger`: lists recorded trials grouped by data file. Exit code 0/1 for SUPPORTED/NOT PROVEN so it works in CI.
