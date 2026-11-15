"""Load a strategy CSV, split it, and decide whether the edge survives the checks."""
import csv
import hashlib
import math
import time

from . import ledger, stats

MIN_SEGMENT = 20  # minimum observations in each of dev and holdout


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _column(rows, header, name, path):
    if name not in header:
        raise ValueError("%s: column %r not found (columns: %s)" % (path, name, ", ".join(header)))
    out = []
    for i, row in enumerate(rows, 2):
        raw = (row.get(name) or "").strip().rstrip("%")
        try:
            v = float(raw)
        except ValueError:
            raise ValueError("%s line %d: %r is not a number in column %r" % (path, i, row.get(name), name))
        if not math.isfinite(v):
            raise ValueError("%s line %d: non-finite value in column %r" % (path, i, name))
        out.append(v)
    return out


def load(path, ret_col, turnover_col=None, bench_col=None):
    """Read returns (and optional turnover / benchmark) as float lists. Row order is time order."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError("%s: no data rows" % path)
    series = {"ret": _column(rows, header, ret_col, path)}
    if turnover_col:
        series["turnover"] = _column(rows, header, turnover_col, path)
        if any(v < 0 for v in series["turnover"]):
            raise ValueError("%s: turnover must be >= 0 (fraction of capital traded each period)" % path)
    if bench_col:
        series["bench"] = _column(rows, header, bench_col, path)
    return series


def net_returns(ret, turnover, cost_bps):
    if turnover is None:
        return list(ret)
    c = cost_bps / 1e4
    return [r - t * c for r, t in zip(ret, turnover)]


def split_index(n, split):
    if not 0.1 <= split <= 0.9:
        raise ValueError("--split must be between 0.1 and 0.9")
    k = int(n * split)
    if k < MIN_SEGMENT or n - k < MIN_SEGMENT:
        raise ValueError("need at least %d observations in both dev and holdout (got %d / %d)" % (MIN_SEGMENT, k, n - k))
    return k


def evaluate(series, *, name, data_sha, ret_col, turnover_col, bench_col, split, cost_bps, ppy,
             prior_entries, extra_trials=0):
    """Compute the report for one strategy. prior_entries: ledger entries (any data) before this run."""
    ret = series["ret"]
    turn = series.get("turnover")
    bench = series.get("bench")
    n = len(ret)
    k = split_index(n, split)

    net = net_returns(ret, turn, cost_bps)
    dev_net, hold_net = net[:k], net[k:]
    dev_gross, hold_gross = ret[:k], ret[k:]

    entry = {
        "name": name, "data_sha256": data_sha, "split": split, "ret_col": ret_col, "turnover_col": turnover_col,
        "n": n, "cost_bps": cost_bps, "ppy": ppy,
        "dev_pp_sharpe": stats.pp_sharpe(dev_net),
        "dev_sharpe_net": stats.sharpe(dev_net, ppy),
        "holdout_sharpe_net": stats.sharpe(hold_net, ppy),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    same_data = [e for e in prior_entries if e["data_sha256"] == data_sha and e["split"] == split]
    key = ledger.trial_key(entry)
    is_new = all(ledger.trial_key(e) != key for e in same_data)
    pool = same_data + ([entry] if is_new else [])
    tracked = len(pool)
    n_trials = tracked + max(0, extra_trials)
    looks = len({ledger.trial_key(e) for e in pool})

    dsr, sr0 = stats.deflated_sharpe(dev_net, [e["dev_pp_sharpe"] for e in pool], n_trials)
    p_hold = stats.one_sided_p(hold_net)
    alpha = 0.05 / max(looks, 1)

    mean_turn = stats.mean(turn) if turn is not None else None
    if turn is None:
        breakeven = None
    elif mean_turn <= 1e-15:
        breakeven = float("inf") if stats.mean(ret) > 0 else 0.0
    else:
        breakeven = max(0.0, stats.mean(ret) / mean_turn * 1e4)

    dev_s, hold_s = entry["dev_sharpe_net"], entry["holdout_sharpe_net"]
    checks = []

    checks.append(("Dev Sharpe survives the number of trials (deflated >= 95%)",
                   dsr >= 0.95, "DSR %.1f%% over %d trial(s), luck level Sharpe %.2f" % (
                       dsr * 100, n_trials, sr0 * math.sqrt(ppy))))
    checks.append(("Holdout is significant after %d look(s) at it" % looks,
                   hold_s > 0 and p_hold <= alpha, "one-sided p = %.4f, needs <= %.4f" % (p_hold, alpha)))
    if dev_s > 0:
        checks.append(("Holdout keeps >= 50% of the dev Sharpe",
                       hold_s >= 0.5 * dev_s, "holdout %.2f vs dev %.2f" % (hold_s, dev_s)))
    else:
        checks.append(("Holdout keeps >= 50% of the dev Sharpe", False, "dev Sharpe is not positive (%.2f)" % dev_s))
    if bench is not None:
        bh = stats.sharpe(bench[k:], ppy)
        checks.append(("Holdout beats buy-and-hold", hold_s > bh, "strategy %.2f vs benchmark %.2f" % (hold_s, bh)))
    else:
        bh = None
        checks.append(("Holdout beats buy-and-hold", None, "no benchmark column given"))
    if breakeven is None:
        checks.append(("Edge survives 2x the assumed cost", None, "no turnover column given, costs not tested"))
    else:
        ok = breakeven >= 2 * cost_bps
        be = "never (no turnover)" if math.isinf(breakeven) else "%.1f bps" % breakeven
        checks.append(("Edge survives 2x the assumed cost", ok, "breakeven cost %s vs assumed %.1f bps" % (be, cost_bps)))

    failed = [c for c in checks if c[1] is False]
    verdict = "SUPPORTED" if not failed else "NOT PROVEN"
    return {
        "name": name, "n": n, "n_dev": k, "n_holdout": n - k, "split": split, "cost_bps": cost_bps,
        "ppy": ppy,
        "dev_sharpe_gross": stats.sharpe(dev_gross, ppy), "dev_sharpe_net": dev_s,
        "holdout_sharpe_gross": stats.sharpe(hold_gross, ppy), "holdout_sharpe_net": hold_s,
        "benchmark_holdout_sharpe": bh,
        "breakeven_cost_bps": breakeven,
        "trials_tracked": tracked, "trials_extra": max(0, extra_trials), "trials_total": n_trials,
        "holdout_looks": looks, "dsr": dsr, "holdout_p": p_hold,
        "checks": [{"name": c[0], "passed": c[1], "detail": c[2]} for c in checks],
        "verdict": verdict, "is_new_trial": is_new, "_entry": entry,
    }
