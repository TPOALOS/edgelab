"""Command line interface."""
import argparse
import json
import math
import sys

from . import DISCLAIMER, __version__, core, ledger


def build_parser():
    ap = argparse.ArgumentParser(prog="edgelab", description="Is this trading edge real? A trial ledger plus an honest edge report.")
    ap.add_argument("--version", action="version", version="edgelab " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate a strategy CSV and record it in the trial ledger")
    r.add_argument("file", help="CSV with one row per period, in time order")
    r.add_argument("--name", required=True, help="strategy/variant name (a new name or new data counts as a new trial)")
    r.add_argument("--ret-col", default="ret", help="gross strategy return per period, as a fraction (default: ret)")
    r.add_argument("--turnover-col", help="fraction of capital traded each period; enables the cost tests")
    r.add_argument("--bench-col", help="buy-and-hold returns for the same periods")
    r.add_argument("--split", type=float, default=0.7, help="chronological dev fraction (default 0.7; the rest is the holdout)")
    r.add_argument("--cost-bps", type=float, default=5.0, help="assumed cost per unit of turnover in basis points (default 5)")
    r.add_argument("--periods-per-year", type=float, default=252.0, help="annualisation (default 252)")
    r.add_argument("--extra-trials", type=int, default=0, help="variants you tried but did not record here")
    r.add_argument("--ledger", default="edgelab.jsonl", help="ledger file (default: edgelab.jsonl)")
    r.add_argument("--no-log", action="store_true", help="evaluate without writing to the ledger")
    r.add_argument("--format", choices=["text", "json"], default="text")

    l = sub.add_parser("ledger", help="list the recorded trials")
    l.add_argument("--ledger", default="edgelab.jsonl")
    l.add_argument("--format", choices=["text", "json"], default="text")
    return ap


def _fmt_be(v):
    if v is None:
        return "n/a"
    return "never" if math.isinf(v) else "%.1f bps" % v


def render(rep):
    out = []
    out.append("%s: %d observations, %d dev / %d holdout (split %.0f%%), cost %.1f bps" % (
        rep["name"], rep["n"], rep["n_dev"], rep["n_holdout"], rep["split"] * 100, rep["cost_bps"]))
    out.append("")
    out.append("  Sharpe (annualised)        dev      holdout")
    out.append("    gross                  %6.2f    %6.2f" % (rep["dev_sharpe_gross"], rep["holdout_sharpe_gross"]))
    out.append("    net of costs           %6.2f    %6.2f" % (rep["dev_sharpe_net"], rep["holdout_sharpe_net"]))
    if rep["benchmark_holdout_sharpe"] is not None:
        out.append("    buy-and-hold                      %6.2f" % rep["benchmark_holdout_sharpe"])
    out.append("  Breakeven cost             %s" % _fmt_be(rep["breakeven_cost_bps"]))
    out.append("  Trials counted             %d  (%d recorded + %d extra), %d look(s) at this holdout" % (
        rep["trials_total"], rep["trials_tracked"], rep["trials_extra"], rep["holdout_looks"]))
    out.append("")
    for c in rep["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "skip"}[c["passed"]]
        out.append("  [%s] %s" % (mark, c["name"]))
        out.append("         %s" % c["detail"])
    out.append("")
    out.append("Verdict: %s" % rep["verdict"])
    if rep["verdict"] == "SUPPORTED":
        out.append("  The edge survived these checks. That is not proof; it only means these tests did not reject it.")
    else:
        out.append("  Not proven real. Fix the FAIL lines or gather more out-of-sample data before trusting it.")
    return "\n".join(out)


def cmd_run(a):
    series = core.load(a.file, a.ret_col, a.turnover_col, a.bench_col)
    sha = core.sha256_file(a.file)
    prior = ledger.read(a.ledger)
    rep = core.evaluate(series, name=a.name, data_sha=sha, ret_col=a.ret_col, turnover_col=a.turnover_col,
                        bench_col=a.bench_col, split=a.split, cost_bps=a.cost_bps, ppy=a.periods_per_year,
                        prior_entries=prior, extra_trials=a.extra_trials)
    entry = rep.pop("_entry")
    logged = False
    if rep["is_new_trial"] and not a.no_log:
        ledger.append(a.ledger, entry)
        logged = True
    rep["logged"] = logged
    if a.format == "json":
        print(json.dumps(rep, indent=2, default=lambda o: None))
    else:
        print(render(rep))
        print("")
        if logged:
            print("Recorded in %s as trial %d for this data." % (a.ledger, rep["trials_tracked"]))
        elif not rep["is_new_trial"]:
            print("Already in the ledger (same name, data, split and columns); trial count unchanged.")
        print(DISCLAIMER)
    return 0 if rep["verdict"] == "SUPPORTED" else 1


def cmd_ledger(a):
    entries = ledger.read(a.ledger)
    if a.format == "json":
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("Ledger %s is empty." % a.ledger)
        return 0
    by_data = {}
    for e in entries:
        by_data.setdefault(e["data_sha256"], []).append(e)
    for sha, items in by_data.items():
        print("data %s...  %d trial(s)" % (sha[:12], len(items)))
        for e in items:
            print("  %-24s dev %5.2f  holdout %5.2f  cost %4.1f bps  %s" % (
                e["name"][:24], e["dev_sharpe_net"], e["holdout_sharpe_net"], e["cost_bps"], e["time"]))
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return {"run": cmd_run, "ledger": cmd_ledger}[a.cmd](a)
    except (ValueError, OSError) as exc:
        print("edgelab: error: %s" % exc, file=sys.stderr)
        return 2
