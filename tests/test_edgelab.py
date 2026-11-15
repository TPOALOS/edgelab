import contextlib
import io
import json
import math
import os
import random
import tempfile
import unittest

from edgelab import cli, core, ledger, stats


def write_csv(path, drift, n=1500, seed=1, turnover=0.5, header="ret,turnover,bench"):
    rng = random.Random(seed)
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for _ in range(n):
            fh.write("%.6f,%.3f,%.6f\n" % (rng.gauss(drift, 0.01), turnover, rng.gauss(0.0003, 0.01)))


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(args)
    return code, out.getvalue(), err.getvalue()


class StatsTests(unittest.TestCase):
    def test_sharpe_flat_series_is_zero(self):
        self.assertEqual(stats.pp_sharpe([0.01] * 10), 0.0)

    def test_expected_max_sharpe_grows_with_trials(self):
        a = stats.expected_max_sharpe(0.01, 10)
        b = stats.expected_max_sharpe(0.01, 1000)
        self.assertGreater(b, a)
        self.assertEqual(stats.expected_max_sharpe(0.01, 1), 0.0)

    def test_more_trials_lower_dsr(self):
        rng = random.Random(3)
        x = [rng.gauss(0.0006, 0.01) for _ in range(800)]
        d1, _ = stats.deflated_sharpe(x, [], 1)
        d100, _ = stats.deflated_sharpe(x, [], 100)
        self.assertGreater(d1, d100)

    def test_p_value_small_for_strong_drift(self):
        rng = random.Random(4)
        x = [rng.gauss(0.003, 0.01) for _ in range(500)]
        self.assertLess(stats.one_sided_p(x), 0.001)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.csv = os.path.join(self.dir.name, "s.csv")
        self.led = os.path.join(self.dir.name, "l.jsonl")

    def args(self, name="a", *extra):
        return ["run", self.csv, "--name", name, "--turnover-col", "turnover", "--bench-col", "bench",
                "--ledger", self.led] + list(extra)

    def test_noise_not_proven_edge_supported(self):
        write_csv(self.csv, 0.0)
        code, out, _ = run_cli(self.args())
        self.assertEqual(code, 1)
        self.assertIn("NOT PROVEN", out)
        write_csv(self.csv, 0.0012)
        code, out, _ = run_cli(self.args("b"))
        self.assertEqual(code, 0, out)
        self.assertIn("SUPPORTED", out)

    def test_rerun_same_trial_not_double_counted(self):
        write_csv(self.csv, 0.001)
        run_cli(self.args())
        run_cli(self.args())
        self.assertEqual(len(ledger.read(self.led)), 1)

    def test_new_names_accumulate_trials_and_looks(self):
        write_csv(self.csv, 0.001)
        for i in range(5):
            run_cli(self.args("v%d" % i))
        code, out, _ = run_cli(self.args("v5", "--format", "json"))
        rep = json.loads(out)
        self.assertEqual(rep["trials_tracked"], 6)
        self.assertEqual(rep["holdout_looks"], 6)
        self.assertEqual(len(ledger.read(self.led)), 6)

    def test_more_trials_never_improve_dsr(self):
        write_csv(self.csv, 0.0007)
        _, out1, _ = run_cli(self.args("only", "--no-log", "--format", "json"))
        _, out2, _ = run_cli(self.args("only", "--no-log", "--format", "json", "--extra-trials", "200"))
        self.assertGreater(json.loads(out1)["dsr"], json.loads(out2)["dsr"])

    def test_different_data_is_separate_pool(self):
        write_csv(self.csv, 0.001, seed=1)
        run_cli(self.args("a"))
        write_csv(self.csv, 0.001, seed=2)
        _, out, _ = run_cli(self.args("a", "--format", "json"))
        self.assertEqual(json.loads(out)["trials_tracked"], 1)

    def test_no_log_writes_nothing(self):
        write_csv(self.csv, 0.001)
        run_cli(self.args("a", "--no-log"))
        self.assertFalse(os.path.exists(self.led))

    def test_breakeven_cost_closed_form(self):
        write_csv(self.csv, 0.001, turnover=0.5)
        s = core.load(self.csv, "ret", "turnover")
        _, out, _ = run_cli(self.args("a", "--no-log", "--format", "json"))
        be = json.loads(out)["breakeven_cost_bps"]
        self.assertAlmostEqual(be, stats.mean(s["ret"]) / 0.5 * 1e4, places=6)

    def test_no_turnover_skips_cost_check(self):
        write_csv(self.csv, 0.001)
        _, out, _ = run_cli(["run", self.csv, "--name", "a", "--ledger", self.led, "--no-log", "--format", "json"])
        checks = {c["name"]: c["passed"] for c in json.loads(out)["checks"]}
        self.assertIsNone(checks["Edge survives 2x the assumed cost"])

    def test_errors(self):
        write_csv(self.csv, 0.001)
        self.assertEqual(run_cli(["run", self.csv, "--name", "a", "--ret-col", "nope", "--ledger", self.led])[0], 2)
        self.assertEqual(run_cli(["run", self.csv + ".missing", "--name", "a", "--ledger", self.led])[0], 2)
        self.assertEqual(run_cli(self.args("a", "--split", "0.99"))[0], 2)
        with open(self.csv, "w") as fh:
            fh.write("ret\n0.01\nabc\n")
        self.assertEqual(run_cli(["run", self.csv, "--name", "a", "--ledger", self.led])[0], 2)

    def test_too_short_series_rejected(self):
        write_csv(self.csv, 0.001, n=30)
        code, _, err = run_cli(self.args("a"))
        self.assertEqual(code, 2)
        self.assertIn("at least", err)

    def test_corrupt_ledger_reported(self):
        write_csv(self.csv, 0.001)
        with open(self.led, "w") as fh:
            fh.write("{not json\n")
        code, _, err = run_cli(self.args("a"))
        self.assertEqual(code, 2)
        self.assertIn("not valid JSON", err)

    def test_ledger_command(self):
        write_csv(self.csv, 0.001)
        run_cli(self.args("a"))
        code, out, _ = run_cli(["ledger", "--ledger", self.led])
        self.assertEqual(code, 0)
        self.assertIn("1 trial", out)
        self.assertEqual(run_cli(["ledger", "--ledger", self.led + "x"])[1].strip(), "Ledger %sx is empty." % self.led)

    def test_negative_turnover_rejected(self):
        with open(self.csv, "w") as fh:
            fh.write("ret,turnover\n" + "0.001,-0.5\n" * 60)
        code, _, err = run_cli(["run", self.csv, "--name", "a", "--turnover-col", "turnover", "--ledger", self.led])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
