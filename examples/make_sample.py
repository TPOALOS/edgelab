"""Write two sample CSVs: a pure-noise strategy and one with a real (synthetic) edge."""
import random

rng = random.Random(7)
N = 1500


def write(path, drift):
    with open(path, "w") as fh:
        fh.write("date,ret,turnover,bench\n")
        for i in range(N):
            bench = rng.gauss(0.0003, 0.01)
            ret = rng.gauss(drift, 0.01)
            fh.write("%d,%.6f,%.3f,%.6f\n" % (i, ret, 0.5, bench))


write("noise.csv", 0.0)
write("real_edge.csv", 0.0012)
