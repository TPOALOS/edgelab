"""Small statistics helpers (standard library only)."""
import math
from statistics import NormalDist

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def mean(x):
    return sum(x) / len(x)


def stdev(x):
    n = len(x)
    if n < 2:
        return 0.0
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))


def pp_sharpe(x):
    """Per-period Sharpe ratio (mean / sample stdev); 0.0 for a flat series."""
    sd = stdev(x)
    if sd < 1e-15:
        return 0.0
    return mean(x) / sd


def sharpe(x, periods_per_year):
    return pp_sharpe(x) * math.sqrt(periods_per_year)


def skew_kurt(x):
    """Population skewness and (non-excess) kurtosis."""
    n = len(x)
    m = mean(x)
    m2 = sum((v - m) ** 2 for v in x) / n
    if m2 < 1e-30:
        return 0.0, 3.0
    m3 = sum((v - m) ** 3 for v in x) / n
    m4 = sum((v - m) ** 4 for v in x) / n
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def expected_max_sharpe(var_sr, n_trials):
    """Expected maximum per-period Sharpe among n_trials unskilled trials (Bailey & Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sr) * (
        (1 - g) * _N.inv_cdf(1 - 1.0 / n_trials) + g * _N.inv_cdf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(x, trial_pp_sharpes, n_trials):
    """Probability that the true Sharpe exceeds what the best of n_trials unskilled trials would show.

    trial_pp_sharpes: per-period Sharpes of the trials recorded so far (their spread sets the luck level).
    With fewer than two recorded trials the spread falls back to the sampling variance of one Sharpe, 1/(T-1).
    Returns (dsr, sr0_per_period).
    """
    t = len(x)
    if t < 3:
        return 0.0, 0.0
    sr = pp_sharpe(x)
    if len(trial_pp_sharpes) >= 2:
        m = mean(trial_pp_sharpes)
        var = sum((s - m) ** 2 for s in trial_pp_sharpes) / (len(trial_pp_sharpes) - 1)
    else:
        var = 1.0 / (t - 1)
    sr0 = expected_max_sharpe(var, n_trials)
    skew, kurt = skew_kurt(x)
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr
    denom = math.sqrt(max(denom, 1e-12))
    z = (sr - sr0) * math.sqrt(t - 1) / denom
    return _N.cdf(z), sr0


def one_sided_p(x):
    """One-sided p-value for 'mean return > 0' using the normal approximation of the Sharpe ratio."""
    if len(x) < 3:
        return 1.0
    return 1.0 - _N.cdf(pp_sharpe(x) * math.sqrt(len(x)))
