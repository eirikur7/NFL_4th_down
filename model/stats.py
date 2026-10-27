"""Confidence intervals and the coach-vs-coach test."""
import math

from scipy.stats import norm


def mean_ci(values):
    """Mean and 95% confidence interval (central limit theorem). Returns (mean, lower, upper)."""
    mean, standard_error = values.mean(), values.std(ddof=1) / math.sqrt(len(values))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def coach_vs_coach_test(results):
    """Test H0: p = 0.5 for coach vs coach. Returns (p̂, lower, upper, p-value)."""
    win_prob, lower, upper = mean_ci(results)
    z = (win_prob - 0.5) / (results.std() / math.sqrt(len(results)))
    return win_prob, lower, upper, 2 * (1 - norm.cdf(abs(z)))
