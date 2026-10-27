"""Whole games, and Monte Carlo comparisons of strategies."""
import random

import numpy as np
import pandas as pd

from .constants import KICKOFF_START
from .drive import drive
from .stats import mean_ci
from .strategies import coach


def game(params, policy, opponent, rng, decisions=None, drive_ends=None):
    """One game: `drives_per_team` drives each, our policy vs a real-coach opponent. Returns [our score, theirs].
    Our 4th-down calls are added to `decisions`, and how our drives ended to `drive_ends` (if given)."""
    n_drives = params["drives_per_team"]
    scores = [0, 0]                                   # team 0 = us, team 1 = the opponent
    drives_left = [n_drives, n_drives]
    policies = (policy, coach)
    play_outcomes = (params["play_outcomes"]["equal"], params["play_outcomes"][opponent])

    team = int(rng.random() < 0.5)                    # coin toss: who starts
    start = KICKOFF_START
    for _ in range(2 * n_drives):
        ours = team == 0
        points_for, points_against, start, ending = drive(
            params, start, policies[team], play_outcomes[team], scores[team] - scores[1 - team],
            drives_left[team], rng, decisions if ours else None)
        if ours and drive_ends is not None:
            drive_ends.append(ending)
        scores[team] += points_for
        scores[1 - team] += points_against
        drives_left[team] -= 1
        team = 1 - team
    return scores


def scaled_games(n_games):
    """The number of games to play: n_games times model.SPEED (but at least 500).
    SPEED is read here so that `model.SPEED = 0.2` in a notebook takes effect."""
    import model
    return max(500, int(n_games * model.SPEED))


def run(params, policy, opponent="equal", n_games=10_000, seed=0):
    """Play n_games. Game i always uses the same random numbers, so strategies can be compared game by game.
    Returns (result per game: 1 win, 0.5 tie, 0 loss;  share of our 4th downs where we went for it)."""
    n_games = scaled_games(n_games)
    results, decisions = np.empty(n_games), []
    for i in range(n_games):
        rng = random.Random(seed * 1_000_003 + i)
        ours, theirs = game(params, policy, opponent, rng, decisions)
        results[i] = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
    return results, decisions.count("go") / max(1, len(decisions))


def compare(params, policies, opponent="equal", n_games=30_000, seed=100):
    """Each policy vs real coaches on the same games: win probability and the paired difference in pp."""
    coach_results = run(params, coach, opponent, n_games, seed)[0]
    rows = []
    for name, policy in policies.items():
        results, go_share = run(params, policy, opponent, n_games, seed)
        win_prob, _, _ = mean_ci(results)
        diff, diff_lower, diff_upper = mean_ci(100 * (results - coach_results))
        rows.append({"strategy": name, "p̂": win_prob, "diff (pp)": diff, "diff lower": diff_lower,
                     "diff upper": diff_upper, "go %": 100 * go_share})
    return pd.DataFrame(rows).set_index("strategy")
