"""Searching for good strategies: the strategy-7 grid and successive halving.

Both run many strategies on the same paired games in parallel. Each worker process gets its settings once,
through the pool's initializer, and keeps them in the module-level dict below.
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from .candidates import policy_from_candidate, rule_count
from .game import run, scaled_games
from .stats import mean_ci
from .strategies import coach, make_policy, short_and_close

# Worker processes inherit this: one BLAS thread each, otherwise many workers run out of memory on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_JOB = {}


def _set_job(job):
    _JOB.update(job)


def short_close_grid(params, fg_range, max_yards_values, max_endzone_values, n_games=100_000, seed=12,
                     workers=None, opponent="equal"):
    """Strategy 7 (go if <= X yards to go and <= Y yards from the end zone) for every (X, Y), on the same games.
    Returns (heatmap: win probability per cell, tied: cells not significantly worse than the best cell)."""
    cells = [(x, y) for x in max_yards_values for y in max_endzone_values]
    job = dict(params=params, fg_range=fg_range, n_games=n_games, seed=seed, opponent=opponent)
    with mp.Pool(workers or mp.cpu_count(), initializer=_set_job, initargs=(job,)) as pool:
        results = dict(zip(cells, pool.map(_run_grid_cell, cells)))

    heatmap = pd.DataFrame(index=max_yards_values, columns=max_endzone_values, dtype=float)
    for (x, y), cell_results in results.items():
        heatmap.loc[x, y] = cell_results.mean()

    # a cell is "tied" with the best one if the paired difference is not significantly negative
    best_cell = max(results, key=lambda cell: results[cell].mean())
    tied = [[int(x), int(y)] for (x, y), cell_results in results.items()
            if mean_ci(100 * (cell_results.astype(float) - results[best_cell]))[2] >= 0]
    return heatmap, tied


def _run_grid_cell(cell):
    job = _JOB
    policy = make_policy([short_and_close(*cell)], job["fg_range"])
    results = run(job["params"], policy, job["opponent"], n_games=job["n_games"], seed=job["seed"])[0]
    return results.astype(np.float32)


def successive_halving(params, all_candidates, fg_range, rounds=4, n_games=1_000, seed=700, opponent="equal",
                       workers=None):
    """Play all candidates on n_games, keep the best quarter, play those on 4x more games, repeat.
    Candidates that make exactly the same decisions are merged, keeping the simplest one.

    Returns (history, game_counts, last_round):
      history      per round, {candidate index: gain over the coaches in pp}
      game_counts  games played per candidate in each round
      last_round   for the last round, {candidate index: p̂, gain and its 95% CI}"""
    workers = workers or mp.cpu_count()
    survivors = sorted(range(len(all_candidates)), key=lambda i: rule_count(all_candidates[i]))
    history, game_counts = [], []

    for round_number in range(rounds):
        round_seed = seed + round_number
        coach_results = run(params, coach, opponent, n_games=n_games, seed=round_seed)[0]
        outcomes = play_round(params, all_candidates, survivors, fg_range, opponent, n_games, round_seed, workers)
        diffs, last_round = score_round(survivors, outcomes, coach_results)

        history.append(diffs)
        game_counts.append(scaled_games(n_games))
        print(f"Round {round_number + 1}: {len(diffs)} distinct candidates on {scaled_games(n_games):,} games each")

        # keep the best quarter (at least 5) and give them 4x more games
        survivors = sorted(diffs, key=diffs.get, reverse=True)[:max(5, len(diffs) // 4)]
        n_games *= 4
    return history, game_counts, last_round


def play_round(params, all_candidates, survivors, fg_range, opponent, n_games, seed, workers):
    """Play every surviving candidate on the same games. Returns [(results, go_share)] in survivors order."""
    job = dict(params=params, candidates=all_candidates, fg_range=fg_range, opponent=opponent,
               n_games=n_games, seed=seed)
    with mp.Pool(workers, initializer=_set_job, initargs=(job,)) as pool:
        return pool.map(_run_candidate, survivors, chunksize=max(1, len(survivors) // (8 * workers)))


def _run_candidate(index):
    job = _JOB
    policy = policy_from_candidate(job["candidates"][index], job["fg_range"])
    return run(job["params"], policy, job["opponent"], n_games=job["n_games"], seed=job["seed"])


def score_round(survivors, outcomes, coach_results):
    """Gain over the coaches for each candidate, skipping candidates identical to an earlier (simpler) one.
    Returns ({index: gain in pp}, {index: p̂, gain, CI and go %})."""
    diffs, details, seen = {}, {}, set()
    for index, (results, go_share) in zip(survivors, outcomes):
        key = (results.tobytes(), round(go_share, 9))
        if key in seen:
            continue
        seen.add(key)

        diff, lower, upper = mean_ci(100 * (results - coach_results))
        diffs[index] = diff
        details[index] = {"p̂": results.mean(), "diff (pp)": diff, "diff lower": lower, "diff upper": upper,
                          "go %": 100 * go_share}
    return diffs, details
