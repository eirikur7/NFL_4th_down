"""Extension: the best strategy against a weak and a strong opponent, found the same way as for the equal opponent.

    python scripts/opponents_search.py                 # about 1 hour
    python scripts/opponents_search.py --skip-curves   # resume: step 1 is already in results.json

Step 1  Opponent curves: the gain of "go if <= X yards" over the coaches, against each opponent.
        Updates results/results.json.
Step 2  For the weak and the strong opponent: find the best strategy-7 cells and dead zones, search combinations
        with successive halving, then a final check on fresh games.
        Saves results/results_opp_<opponent>.json (resumable) and results/results_opponents.json.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model

OPPONENTS = ["weak", "equal", "strong"]
MAX_YARDS_CURVE = [0, 1, 2, 3, 4, 6, 8, 10, 15]      # X = 0 means "never go"
N_GAMES_CURVE = 40_000
DEAD_ZONE_NEAR = [25, 30, 35, 40, 45, 50]
DEAD_ZONE_FAR = [35, 40, 45, 50, 55, 60, 65]
DEAD_ZONE_MAX_YARDS = 10

# The estimated parameters and the equal-opponent results (workers only get params and fg_range, see init_worker).
shared = {}


def main():
    shared["params"] = model.estimate(model.load_data())
    shared["results"] = model.load_results("results.json")

    if "--skip-curves" not in sys.argv:
        opponent_curves()
    coach_win_rate = model.load_results("results.json")["coach_win_rate"]

    saved = {"equal": equal_opponent_summary(coach_win_rate)}
    for k, opponent in enumerate(["weak", "strong"]):
        saved[opponent] = search_or_load(opponent, seed_base=1700 + 1000 * k, coach_win_rate=coach_win_rate)

    flat = {f"{opponent}/{key}": value for opponent, values in saved.items() for key, value in values.items()}
    model.save_results(flat, "results_opponents.json")
    print("Saved results/results_opponents.json")


def play_in_parallel(jobs):
    """jobs = [(policy spec, opponent, n_games, seed)]. Returns the results of each job (win = 1, tie = 0.5)."""
    fg_range = shared["results"]["fg_range"]
    with Pool(2, initializer=init_worker, initargs=(shared["params"], fg_range)) as pool:
        return pool.map(play_job, jobs)


def init_worker(params, fg_range):
    shared.update(params=params, fg_range=fg_range)


def play_job(job):
    spec, opponent, n_games, seed = job
    policy = build_policy(spec, shared["fg_range"])
    return model.run(shared["params"], policy, opponent, n_games=n_games, seed=seed)[0].astype(np.float32)


def build_policy(spec, fg_range):
    """Policies are closures and cannot be sent to worker processes, so jobs carry a spec and workers build them:
    ("coach",), ("always",), ("short", X) or ("dead_zone", X, near, far)."""
    kind, *args = spec
    if kind == "coach":
        return model.coach
    if kind == "always":
        return model.always
    rules = [model.short_yardage(args[0])]
    if kind == "dead_zone":
        rules.append(model.dead_zone(args[1], args[2], DEAD_ZONE_MAX_YARDS))
    return model.make_policy(rules, fg_range)


def opponent_curves():
    """Step 1: gain of "go if <= X" over the coaches against each opponent, and of our best strategies."""
    params, results = shared["params"], shared["results"]
    fg_range = results["fg_range"]
    specs = {str(x): ("short", x) for x in MAX_YARDS_CURVE}
    specs["always"] = ("always",)
    specs["coach"] = ("coach",)

    names = [str(x) for x in MAX_YARDS_CURVE] + ["always"]
    gain, gain_ci_low, gain_ci_high = (pd.DataFrame(index=names, columns=OPPONENTS, dtype=float) for _ in range(3))
    coach_win_rate = {}

    for k, opponent in enumerate(OPPONENTS):
        jobs = [(specs[name], opponent, N_GAMES_CURVE, 1000 + k) for name in ["coach"] + names]
        coach_results, *strategy_results = [r.astype(float) for r in play_in_parallel(jobs)]
        coach_win_rate[opponent] = coach_results.mean()
        for name, strategy_result in zip(names, strategy_results):
            diff = model.mean_ci(100 * (strategy_result - coach_results))
            gain.loc[name, opponent], gain_ci_low.loc[name, opponent], gain_ci_high.loc[name, opponent] = diff

    best_max_yards = {opponent: gain[opponent].drop("always").idxmax() for opponent in OPPONENTS}

    # how much do our best single rule and our best combination gain against each opponent?
    best_single = model.make_policy([model.short_yardage(results["best_max_yards"])], fg_range)
    best_combination = model.policy_from_candidate(results["best"], fg_range)
    strategies = {f"best single (≤{results['best_max_yards']} yd)": best_single,
                  "best combination": best_combination}
    best_strategy_gain = pd.DataFrame({
        opponent: model.compare(params, strategies, opponent, n_games=30_000, seed=1100 + k)["diff (pp)"]
        for k, opponent in enumerate(OPPONENTS)})

    results.update(gain_by_opponent=gain, gain_ci_low=gain_ci_low, gain_ci_high=gain_ci_high,
                   coach_win_rate=coach_win_rate, best_max_yards_by_opponent=best_max_yards,
                   best_strategy_gain_by_opponent=best_strategy_gain)
    model.save_results(results, "results.json")
    print("Opponent curves:", {o: round(coach_win_rate[o], 3) for o in OPPONENTS}, best_max_yards, flush=True)


def equal_opponent_summary(coach_win_rate):
    """The equal opponent is the main analysis, so its results are copied from results.json."""
    results = shared["results"]
    return {"best": results["best"], "best_name": results["best_name"], "final": results["final"],
            "top_short_close_cells": results["top_short_close_cells"], "top_dead_zones": results["top_dead_zones"],
            "coaches": coach_win_rate["equal"]}


def search_or_load(opponent, seed_base, coach_win_rate):
    """Step 2 for one opponent, unless it finished before a restart."""
    filename = f"results_opp_{opponent}.json"
    if (model.RESULTS_DIR / filename).exists():
        print(opponent, "loaded from", filename, flush=True)
        return model.load_results(filename)

    found = search_against(opponent, seed_base)
    found["coaches"] = coach_win_rate[opponent]
    model.save_results(found, filename)
    return found


def search_against(opponent, seed_base):
    """The whole strategy search against one opponent. Returns everything worth saving."""
    params, results = shared["params"], shared["results"]
    fg_range = results["fg_range"]

    # strategy 7: the best (X, Y) cells; the best X anywhere on the field (Y = 100) is the best single rule
    heatmap, tied = model.short_close_grid(params, fg_range, list(range(1, 9)), list(range(10, 101, 10)),
                                           n_games=100_000, seed=12, opponent=opponent)
    top_cells = [[int(x), int(y)] for x, y in heatmap.stack().sort_values(ascending=False).index[:5]]
    best_max_yards = int(heatmap[100].idxmax())

    dead_zone_heatmap, dead_zone_baseline = dead_zone_grid(opponent, best_max_yards)
    top_dead_zones = [[int(a), int(b)] for a, b in dead_zone_heatmap.stack().sort_values(ascending=False).index[:5]]
    print(opponent, "top strategy 7:", top_cells, "top dead zone:", top_dead_zones, flush=True)

    # search combinations of the top cells and dead zones, then test the best few on fresh games
    all_candidates = model.candidates(top_cells, top_dead_zones)
    history, _, _ = model.successive_halving(params, all_candidates, fg_range, n_games=1_000, seed=seed_base,
                                             opponent=opponent)
    best, best_name, final = final_check(opponent, all_candidates, history[-1], best_max_yards, seed_base + 200)

    return {"best": best, "best_name": best_name, "final": final, "top_short_close_cells": top_cells,
            "top_dead_zones": top_dead_zones, "short_close_heatmap": heatmap, "short_close_tied": tied,
            "dead_zone_heatmap": dead_zone_heatmap, "dead_zone_baseline": dead_zone_baseline,
            "best_max_yards": best_max_yards,
            "handbook": model.handbook(params, model.policy_from_candidate(best, fg_range))}


def dead_zone_grid(opponent, best_max_yards):
    """Win probability of "go if <= best_max_yards, and also in the dead zone (near-far)" for every range.
    Returns (heatmap, win probability without the dead zone)."""
    ranges = [(near, far) for near in DEAD_ZONE_NEAR for far in DEAD_ZONE_FAR if near < far]
    specs = [("short", best_max_yards)] + [("dead_zone", best_max_yards, near, far) for near, far in ranges]
    outcomes = play_in_parallel([(spec, opponent, 10_000, 14) for spec in specs])

    heatmap = pd.DataFrame(index=DEAD_ZONE_NEAR, columns=DEAD_ZONE_FAR, dtype=float)
    for (near, far), range_results in zip(ranges, outcomes[1:]):
        heatmap.loc[near, far] = range_results.mean()
    return heatmap, outcomes[0].mean()


def final_check(opponent, all_candidates, last_round, best_max_yards, seed):
    """Test the 5 best candidates of the last round on fresh games, next to the coaches, the best single rule
    and the equal-opponent winner. Returns (best candidate, its name, the table of all strategies)."""
    params, results = shared["params"], shared["results"]
    fg_range = results["fg_range"]
    finalists = sorted(last_round, key=last_round.get, reverse=True)[:5]

    single_name = f"Best single rule: 5 (X = {best_max_yards})"
    policies = {"1. Real coaches": model.coach,
                single_name: model.make_policy([model.short_yardage(best_max_yards)], fg_range)}
    for i in finalists:
        policies[model.label(all_candidates[i])] = model.policy_from_candidate(all_candidates[i], fg_range)
    equal_winner_name = "Equal-opponent winner: " + results["best_name"]
    policies.setdefault(equal_winner_name, model.policy_from_candidate(results["best"], fg_range))

    final = model.compare(params, policies, opponent, n_games=50_000, seed=seed)
    print(final[["p̂", "diff (pp)", "diff lower", "diff upper"]].round(3).to_string(), flush=True)

    # the winner is the best of the searched combinations (not the coaches, single rule or equal-opponent winner)
    combinations = final.drop(["1. Real coaches", single_name])
    combinations = combinations[~combinations.index.str.startswith("Equal-opponent winner")]
    best_name = combinations["diff (pp)"].idxmax()
    best = all_candidates[next(i for i in finalists if model.label(all_candidates[i]) == best_name)]
    print(opponent, "best:", best_name, flush=True)
    return best, best_name, final


if __name__ == "__main__":
    main()
