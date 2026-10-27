"""Why (not) go for it deep in our own half?

    python scripts/deep_tradeoff.py     # about 30 s; saves results/deep_tradeoff.csv (shown in results.ipynb)

For a 4th down at our own spot we compare two choices, punt or go for it, by the net points of
(the rest of our drive + the opponent's next drive). We also count how often the opponent scores
after a punt and after a failed attempt.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model

N_SIMULATIONS = 20_000
OWN_YARD_LINES = [1, 5, 10, 20, 30]      # where the 4th down is, yards from our own goal line
YARDS_TO_GO = [1, 2, 4]
DRIVES_LEFT = 6                          # "plenty of drives left"

params = model.estimate(model.load_data())
fg_range = model.load_results("results.json")["fg_range"]
play_outcomes = params["play_outcomes"]["equal"]
later_policy = model.make_policy([model.short_yardage(4)], fg_range)    # what we do on later 4th downs


def opponent_drive(start, rng):
    """The opponent's next drive. Returns (points for the opponent, points for us (a safety))."""
    their_points, our_points, _, _ = model.drive(params, start, model.coach, play_outcomes, 0, DRIVES_LEFT, rng)
    return their_points, our_points


def attempt(yards_to_endzone, yards_to_go, rng):
    """Go for it on 4th down. Returns (what happened, yards to the end zone afterwards).
    What happened: "turnover", "touchdown", "safety", "converted" or "failed"."""
    gains, p_turnover = play_outcomes[(3, min(yards_to_go, 11))]
    if rng.random() < p_turnover:
        return "turnover", yards_to_endzone
    gain = gains[int(rng.random() * len(gains))]
    new_yards_to_endzone = yards_to_endzone - gain
    if new_yards_to_endzone <= 0:
        return "touchdown", new_yards_to_endzone
    if new_yards_to_endzone >= 100:
        return "safety", new_yards_to_endzone
    return ("converted" if gain >= yards_to_go else "failed"), new_yards_to_endzone


def simulate_punt(yards_to_endzone, rng):
    """Punt, then let the opponent drive. Returns the net points for us and how often the opponent scored."""
    net_points, touchdowns, scores = [], 0, 0
    for _ in range(N_SIMULATIONS):
        start = model.punt(params, yards_to_endzone, rng)
        their_points, our_points = opponent_drive(start, rng)
        touchdowns += their_points >= 6
        scores += their_points > 0
        net_points.append(our_points - their_points)
    return net_points, touchdowns, scores


def simulate_go(yards_to_endzone, yards_to_go, rng):
    """Go for it. Returns the net points for us, the number of conversions and failures, and how often
    the opponent scored (a touchdown, or any points) after a failure."""
    net_points, conversions, failures, touchdowns, scores = [], 0, 0, 0, 0
    for _ in range(N_SIMULATIONS):
        outcome, new_yards = attempt(yards_to_endzone, yards_to_go, rng)

        if outcome == "touchdown":
            conversions += 1
            net_points.append(model.touchdown_points(params, rng))
        elif outcome == "safety":
            net_points.append(-2)
        elif outcome == "converted":
            conversions += 1
            net_points.append(converted_attempt_points(new_yards, rng))
        else:                                              # a turnover or a failed attempt: they get the ball
            failures += 1
            their_start = max(1, 100 - yards_to_endzone) if outcome == "turnover" else 100 - new_yards
            their_points, our_points = opponent_drive(their_start, rng)
            touchdowns += their_points >= 6
            scores += their_points > 0
            net_points.append(our_points - their_points)
    return net_points, conversions, failures, touchdowns, scores


def converted_attempt_points(yards_to_endzone, rng):
    """After a first down: the rest of our drive, then the opponent's next drive. Net points for us."""
    our_points, their_points_now, their_start, _ = model.drive(
        params, yards_to_endzone, later_policy, play_outcomes, 0, DRIVES_LEFT, rng)
    their_points, our_safety_points = opponent_drive(their_start, rng)
    return our_points - their_points_now + our_safety_points - their_points


def main():
    rows = []
    for own_yard_line in OWN_YARD_LINES:
        yards_to_endzone = 100 - own_yard_line
        for yards_to_go in YARDS_TO_GO:
            if yards_to_go > yards_to_endzone:
                continue
            rng = random.Random(own_yard_line * 100 + yards_to_go)
            punt_net, punt_touchdowns, punt_scores = simulate_punt(yards_to_endzone, rng)
            go_net, conversions, failures, go_touchdowns, go_scores = simulate_go(yards_to_endzone, yards_to_go, rng)
            rows.append({"our own": own_yard_line, "4th &": yards_to_go, "P(convert)": conversions / N_SIMULATIONS,
                         "opp TD after failed go": go_touchdowns / max(failures, 1),
                         "opp TD after punt": punt_touchdowns / N_SIMULATIONS,
                         "opp score after failed go": go_scores / max(failures, 1),
                         "opp score after punt": punt_scores / N_SIMULATIONS,
                         "net pts: go": np.mean(go_net), "net pts: punt": np.mean(punt_net)})

    table = pd.DataFrame(rows)
    table["go − punt (pts)"] = table["net pts: go"] - table["net pts: punt"]
    print(table.round(3).to_string(index=False))
    model.RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(model.RESULTS_DIR / "deep_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
