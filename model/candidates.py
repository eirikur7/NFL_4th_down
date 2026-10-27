"""Candidate strategies for the combination search (step 2).

A candidate is a dict that says which rules are on:
  yards_rule         None / ["5", X] / ["7", [X, Y]]        rule 5 or rule 7
  past_midfield      True / False                           rule 6
  dead_zone          None / [near, far, max_yards_to_go]    rule 8
  goal_line          True / False                           rule 9
  trailing_late      None / N                               rule 10
  score_aware_fg     True / False                           rule 11
  score_changing_fg  None / N                               rule 12
  leading_late       None / [N, Z]                          rule 13
  fallback           "kick" / "coach"                       what to do when no rule says go
"""
from itertools import product

import numpy as np
import pandas as pd

from .strategies import (dead_zone, goal_line, make_policy, past_midfield, short_and_close, short_yardage,
                         trailing_late)


def candidates(top_short_close_cells, top_dead_zones, dead_zone_max_yards=(6, 10)):
    """All combinations searched in step 2.
    The yards rule is one of the top strategy-7 cells (X, Y), where Y = 100 means plain "go if <= X yards".
    The dead zone is one of the top (near, far) ranges.

    Left out: rule 6 (it always lowered the win probability), rule 11 (its kicking half is rule 12, and
    its "go late in field-goal range" half only lowers the win probability) and the coach fallback.
    Every candidate kicks (field goal in range, else punt) when no rule says go."""
    yards_rules = [None] + [["7", [int(x), int(y)]] for x, y in top_short_close_cells]
    dead_zones = [None] + [[near, far, max_yards]
                           for near, far in top_dead_zones for max_yards in dead_zone_max_yards]
    options = product(yards_rules, dead_zones, [False, True], [None, 2, 3, 6], [None, 1, 2, 3], [None, [2, 50]])
    return [dict(yards_rule=yards_rule, past_midfield=False, dead_zone=dead_zone_, goal_line=goal_line_,
                 trailing_late=trailing, score_aware_fg=False, score_changing_fg=score_changing,
                 leading_late=leading, fallback="kick")
            for yards_rule, dead_zone_, goal_line_, trailing, score_changing, leading in options]


def policy_from_candidate(candidate, fg_range):
    """Turn a candidate dict into a policy."""
    rules = []
    yards_rule = candidate["yards_rule"]
    if yards_rule and yards_rule[0] == "5":
        rules.append(short_yardage(yards_rule[1]))
    if yards_rule and yards_rule[0] == "7":
        rules.append(short_and_close(*yards_rule[1]))
    if candidate["past_midfield"]:
        rules.append(past_midfield())
    if candidate["dead_zone"]:
        rules.append(dead_zone(*candidate["dead_zone"]))
    if candidate["goal_line"]:
        rules.append(goal_line())
    if candidate["trailing_late"]:
        rules.append(trailing_late(candidate["trailing_late"]))

    return make_policy(rules, fg_range, score_aware_fg=candidate["score_aware_fg"],
                       fallback=candidate["fallback"], score_changing_fg=candidate.get("score_changing_fg"),
                       leading_late=candidate.get("leading_late"))


def label(candidate):
    """Readable name, e.g. '≤4 yd + trailing (≤2 drives) | else kick'."""
    parts = []
    if candidate["yards_rule"]:
        parts.append(yards_rule_label(candidate["yards_rule"]))
    if candidate["past_midfield"]:
        parts.append("past midfield")
    if candidate["dead_zone"]:
        near, far, max_yards = candidate["dead_zone"]
        parts.append(f"no man's land {near}–{far} (≤{max_yards} yd)")
    if candidate["goal_line"]:
        parts.append("goal line")
    if candidate["trailing_late"]:
        parts.append(f"trailing (≤{candidate['trailing_late']} drives)")
    if candidate["score_aware_fg"]:
        parts.append("score-aware FG")
    if candidate.get("score_changing_fg"):
        parts.append(f"score-changing FG (≤{candidate['score_changing_fg']} drives)")
    if candidate.get("leading_late"):
        drives, spot = candidate["leading_late"]
        parts.append(f"leading late: no go at pos ≥{spot} (≤{drives} drives)")
    return (" + ".join(parts) or "no rules") + f" | else {candidate['fallback']}"


def yards_rule_label(yards_rule):
    """'≤5 yd' for rule 5 (or rule 7 with Y = 100), '≤6 yd & pos ≤70' for rule 7."""
    kind, value = yards_rule
    if kind == "5":
        return f"≤{value} yd"
    max_yards, max_spot = value
    if max_spot >= 100:
        return f"≤{max_yards} yd"
    return f"≤{max_yards} yd & pos ≤{max_spot}"


def rule_count(candidate):
    """Number of rules that are on."""
    return (bool(candidate["yards_rule"]) + candidate["past_midfield"] + bool(candidate["dead_zone"])
            + candidate["goal_line"] + bool(candidate["trailing_late"]) + candidate["score_aware_fg"]
            + bool(candidate.get("score_changing_fg")) + bool(candidate.get("leading_late")))


RULE_NAMES = {"5": "5. Short yardage", "7": "7. Short and close", "6": "6. Past midfield",
              "8": "8. No man's land", "9": "9. Goal line", "10": "10. Trailing late",
              "11": "11. Score-aware FG (late)", "12": "12. Score-changing FG (late)",
              "13": "13. Leading late: no go in own half"}
RULE_KEYS = np.array(["", "5", "7", "6", "8", "9", "10", "11", "12", "13"])     # "" = the intercept


def rule_indicators(candidate):
    """One regression row: 1 for the intercept, then 0/1 for each rule in RULE_KEYS order."""
    yards_kind = candidate["yards_rule"][0] if candidate["yards_rule"] else None
    return [1, yards_kind == "5", yards_kind == "7", candidate["past_midfield"], bool(candidate["dead_zone"]),
            candidate["goal_line"], bool(candidate["trailing_late"]), candidate["score_aware_fg"],
            bool(candidate.get("score_changing_fg")), bool(candidate.get("leading_late"))]


def rule_effects(all_candidates, diffs):
    """Regress each candidate's gain (pp) on 0/1 indicators of its rules, one regression per fallback.
    `diffs` maps a candidate index to its gain. Returns a table with β (the average effect of adding the rule)
    and its standard error, and the R² of each regression."""
    rows, r_squared = [], {}
    for fallback in dict.fromkeys(all_candidates[i]["fallback"] for i in diffs):
        indices = [i for i in diffs if all_candidates[i]["fallback"] == fallback]
        indicators = np.array([rule_indicators(all_candidates[i]) for i in indices], dtype=float)
        gains = np.array([diffs[i] for i in indices])

        # a rule that is always on or always off cannot be estimated
        used = [0] + [j for j in range(1, indicators.shape[1])
                      if indicators[:, j].min() != indicators[:, j].max()]
        indicators, keys = indicators[:, used], RULE_KEYS[used]

        beta, standard_errors, r_squared[fallback] = least_squares(indicators, gains)
        for k, key in enumerate(keys[1:], start=1):
            rows.append({"rule": RULE_NAMES[key], "fallback": fallback, "β": beta[k], "se": standard_errors[k]})
    return pd.DataFrame(rows), r_squared


def least_squares(indicators, gains):
    """Ordinary least squares. Returns (coefficients, their standard errors, R²)."""
    beta, *_ = np.linalg.lstsq(indicators, gains, rcond=None)
    residuals = gains - indicators @ beta
    variance = residuals @ residuals / (len(gains) - indicators.shape[1])
    standard_errors = np.sqrt(np.diag(variance * np.linalg.inv(indicators.T @ indicators)))
    r_squared = 1 - residuals @ residuals / np.sum((gains - gains.mean()) ** 2)
    return beta, standard_errors, r_squared
