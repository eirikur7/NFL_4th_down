"""Strategies: how a team decides on 4th down.

A policy decides one 4th down and returns "go", "fg" or "punt":
  policy(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng)

A go rule only says whether to go for it (True / False):
  rule(yards_to_go, yards_to_endzone, score_diff, drives_left)

`make_policy` combines go rules into a policy. Rule numbers (5, 6, ...) are used in the notebooks and plots.
"""
from .constants import LATE_DRIVES, field_zone, yards_to_go_bin


# Simple policies

def coach(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng):
    """Real coaches: go with the probability coaches go in this cell, otherwise kick like coaches."""
    cell = (yards_to_go_bin(yards_to_go), field_zone(yards_to_endzone))
    if rng.random() < params["p_coach_go"][cell]:
        return "go"
    return "fg" if rng.random() < params["p_coach_field_goal"][yards_to_endzone] else "punt"


def always(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng):
    return "go"


def kick(yards_to_endzone, fg_range):
    """Our kick: field goal if within fg_range yards of the end zone, otherwise punt."""
    return "fg" if yards_to_endzone <= fg_range else "punt"


def random_go(go_prob, fg_range):
    """Strategy 4: go with probability go_prob anywhere, otherwise kick."""
    def policy(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng):
        return "go" if rng.random() < go_prob else kick(yards_to_endzone, fg_range)
    return policy


# Go rules

def short_yardage(max_yards_to_go):
    """Rule 5: go if at most max_yards_to_go to go."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: yards_to_go <= max_yards_to_go


def past_midfield():
    """Rule 6: go in the opponent's half."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: yards_to_endzone < 50


def short_and_close(max_yards_to_go, max_yards_to_endzone):
    """Rule 7: go if at most max_yards_to_go to go and at most max_yards_to_endzone from the end zone."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: (
        yards_to_go <= max_yards_to_go and yards_to_endzone <= max_yards_to_endzone)


def dead_zone(near, far, max_yards_to_go=99):
    """Rule 8 ("no man's land"): go if near <= yards_to_endzone <= far and at most max_yards_to_go to go."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: (
        near <= yards_to_endzone <= far and yards_to_go <= max_yards_to_go)


def goal_line():
    """Rule 9: go inside the 5-yard line."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: yards_to_endzone <= 5


def trailing_late(max_drives_left):
    """Rule 10: go when behind with at most max_drives_left drives left."""
    return lambda yards_to_go, yards_to_endzone, score_diff, drives_left: (
        score_diff < 0 and drives_left <= max_drives_left)


# Rules that overrule the go rules

def fg_changes_scores(score_diff):
    """Would a field goal change how many scores separate the teams? (up 6-8 -> 9+, or down 9-11 -> within 8)"""
    return 6 <= score_diff <= 8 or -11 <= score_diff <= -9


def score_changing_fg_applies(max_drives_left, yards_to_endzone, fg_range, score_diff, drives_left):
    """Rule 12: with max_drives_left or fewer drives left and in field-goal range, kick if the field goal
    changes how many scores separate the teams."""
    return (max_drives_left is not None and drives_left <= max_drives_left and yards_to_endzone <= fg_range
            and fg_changes_scores(score_diff))


def leading_late_applies(leading_late, yards_to_endzone, score_diff, drives_left):
    """Rule 13: leading with at most N drives left and at least Z yards from the end zone (our own half
    for Z = 50), where leading_late = (N, Z)."""
    return (leading_late is not None and score_diff > 0 and drives_left <= leading_late[0]
            and yards_to_endzone >= leading_late[1])


def make_policy(go_rules=(), fg_range=37, score_aware_fg=False, fallback="kick", score_changing_fg=None,
                leading_late=None):
    """Go if ANY go rule says so. Otherwise kick (or do what coaches do, if fallback = "coach").

    score_aware_fg = rule 11: in field-goal range with LATE_DRIVES or fewer left, kick the field goal if it changes
      the number of scores, otherwise go.
    score_changing_fg = N, rule 12: with <= N drives left always kick the field goal when it changes the number
      of scores, overriding every go rule.
    leading_late = (N, Z), rule 13: when leading with <= N drives left at yards_to_endzone >= Z never go for it
      (kick instead), overriding everything: a failed attempt there hands the opponent a short field."""

    def first_call(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng):
        """The call before rule 13 is applied."""
        if score_changing_fg_applies(score_changing_fg, yards_to_endzone, fg_range, score_diff, drives_left):
            return "fg"
        if any(rule(yards_to_go, yards_to_endzone, score_diff, drives_left) for rule in go_rules):
            return "go"
        if score_aware_fg and yards_to_endzone <= fg_range and drives_left <= LATE_DRIVES:
            return "fg" if fg_changes_scores(score_diff) else "go"
        if fallback == "coach":
            return coach(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng)
        return kick(yards_to_endzone, fg_range)

    def policy(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng):
        call = first_call(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng)
        if call == "go" and leading_late_applies(leading_late, yards_to_endzone, score_diff, drives_left):
            return kick(yards_to_endzone, fg_range)
        return call
    return policy
