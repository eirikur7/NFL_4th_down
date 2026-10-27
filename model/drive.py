"""One drive: punts and the Markov chain on (down, yards to go, spot).

Vocabulary: yards_to_endzone is the distance to the end zone the offense attacks (1 = goal line).
A drive returns (points for the offense, points for the defense, the other team's start, how the drive ended).
"""
from .constants import KICKOFF_START
from .estimate import MUFF_RECOVERED, MUFF_TOUCHDOWN, RETURN_TOUCHDOWN

# Special "start" values: punt() returns them, drive() handles them
PUNT_RETURN_TD = 0       # the receiving team returned the punt for a touchdown
MUFF_OFFSET = 100        # MUFF_OFFSET + k: the receiver lost the ball, the kicking team starts again k yards out
MUFF_TD = -1             # the kicking team recovered the muff in the end zone


def punt(params, yards_to_endzone, rng):
    """The opponent's start (yards to their end zone) after our punt from `yards_to_endzone`.
    Drawn from real punts near the spot. Can also return a special code for a return touchdown or a muff."""
    punts = params["punts_by_spot"][yards_to_endzone]
    kick_distance, return_yards, kind = punts[int(rng.random() * len(punts))]
    landing_spot = yards_to_endzone - kick_distance

    if kind == RETURN_TOUCHDOWN:
        return PUNT_RETURN_TD
    if kind == MUFF_TOUCHDOWN:
        return MUFF_TD
    if kind == MUFF_RECOVERED:
        return MUFF_OFFSET + min(99, max(1, landing_spot + return_yards))
    if landing_spot <= 0:
        return 80                                                     # touchback
    return min(99, max(1, 100 - landing_spot + return_yards))        # return yards are added to the punt


def drive(params, start, policy, play_outcomes, score_diff, drives_left, rng, decisions=None):
    """One drive from 1st & 10, `start` yards from the end zone. Every 4th-down call is added to `decisions`.
    `start` can be a punt code from punt(): a return touchdown, or a muff that gives the ball straight back."""
    if start <= PUNT_RETURN_TD or start > MUFF_OFFSET:
        return after_special_punt(params, start, rng)

    yards_to_endzone = start
    down, yards_to_go = 1, min(10, yards_to_endzone)
    while True:
        if down == 4:
            call = policy(params, yards_to_go, yards_to_endzone, score_diff, drives_left, rng)
            if decisions is not None:
                decisions.append(call)
            if call != "go":
                return kick_result(params, call, yards_to_endzone, rng)

        # one play: a turnover, or yards drawn from real plays
        gains, p_turnover = play_outcomes[(min(down, 3), min(yards_to_go, 11))]
        if rng.random() < p_turnover:
            return 0, 0, max(1, 100 - yards_to_endzone), "Turnover"
        gain = gains[int(rng.random() * len(gains))]
        yards_to_endzone -= gain

        if yards_to_endzone <= 0:
            return touchdown_points(params, rng), 0, KICKOFF_START, "Touchdown"
        if yards_to_endzone >= 100:
            return 0, 2, 60, "Safety"
        if gain >= yards_to_go:                                      # first down
            down, yards_to_go = 1, min(10, yards_to_endzone)
        elif down == 4:
            return 0, 0, 100 - yards_to_endzone, "Turnover on downs"
        else:
            down, yards_to_go = down + 1, yards_to_go - gain


def touchdown_points(params, rng):
    """6 points plus the extra point if it is good."""
    return 6 + (rng.random() < params["p_extra_point"])


def after_special_punt(params, start, rng):
    """The drive that follows a punt-return touchdown or a muff is decided at once, no plays are run."""
    if start == PUNT_RETURN_TD:
        return touchdown_points(params, rng), 0, KICKOFF_START, "Punt return TD"
    if start == MUFF_TD:
        return 0, touchdown_points(params, rng), KICKOFF_START, "Muffed punt"
    return 0, 0, start - MUFF_OFFSET, "Muffed punt"


def kick_result(params, call, yards_to_endzone, rng):
    """A field goal attempt or a punt (a 4th down where we do not go for it)."""
    if call == "fg":
        if rng.random() < params["p_field_goal"][yards_to_endzone]:
            return 3, 0, KICKOFF_START, "Field goal"
        return 0, 0, min(93 - yards_to_endzone, 80), "Missed field goal"
    return 0, 0, punt(params, yards_to_endzone, rng), "Punt"
