"""Estimate every probability the simulator needs from the play-by-play data.

`estimate(pbp)` returns one dict, `params`, with these entries:
  play_outcomes       opponent -> {(down, yards to go): (yards gained on real plays, P(turnover))}
  p_first_down        P(1st down | yards to go)
  fg_logit, p_field_goal, fg_by_distance, n_field_goals     field goals
  punts, punts_by_spot                                      punts
  p_extra_point, drives_per_team, drives_mean               extra points and drives per game
  p_coach_go, p_coach_field_goal, coach_go_table, ...       what real coaches do on 4th down
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import FG_EXTRA_YARDS, MAX_FG_DISTANCE, YARDS_TO_GO_EDGES


def estimate(pbp):
    plays = neutral_plays(pbp)

    params = {"plays": plays, "play_outcomes": {"equal": play_outcomes_table(plays)}}
    params["p_first_down"] = first_down_rates(plays)
    params.update(field_goal_model(pbp))
    params.update(punt_model(pbp))
    params.update(extra_point_and_drive_model(pbp))
    params.update(coach_model(pbp))

    # weak / strong opponents: same table, built from the bottom / top third of teams
    params["play_outcomes"].update(play_outcomes_by_strength(plays))
    return params


def neutral_plays(pbp):
    """Run and pass plays in neutral situations, with the columns the simulator groups by.
    3rd and 4th down are pooled (down_pooled = 3) and yards to go is capped at 11."""
    plays = pbp[pbp.neutral & pbp.play_type.isin(["run", "pass"]) & pbp.down.notna()].copy()
    plays["down_pooled"] = plays.down.clip(upper=3).astype(int)
    plays["yards_to_go_capped"] = plays.ydstogo.clip(upper=11).astype(int)
    plays["turnover"] = (plays.interception == 1) | (plays.fumble_lost == 1)
    return plays


def play_outcomes_table(plays):
    """(down, yards to go) -> (list of yards gained on real plays, P(turnover))."""
    return {(down, yards_to_go): (group.loc[~group.turnover, "yards_gained"].astype(int).tolist(),
                                  group.turnover.mean())
            for (down, yards_to_go), group in plays.groupby(["down_pooled", "yards_to_go_capped"])}


def play_outcomes_by_strength(plays):
    """Play outcomes using only the plays of the strongest / weakest third of teams (by EPA per play, per season)."""
    team_epa = plays.groupby(["season", "posteam"]).epa.mean()
    strength = team_epa.groupby(level=0).transform(
        lambda epa: pd.qcut(epa.rank(method="first"), 3, labels=["weak", "middle", "strong"]))
    plays = plays.join(strength.rename("strength"), on=["season", "posteam"])
    return {"strong": play_outcomes_table(plays[plays.strength == "strong"]),
            "weak": play_outcomes_table(plays[plays.strength == "weak"])}


def first_down_rates(plays):
    """P(1st down | yards to go), measured on 3rd and 4th downs. Columns: mean, count."""
    short_plays = plays[(plays.down_pooled == 3) & (plays.ydstogo <= 15)].copy()
    short_plays["converted"] = (short_plays.first_down == 1) | (short_plays.touchdown == 1)
    return short_plays.groupby("ydstogo").converted.agg(["mean", "count"])


# Field goals

def fg_make_prob(distance, intercept, slope):
    """P(field goal made) from the kick distance: logistic fit 1 / (1 + exp(-(intercept + slope * distance)))."""
    return 1 / (1 + math.exp(-(intercept + slope * distance)))


def fit_logistic(distance, made):
    """Logistic regression by maximum likelihood. Returns (intercept, slope)."""
    def negative_log_likelihood(coefs):
        z = coefs[0] + coefs[1] * distance
        return np.sum(np.logaddexp(0, z) - made * z)

    intercept, slope = minimize(negative_log_likelihood, [0, 0]).x
    return intercept, slope


def field_goal_model(pbp):
    field_goals = pbp[(pbp.play_type == "field_goal") & pbp.kick_distance.notna()]
    distance = field_goals.kick_distance.to_numpy(float)
    made = (field_goals.field_goal_result == "made").to_numpy(float)
    intercept, slope = fit_logistic(distance, made)

    # the simulator looks up the probability by yards_to_endzone (index 0 is unused)
    p_field_goal = [0.0]
    for yards_to_endzone in range(1, 100):
        kick_distance = yards_to_endzone + FG_EXTRA_YARDS
        in_range = kick_distance <= MAX_FG_DISTANCE
        p_field_goal.append(fg_make_prob(kick_distance, intercept, slope) if in_range else 0.0)

    # make rate per distance, only used for the plot
    by_distance = field_goals.groupby(field_goals.kick_distance.astype(int)).field_goal_result.agg(
        made=lambda results: (results == "made").sum(), n="size")

    return {"fg_logit": (intercept, slope), "fg_by_distance": by_distance, "n_field_goals": len(field_goals),
            "p_field_goal": p_field_goal}


# Punts

# How a punt ends (the "kind" column below)
NORMAL, RETURN_TOUCHDOWN, MUFF_RECOVERED, MUFF_TOUCHDOWN = 0, 1, 2, 3


def punt_model(pbp):
    """Punts are drawn from real punts. Punters aim differently by field position (pinning deep near midfield),
    so a punt from a spot is drawn only from real punts kicked within ±5 yards of that spot
    (the window is widened until it holds 200 punts)."""
    punt_plays = pbp[(pbp.play_type == "punt") & pbp.yardline_100.notna()]
    spot = punt_plays.yardline_100.to_numpy(int)
    kick_distance = punt_plays.kick_distance.fillna(0).to_numpy(int)
    return_yards = punt_plays.return_yards.fillna(0).to_numpy(int)
    kind = punt_kinds(punt_plays)

    punts_by_spot = [None]                                      # index 0 is unused
    for yards_to_endzone in range(1, 100):
        window = 5
        while np.sum(np.abs(spot - yards_to_endzone) <= window) < 200:
            window += 1
        nearby = np.abs(spot - yards_to_endzone) <= window
        punts_by_spot.append(list(zip(kick_distance[nearby], return_yards[nearby], kind[nearby])))

    return {"punts": list(zip(kick_distance, return_yards)), "punts_by_spot": punts_by_spot}


def punt_kinds(punt_plays):
    """How each punt ended: normal, return touchdown, muff recovered by the kicking team,
    or muff recovered in the end zone (touchdown for the kicking team)."""
    fumble_lost = (punt_plays.fumble_lost == 1).to_numpy()
    touchdown = (punt_plays.touchdown == 1).to_numpy()
    return np.where(fumble_lost,
                    np.where(touchdown, MUFF_TOUCHDOWN, MUFF_RECOVERED),
                    np.where(touchdown, RETURN_TOUCHDOWN, NORMAL))


# Extra points and drives per game

def extra_point_and_drive_model(pbp):
    p_extra_point = (pbp[pbp.play_type == "extra_point"].extra_point_result == "good").mean()

    drives = pbp.dropna(subset=["posteam", "fixed_drive"]).groupby(["game_id", "posteam"]).fixed_drive.nunique()
    return {"p_extra_point": p_extra_point, "drives_mean": drives.mean(),
            "drives_per_team": int(round(drives.mean()))}


# Real coaches

def coach_model(pbp):
    fourth_downs = pbp[pbp.neutral & (pbp.down == 4)
                       & pbp.play_type.isin(["run", "pass", "punt", "field_goal"])].copy()
    fourth_downs["go"] = fourth_downs.play_type.isin(["run", "pass"])
    fourth_downs["togo_bin"] = pd.cut(fourth_downs.ydstogo, YARDS_TO_GO_EDGES, labels=False)
    fourth_downs["zone"] = ((fourth_downs.yardline_100 - 1) // 10).clip(upper=9).astype(int)

    overall_go_rate = fourth_downs.go.mean()
    p_go = coach_go_rates(fourth_downs, overall_go_rate)
    return {"p_coach_go": p_go["dict"], "coach_go_table": p_go["table"],
            "coach_go_rate": overall_go_rate, "n_fourth_downs": len(fourth_downs),
            "p_coach_field_goal": coach_field_goal_shares(fourth_downs)}


def coach_go_rates(fourth_downs, overall_go_rate):
    """P(go | yards-to-go bin, 10-yard zone), shrunk towards the overall rate with 2 pseudo-observations."""
    counts = fourth_downs.groupby(["togo_bin", "zone"]).go.agg(["sum", "count"])
    shrunk = (counts["sum"] + 2 * overall_go_rate) / (counts["count"] + 2)

    as_dict = {(togo_bin, zone): overall_go_rate               # cells never seen get the overall rate
               for togo_bin in range(7) for zone in range(10)}
    as_dict.update(shrunk.to_dict())
    table = (100 * shrunk).unstack().reindex(index=range(7), columns=range(10))   # in %, for the plot
    return {"dict": as_dict, "table": table}


def coach_field_goal_shares(fourth_downs):
    """When coaches kick: the share of field goals (vs punts) at each spot, using kicks within ±2 yards."""
    kicks = fourth_downs[~fourth_downs.go]
    by_spot = kicks.groupby("yardline_100").play_type.agg(
        field_goals=lambda kinds: (kinds == "field_goal").sum(), n="size")

    shares = [0.0]                                              # index 0 is unused
    for yards_to_endzone in range(1, 100):
        nearby = by_spot.loc[yards_to_endzone - 2:yards_to_endzone + 2]
        shares.append(nearby.field_goals.sum() / nearby.n.sum() if nearby.n.sum() else 0.0)
    return shares
