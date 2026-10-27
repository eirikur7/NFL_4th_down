"""The handbook: what a strategy does on every 4th down."""
import random
from itertools import product

import numpy as np
import pandas as pd

from .constants import FG_EXTRA_YARDS, field_zone, yards_to_go_bin


def handbook(params, policy, max_yards_to_go=15):
    """For every 4th down (yards to go, yards to the end zone): the call the policy makes in a normal situation
    (tied game, plenty of drives left), next to the probabilities behind it and what coaches do."""
    rng = random.Random(0)
    rows = []
    for yards_to_go, yards_to_endzone in product(range(1, max_yards_to_go + 1), range(1, 100)):
        if yards_to_go > yards_to_endzone:         # impossible: the line to gain is past the goal line
            continue
        cell = (yards_to_go_bin(yards_to_go), field_zone(yards_to_endzone))
        call = policy(params, yards_to_go, yards_to_endzone, 0, params["drives_per_team"], rng)
        rows.append({"yards_to_go": yards_to_go, "yards_to_endzone": yards_to_endzone, "call": call,
                     "P(1st down)": params["p_first_down"]["mean"].get(yards_to_go, np.nan),
                     "FG distance": yards_to_endzone + FG_EXTRA_YARDS,
                     "P(FG good)": params["p_field_goal"][yards_to_endzone],
                     "coaches go %": 100 * params["p_coach_go"][cell],
                     "coaches usually": coaches_usually(params, cell, yards_to_endzone)})
    return pd.DataFrame(rows)


def coaches_usually(params, cell, yards_to_endzone):
    """The call real coaches make most often in this situation."""
    if params["p_coach_go"][cell] > 0.5:
        return "go"
    return "fg" if params["p_coach_field_goal"][yards_to_endzone] > 0.5 else "punt"


def go_thresholds(handbook_table):
    """The handbook in one table: per 10-yard zone, go for it with this many yards to go or fewer."""
    table = handbook_table.assign(zone=((handbook_table.yards_to_endzone - 1) // 10) * 10 + 1)
    go_limit = table[table.call == "go"].groupby("zone").yards_to_go.max()
    otherwise = table[table.call != "go"].groupby("zone").call.agg(lambda calls: "/".join(sorted(set(calls))))

    out = pd.DataFrame({"go if yards to go ≤": go_limit, "otherwise": otherwise})
    out["go if yards to go ≤"] = out["go if yards to go ≤"].map(lambda v: "never" if pd.isna(v) else int(v))
    out.index = [f"{zone}-{min(zone + 9, 99)}" for zone in out.index]
    out.index.name = "pos (yards to end zone)"
    return out
