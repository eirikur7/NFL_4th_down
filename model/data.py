"""Download and cache the nflverse play-by-play data."""
from pathlib import Path

import pandas as pd

from .constants import DATA_DIR, PBP_COLUMNS, PBP_URL, SEASONS


def load_data(folder=DATA_DIR):
    """Regular-season play-by-play 2018-2025, only the columns we use.
    Reads data/pbp_<year>.csv.gz (downloads it if missing) and caches the result in data/pbp.pkl."""
    folder = Path(folder)
    cache = folder / "pbp.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    seasons = []
    for year in SEASONS:
        local_file = folder / f"pbp_{year}.csv.gz"
        source = local_file if local_file.exists() else PBP_URL.format(year)
        seasons.append(pd.read_csv(source, usecols=PBP_COLUMNS, low_memory=False))
    pbp = pd.concat(seasons, ignore_index=True)
    pbp = pbp[pbp.season_type == "REG"].reset_index(drop=True)

    # neutral = quarters 1-3, not the last 2 minutes of the first half
    pbp["neutral"] = (pbp.qtr <= 3) & ~((pbp.qtr == 2) & (pbp.half_seconds_remaining <= 120))

    folder.mkdir(exist_ok=True)
    pbp.to_pickle(cache)
    return pbp
