"""Paths, settings and two small lookup helpers shared by the whole model package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # downloaded play-by-play (not in git)
RESULTS_DIR = ROOT / "results"    # saved simulation results (.json, .csv)

SEASONS = range(2018, 2026)
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz"
PBP_COLUMNS = ["game_id", "season", "season_type", "qtr", "half_seconds_remaining", "posteam", "down", "ydstogo",
               "yardline_100", "play_type", "yards_gained", "first_down", "touchdown", "epa",
               "field_goal_result", "kick_distance", "extra_point_result", "return_yards",
               "interception", "fumble_lost", "incomplete_pass", "fixed_drive", "fixed_drive_result"]

MAX_FG_DISTANCE = 68   # longest field goal allowed (NFL record)
FG_EXTRA_YARDS = 18    # kick distance = yards_to_endzone + 18 (end zone depth + snap position)
KICKOFF_START = 70     # after a score the other team starts 70 yards from the end zone
LATE_DRIVES = 3        # "late in the game" = 3 or fewer drives left (about the 4th quarter)

YARDS_TO_GO_EDGES = [0, 1, 2, 3, 5, 7, 10, 99]
YARDS_TO_GO_LABELS = ["1", "2", "3", "4-5", "6-7", "8-10", "11+"]


def yards_to_go_bin(yards_to_go):
    """Index into YARDS_TO_GO_LABELS."""
    return next(i for i in range(7) if yards_to_go <= YARDS_TO_GO_EDGES[i + 1])


def field_zone(yards_to_endzone):
    """10-yard zones: 0 = 1-10 yards from the end zone, ..., 9 = 91-99."""
    return min(9, (yards_to_endzone - 1) // 10)
