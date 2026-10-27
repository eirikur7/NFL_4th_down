"""4th down model. Used by compute.ipynb (runs the simulations) and results.ipynb (shows the results).

  constants   paths, settings, lookup helpers
  data        load the play-by-play data
  estimate    probabilities estimated from the data (`params`)
  drive       one drive (punts and the Markov chain)
  game        whole games and Monte Carlo comparisons
  strategies  policies and go rules
  candidates  combinations of rules for the search
  search      strategy-7 grid and successive halving
  handbook    what a strategy does on every 4th down
  stats       confidence intervals
  results_io  save and load results

Everything is imported here, so the notebooks just write `model.run(...)`, `model.estimate(...)`, and so on.
"""
SPEED = 1.0        # scales the number of simulated games (1.0 ~ 30 min for compute.ipynb, 0.2 ~ 6 min)

from .constants import *                                                          # noqa: E402,F401,F403
from .data import load_data                                                       # noqa: E402,F401
from .estimate import estimate, fg_make_prob                                      # noqa: E402,F401
from .drive import MUFF_OFFSET, drive, punt, touchdown_points                    # noqa: E402,F401
from .game import compare, game, run, scaled_games                                # noqa: E402,F401
from .strategies import *                                                         # noqa: E402,F401,F403
from .candidates import (RULE_NAMES, candidates, label, policy_from_candidate,    # noqa: E402,F401
                         rule_count, rule_effects)
from .search import short_close_grid, successive_halving                          # noqa: E402,F401
from .handbook import go_thresholds, handbook                                     # noqa: E402,F401
from .stats import coach_vs_coach_test, mean_ci                                   # noqa: E402,F401
from .results_io import load_results, save_results                                # noqa: E402,F401
