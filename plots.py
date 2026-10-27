"""All plots for results.ipynb. Each function takes the estimated parameters `params` (model.estimate)
and/or the saved simulation `results` (model.load_results)."""
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import norm

import model

plt.rcParams.update({"figure.figsize": (9, 4.8), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY, DARK, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#13202e", "#9ec5ef"

CALLS = ["go", "fg", "punt"]
CALL_COLORS = [BLUE, ORANGE, "#d9d8d4"]
CALL_LABELS = ["Go for it", "Field goal", "Punt"]


# ---------------------------------------------------------------- Data

def yards_per_play(params):
    yards = np.array(params["play_outcomes"]["equal"][(1, 10)][0])
    values, counts = np.unique(np.clip(yards, -10, 40), return_counts=True)
    plt.bar(values, counts / counts.sum(), width=0.9, color=BLUE, label="Data")
    x = np.linspace(-10, 40, 300)
    plt.plot(x, norm.pdf(x, yards.mean(), yards.std()), color=ORANGE, lw=2, label="Normal, same mean and SD")
    plt.axvline(yards.mean(), color="black", ls="--", lw=1)
    plt.xlabel("Yards gained on 1st & 10 (40 = 40+)")
    plt.ylabel("Probability")
    plt.title(f"Yards per play: mean {yards.mean():.1f}, median {np.median(yards):.0f}")
    plt.legend()
    plt.show()


def field_goals(params):
    by_distance = params["fg_by_distance"]
    distances = np.arange(18, model.MAX_FG_DISTANCE + 1)
    intercept, slope = params["fg_logit"]
    plt.scatter(by_distance.index, by_distance.made / by_distance.n, s=np.sqrt(by_distance.n) * 2, color=ORANGE,
                alpha=0.6, label="Each distance (size = number of kicks)")
    plt.plot(distances, [model.fg_make_prob(d, intercept, slope) for d in distances], color=BLUE, lw=2.5,
             label=f"Logistic fit (β₀ = {intercept:.2f}, β₁ = {slope:.3f})")
    plt.xlim(17, 70)
    plt.ylim(0, 1.02)
    plt.xlabel("Kick distance (yards)")
    plt.ylabel("P(made)")
    plt.title(f"Field goals ({params['n_field_goals']:,} kicks)")
    plt.legend(loc="lower left")
    plt.show()


def coach_go_rates(params):
    rates = params["coach_go_table"].values
    fig, ax = plt.subplots(figsize=(9, 3.8))
    image = ax.imshow(rates, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for i, j in product(range(7), range(10)):
        if not np.isnan(rates[i, j]):
            ax.text(j, i, f"{rates[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if rates[i, j] > 55 else "black")
    ax.set_xticks(range(10), [f"{10 * j + 1}-{10 * j + 10}" for j in range(10)])
    ax.set_yticks(range(7), model.YARDS_TO_GO_LABELS)
    ax.set(xlabel="pos (yards to opponent's end zone)", ylabel="Yards to go",
           title=f"Real coaches: % going for it ({params['n_fourth_downs']:,} decisions, "
                 f"{params['coach_go_rate']:.1%} overall)")
    ax.grid(False)
    fig.colorbar(image, label="% going for it")
    plt.show()


# ---------------------------------------------------------------- Simulator check

def validation(results):
    conversion, drive_ends = results["conversion"], results["drive_ends"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(conversion.index, conversion["mean"], "o-", color=BLUE, label="Data")
    axs[0].plot(conversion.index, conversion["simulated"], "^--", color=ORANGE, label="Simulator")
    axs[0].set(xlabel="Yards to go", ylabel="P(1st down)", ylim=(0, 0.8), title="Probability of a 1st down")
    axs[0].legend()
    y = np.arange(len(drive_ends))
    axs[1].barh(y - 0.2, drive_ends["Data"], 0.4, color=BLUE, label="Data")
    axs[1].barh(y + 0.2, drive_ends["Simulator"], 0.4, color=ORANGE, label="Simulator")
    axs[1].set_yticks(y, drive_ends.index)
    axs[1].invert_yaxis()
    axs[1].set(xlabel="Share of drives", title="How drives end")
    axs[1].legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- Step 1

def fg_range_sweep(results):
    sweep = results["fg_range_sweep"]
    plt.fill_between(sweep.index, sweep.ci_low, sweep.ci_high, color=BLUE, alpha=0.2)
    plt.plot(sweep.index, sweep.win_prob, "o-", color=BLUE)
    plt.xlabel("Longest field goal our team attempts, K (yards)")
    plt.ylabel("Win probability")
    plt.title("How far should we try field goals? (go if ≤ 4 yards, else kick)")
    plt.show()


def aggressiveness(results):
    for sweep, color, name in [(results["random_go_sweep"], ORANGE, "4. Go with probability q"),
                               (results["short_yardage_sweep"], BLUE, "5. Go if yards to go ≤ X")]:
        plt.fill_between(100 * sweep.go_share, sweep.ci_low, sweep.ci_high, color=color, alpha=0.2)
        plt.plot(100 * sweep.go_share, sweep.win_prob, "o-", color=color, label=name)
    for _, row in results["short_yardage_sweep"].iterrows():
        plt.annotate(f"X={row.parameter:.0f}", (100 * row.go_share, row.win_prob), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8)
    go_share, win_prob = results["coach_point"]
    plt.plot(100 * go_share, win_prob, "kD", ms=8)
    plt.annotate("Real coaches", (100 * go_share, win_prob), textcoords="offset points", xytext=(8, -12))
    plt.axhline(0.5, color=GRAY, ls=":")
    plt.xlabel("Share of 4th downs where the team goes for it (%)")
    plt.ylabel("Win probability")
    plt.title("Win probability vs aggressiveness (equal opponent)")
    plt.legend(loc="lower center")
    plt.show()


def draw_heatmap(ax, table, fmt, vmin, vmax, boxed=()):
    """Coloured table with a number in each cell; cells in `boxed` (row, column pairs) get a box."""
    values = table.values.astype(float)
    image = ax.imshow(values, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    for i, j in product(range(values.shape[0]), range(values.shape[1])):
        if np.isnan(values[i, j]):
            continue
        is_boxed = any(str(table.index[i]) == str(row) and str(table.columns[j]) == str(column)
                       for row, column in boxed)
        ax.text(j, i, fmt(values[i, j]), ha="center", va="center", fontsize=8,
                fontweight="bold" if is_boxed else "normal")
        if is_boxed:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.5))
    ax.set_xticks(range(values.shape[1]), table.columns)
    ax.set_yticks(range(values.shape[0]), table.index)
    ax.grid(False)
    return image


def short_close_heatmap(results):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    image = draw_heatmap(ax, results["short_close_heatmap"], lambda v: f"{v:.4f}", 0.44, 0.56,
                         results.get("short_close_tied", results["top_short_close_cells"]))
    ax.set(xlabel="Y: go only if pos ≤ Y (100 = anywhere)", ylabel="X: max yards to go",
           title="Strategy 7 (boxed: not significantly worse than the best cell)")
    fig.colorbar(image, label="Win probability")
    plt.show()


def dead_zone(params, results):
    fg_range = results["fg_range"]
    near, far = results["top_dead_zones"][0]
    rng = random.Random(3)
    spots = np.arange(20, 71)

    def opponent_start(spot):                  # opponent's start after a punt, as yards from the punter's goal line
        start = model.punt(params, spot, rng)
        return 100 - (start - model.MUFF_OFFSET) if start > model.MUFF_OFFSET else max(start, 0)

    net_punt = [np.mean([spot - (100 - opponent_start(spot)) for _ in range(4000)]) for spot in spots]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    axs[0].plot(spots, [params["p_field_goal"][s] if s <= model.MAX_FG_DISTANCE - model.FG_EXTRA_YARDS else np.nan
                        for s in spots], color=BLUE, lw=2, label="P(field goal made)")
    axs[0].axvline(fg_range, color=BLUE, ls=":", lw=1)
    axs[0].text(fg_range - 0.5, 0.9, "our FG limit", color=BLUE, fontsize=8)
    axs[0].set(ylabel="P(field goal made)", ylim=(0, 1), xlabel="pos")
    axs[0].invert_xaxis()
    punt_axis = axs[0].twinx()
    punt_axis.plot(spots, net_punt, color=ORANGE, lw=2, label="Expected net punt (yards)")
    punt_axis.set(ylabel="Expected net punt (yards)", ylim=(0, 50))
    punt_axis.grid(False)
    axs[0].axvspan(near, far, color=GRAY, alpha=0.2)
    axs[0].text((near + far) / 2, 0.06, "best no man's land", ha="center", fontsize=9)
    lines = [line for line in axs[0].get_lines() + punt_axis.get_lines() if not line.get_label().startswith("_")]
    axs[0].legend(handles=lines, loc="lower left", fontsize=9)
    axs[0].set_title("Field goals get unlikely, punts gain little")
    gain = 100 * (results["dead_zone_heatmap"] - results["dead_zone_baseline"])
    image = draw_heatmap(axs[1], gain, lambda v: f"{v:+.2f}", -2, 2, results["top_dead_zones"])
    axs[1].set(xlabel="B (upper end of pos range)", ylabel="A (lower end)",
               title=f"Adding 'go if A ≤ pos ≤ B and ≤ 10 yards' to 'go if ≤ {results['best_max_yards']} yards'")
    fig.colorbar(image, ax=axs[1], label="pp vs short yardage alone")
    plt.tight_layout()
    plt.show()


def strategy_diffs(table, title):
    """Each strategy's change in win probability vs real coaches, with 95% CI."""
    table = table.sort_values("diff (pp)")
    y = np.arange(len(table))
    colors = [BLUE if lower > 0 else (ORANGE if upper < 0 else GRAY)
              for lower, upper in zip(table["diff lower"], table["diff upper"])]
    plt.figure(figsize=(9, 0.42 * len(table) + 1.2))
    plt.errorbar(table["diff (pp)"], y,
                 xerr=[table["diff (pp)"] - table["diff lower"], table["diff upper"] - table["diff (pp)"]],
                 fmt="none", ecolor=GRAY, capsize=3)
    plt.scatter(table["diff (pp)"], y, c=colors, s=45, zorder=3)
    plt.yticks(y, table.index)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Win probability vs real coaches (percentage points, 95% CI)")
    plt.title(title)
    plt.show()


# ---------------------------------------------------------------- Step 2

def halving(results):
    rounds = len(results["halving_diffs"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r, diffs in enumerate(results["halving_diffs"]):
        jitter = np.random.default_rng(r).uniform(-0.25, 0.25, len(diffs))
        ax.scatter(r + jitter, diffs, s=6, color=BLUE if r == rounds - 1 else LIGHT, alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(rounds), [f"round {r + 1}: {games:,} games"
                                  for r, games in enumerate(results["halving_game_counts"])])
    ax.set(ylabel="Win prob. vs coaches (pp)", title="Successive halving: each round keeps the top quarter")
    plt.show()


def effects(results):
    table = results["effects"]
    rules = list(dict.fromkeys(table.rule))
    y = np.arange(len(rules))
    plt.figure(figsize=(9, 4.3))
    fallbacks = [fb for fb in ["kick", "coach"] if (table.fallback == fb).any()]
    bar_height = 0.76 / len(fallbacks)
    for k, fallback in enumerate(fallbacks):
        rows = table[table.fallback == fallback].set_index("rule").loc[rules]
        plt.barh(y + (k - (len(fallbacks) - 1) / 2) * bar_height, rows["β"], bar_height, xerr=1.96 * rows["se"],
                 color=ORANGE if fallback == "kick" else BLUE, label=f"fallback: {fallback}",
                 error_kw=dict(ecolor=GRAY, capsize=2))
    plt.yticks(y, rules)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Average effect of adding the rule (pp)")
    plt.title("Regression: effect of each rule")
    plt.legend()
    plt.show()


def short_label(name):
    """Compact strategy name for plots."""
    if name.startswith("1."):
        return "Real coaches"
    if name.startswith("Best single"):
        return "Best single rule: go if ≤ " + name.split("X = ")[1].rstrip(")") + " yd"
    for old, new in [(" | else kick", ""), (" | else coach", ", else coach"),
                     (" yd inside own 30)", " inside own 30)"), ("(not inside own 30)", "(never inside own 30)"),
                     ("no man's land ", ""), (" (≤6 yd)", " ≤6 yd"), (" (≤10 yd)", " ≤10 yd"),
                     ("trailing (≤", "trailing late ("), ("score-changing FG (≤3 drives)", "score-changing FG"),
                     (" drives)", ")"), (" + ", " · ")]:
        name = name.replace(old, new)
    return name


def final_check(results, ax=None):
    """Final check with the chosen (refined) strategy highlighted."""
    table = results["final"].sort_values("diff (pp)")
    chosen = results["best_name"]
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.42 * len(table) + 1.2))
    for i, (name, row) in enumerate(table.iterrows()):
        color = ORANGE if name == chosen else (GRAY if name.startswith("1.") else BLUE)
        ax.errorbar(row["diff (pp)"], i, xerr=[[row["diff (pp)"] - row["diff lower"]],
                                               [row["diff upper"] - row["diff (pp)"]]],
                    fmt="none", ecolor=GRAY, capsize=3)
        ax.scatter(row["diff (pp)"], i, color=color, s=70 if name == chosen else 45, zorder=3)
    ax.set_yticks(range(len(table)), [short_label(name) for name in table.index])
    for tick_label, name in zip(ax.get_yticklabels(), table.index):
        if name == chosen:
            tick_label.set_color(ORANGE)
            tick_label.set_fontweight("bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Win probability vs real coaches (pp, 95% CI)")
    ax.set_title("Final check: top combinations vs real coaches (orange = chosen)")
    if show:
        plt.show()


# ---------------------------------------------------------------- Step 3

def opponents(results):
    gain = results["gain_by_opponent"]
    x = range(len(gain))
    for opponent, color, marker in zip(["weak", "equal", "strong"], [LIGHT, BLUE, DARK], "oDs"):
        if "gain_ci_low" in results:                                # 95% CI bands
            plt.fill_between(x, results["gain_ci_low"][opponent], results["gain_ci_high"][opponent],
                             color=color, alpha=0.18, lw=0)
        win_rate = f" (coaches win {100 * results['coach_win_rate'][opponent]:.0f}%)" if "coach_win_rate" in results else ""
        plt.plot(x, gain[opponent], marker=marker, color=color, label=f"Opponent: {opponent}{win_rate}")
    plt.xticks(x, ["never"] + [f"≤{v}" for v in gain.index[1:-1]] + ["always"])
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Go for it if yards to go ≤ X   (more aggressive →)")
    plt.ylabel("Gain over coaches, same opponent (pp)")
    plt.title("Gain in win probability over real coaches facing the same opponent (95% CI)")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------- Handbook

def _call_map(ax, handbook_table, column, max_yards_to_go):
    """Draw the call in `column` of a handbook table as a coloured map (yards to go vs pos)."""
    calls = handbook_table.pivot(index="yards_to_go", columns="yards_to_endzone", values=column)
    calls = calls.reindex(columns=range(1, 100))
    codes = calls.apply(lambda row: row.map({call: i for i, call in enumerate(CALLS)})).values.astype(float)
    # cell k spans (k-1, k]: "≤ 4" ends exactly at tick 4
    ax.imshow(codes, cmap=ListedColormap(CALL_COLORS), vmin=-0.5, vmax=2.5, aspect="auto", origin="lower",
              extent=(0, 99, 0, max_yards_to_go))
    ax.invert_xaxis()
    ax.set_xlabel("pos (yards to opponent's end zone)")
    ax.set_yticks(range(1, max_yards_to_go + 1))
    ax.set_xticks(range(0, 100, 10))
    ax.grid(False)


def _call_legend(fig, y_anchor):
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CALL_COLORS, CALL_LABELS)],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, y_anchor))


def handbook_pair(handbooks, coach_win_rates, max_yards_to_go=10):
    """Handbooks side by side, e.g. against a weak and a strong opponent (coaches' win rate in the titles)."""
    fig, axs = plt.subplots(1, len(handbooks), figsize=(13, 4.0), sharey=True)
    for ax, (opponent, table) in zip(axs, handbooks.items()):
        _call_map(ax, table[table.yards_to_go <= max_yards_to_go], "call", max_yards_to_go)
        ax.set_title(f"{opponent.capitalize()} opponent (coaches win {100 * coach_win_rates[opponent]:.0f}%)")
    axs[0].set_ylabel("Yards to go")
    _call_legend(fig, 1.07)
    plt.tight_layout()
    plt.show()


def handbook_map(handbook_table, max_yards_to_go=10):
    """Two maps side by side: the handbook's call vs what coaches usually do."""
    table = handbook_table[handbook_table.yards_to_go <= max_yards_to_go]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, column, title in [(axs[0], "call", "Handbook (best strategy)"),
                              (axs[1], "coaches usually", "Real coaches (most common call)")]:
        _call_map(ax, table, column, max_yards_to_go)
        ax.set_title(title)
    axs[0].set_ylabel("Yards to go")
    _call_legend(fig, 1.04)
    plt.tight_layout()
    plt.show()
