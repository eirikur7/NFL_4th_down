# 4th down: go for it, kick, or punt?

Applied Probability (T-811, Reykjavík University)
Eiríkur Bernharðsson, Davíð Rúnar Matthíasson, Grétar Másson.

Which rule for when to go for it on 4th down wins the most games? We estimate probabilities from NFL play-by-play data
(2018–2025, nflverse), simulate whole games as a Markov chain, and compare strategies with Monte Carlo.

**Start with [`results.ipynb`](results.ipynb)**: all plots, tables, the handbook and the conclusions, read from saved results.

## Files

```
results.ipynb        plots, tables, handbook and conclusions (reads the saved results; runs in seconds)
compute.ipynb        works through the data and runs all simulations (~45 min); saves results/results.json
model/               the model, one file per step
  estimate.py          probabilities estimated from the data
  drive.py, game.py    the simulator: a drive (Markov chain), a whole game, Monte Carlo comparisons
  strategies.py        policies and go rules
  candidates.py, search.py   combinations of rules, strategy-7 grid, successive halving
  handbook.py          what a strategy does on every 4th down
  data.py, stats.py, results_io.py, constants.py   data loading, confidence intervals, saving, settings
plots.py             the plots used in results.ipynb
scripts/
  opponents_search.py   extension: the same search against a weak and a strong opponent (~1 h)
  deep_tradeoff.py      go for it vs punt deep in our own half, in net points (~30 s)
results/             saved simulation results
  results.json             main analysis (equal opponent), written by compute.ipynb
  results_opponents.json   weak / equal / strong side by side
  results_opp_weak.json, results_opp_strong.json   the searches against each opponent
  deep_tradeoff.csv        output of scripts/deep_tradeoff.py
data/                play-by-play downloaded from nflverse (not in git)
requirements.txt     Python packages
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook results.ipynb
```

## Method in short

1. **Probabilities from data:**
   - yards gained per play, sampled from real plays with the same down and yards to go (3rd and 4th down pooled)
   - P(1st down | yards to go)
   - P(field goal good | distance), by logistic regression
   - punts, drawn from real punts kicked within ±5 yards of the same spot
   - what real coaches do on 4th down: P(go | yards to go, field position)
2. **Simulator:** a drive is a Markov chain on (down, yards to go, field position). A game is 11 drives per team, and the number of drives left replaces the clock. The opponent always plays like the real coaches.
3. **Monte Carlo:** each game is a Bernoulli-type trial (win 1, tie ½, loss 0). All strategies play the same games with the same random numbers, so differences are paired. The 95% CIs come from the CLT. Coach vs coach gives 49.9% (p = 0.60), so the simulator has no built-in bias.
4. **Step 1:** each rule on its own.
   - the field-goal range (57 yards)
   - aggressiveness (random vs short yardage)
   - a heatmap of "go if ≤ X yards and at most Y yards from the end zone", on 100,000 paired games per cell
   - a dead-zone heatmap
   - the single strategies
5. **Step 2:** combinations.
   - The candidates are the top 5 of each heatmap plus the goal-line and late-game rules: 4,224 candidates.
   - They are searched by successive halving, 1,000 → 64,000 games, keeping the best quarter each round.
   - The top 5 are then re-tested on 50,000 fresh paired games (against the winner's curse).
6. **Extension:** the same search against a weak and a strong opponent, whose plays are drawn only from the bottom or top third of teams by EPA per play.

## Assumptions and choices

- **No clock or penalties;** the number of drives left stands in for the clock. No 2-point tries or overtime.
- **3rd and 4th downs are pooled** (about 34,900 3rd-down and 2,600 4th-down plays), because real 4th-down attempts are rare and biased: teams mostly go for it when it is short and looks good.
- **The opponent doesn't adapt to our strategy.** Opponent strength only changes its offense.
- **The simulated coaches use their quarter 1–3 behaviour all game,** so they don't react to the score late. As a check, we re-ran the main analysis with score-aware coaches, using their real 4th-quarter decisions per score state. The gain fell from 4.0 to 3.6 pp, and the handbook barely changed.
- **Punts:** the kick distance and return yards of the sampled real punt are both added to the punt. Punt-return touchdowns and muffed punts are included.
- **Dead zone:** the five best ranges go into the search. 25–55 (+0.61 pp) was replaced by the next range, 40–55 (+0.57 pp); the two are within noise.
- **Comparisons only:** the results compare strategies; they aren't exact win probabilities.
