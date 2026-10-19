# Leak-Fixed Re-Runs and the Random-Rejection Control: the Relation Beats Chance, and the Leak Was Biased Toward the Filter

> **Verified under a corrected fix (`results/leakfix2/`).** A code review found the first version
> of the fix incomplete in three ways: (1) widening fired only on an *empty* frontier, so a run could
> still end at the old 10 × budget iteration cap with budget unspent while PUCT kept re-selecting
> known-wrong answers over a reachable unexpanded node — 4 such runs in the first re-run; (2) widening
> could target a node at `max_depth`, whose candidates are all discarded for depth; (3)
> `random_reject` without widening could end a search at the root (not used here). Corrected: past
> the old cap every terminal selection widens (identical to the old code up to that point), the
> target skips `max_depth` nodes, and the runner refuses random rejection without widening. Both
> 1.5B settings were re-run in full under the corrected code: **all 328 run outcomes are identical**
> to the first version, and no unsolved run now ends with budget unspent (4 → 0). Every number below
> stands.
>
> One method caveat: the matched rate includes rejections made while widening (re-samples mostly
> duplicate existing siblings: Countdown-3 filter 0.568 in runs that never widened vs 0.639 in runs
> that did), and the random arm applies its rate everywhere, including the root's first expansion
> where the filter cannot reject. Both make the random control somewhat harsher than the filter,
> which favours "filter beats random".

## What changed in the harness

`search.py` gained three opt-in options (default off, so every recorded run stays reproducible):

- **`widen_on_starvation`.** A search *starves* when no unexpanded step node exists anywhere, so
  every selection reaches a terminal and the loop idles until its iteration cap with budget unspent.
  That state explained **41/41** short, unsolved runs in the published LLM shards (including 5/42
  baseline runs on 1.5B ProsQA). When set, a starved iteration re-samples K fresh candidates at the
  deepest non-answer node on the selected path and charges it as an expansion. It fires only on an
  empty frontier, so a run that never starves is unchanged (tested as a strict no-op in every arm).
- **`stop_on_solve`.** Ends a search once a verified-correct answer exists. Cannot change the solve
  outcome; makes `expansions_used` read as cost-to-solve.
- **`random_reject` arm.** Drops step candidates (answers exempt, as in the filter) with a hashed,
  deterministic draw at a fixed rate — set to the filter's *realized* rejection rate in the same
  leak-fixed run. If random rejection matches the filter, the filter's effect is forced depth
  (fewer branches, the budget concentrated); if the filter beats it, the relation is choosing *which*
  candidates to drop better than chance, i.e. it is deduplicating.

Runner: `--widen-on-starvation --stop-on-solve`, `--arms … random_reject --random-reject-rate R`.
Driver and analysis: `results/leakfix/driver.sh`, `results/leakfix/analyze.py`.

## 1.5B Countdown-3 (n=40) — the leak cost nothing here, and the filter is below baseline

Same configuration as `THIRD_MODEL_BUDGET_LEAK_FINDINGS.md`, with the fix. Filter's realized
rejection rate: **0.601**.

| arm | solved | expansions/problem | widenings/problem |
|---|---|---|---|
| baseline | 15/40 | 8.7 | 0.1 |
| pool | 10/40 | 8.6 | 3.1 |
| filter | 11/40 | 8.4 | 2.9 |
| random rejection @ 0.601 | **6/40** | 9.1 | 3.8 |

| comparison | Δ | 95% CI | discordant | p |
|---|---|---|---|---|
| pool − baseline | −0.125 | [−0.275, +0.025] | 8 vs 3 | 0.23 |
| filter − baseline | −0.100 | [−0.250, +0.050] | 7 vs 3 | 0.34 |
| **random − baseline** | **−0.225** | **[−0.350, −0.100]** | **9 vs 0** | **0.0039** |
| filter − pool | +0.025 | [−0.100, +0.151] | 4 vs 3 | 1.00 |
| filter − random | +0.125 | [+0.000, +0.250] | 6 vs 1 | 0.13 |

### 1. The fix changed no outcome, so the post-hoc compute-matching was wrong here

Compute is now matched (≈8.5 expansions in every arm; under 10 because solved runs stop), and the
contingency tables are **identical, problem for problem**, to the leaky run's. Per-arm seeding makes
each run's prefix identical until it starves, and the 123 (pool) and 116 (filter) widenings never
turned a starved run into a solve: 26 starved pool runs and 23 starved filter runs, **0 solved**,
each ending with its full budget spent (under the corrected fix; one filter run stopped at the
iteration cap under the first version, with the same outcome). The filter's starved runs kept re-sampling until they held
~28 answers each, all wrong.

`THIRD_MODEL_BUDGET_LEAK_FINDINGS.md` compute-matched this setting the other way — truncating the
baseline to the expansions each deduplicating arm had spent — and found the arms slightly *ahead*
(+0.050 / +0.075), concluding that "the third model's apparent harm was the leak". **Retracted for
this setting.** Truncation measures per-expansion efficiency up to the point an arm starves; it is
biased in the arm's favour whenever starvation means the relation has already merged away the
branches that held the answer, because then no amount of extra budget recovers them. That is what
happened here. Whether the same is true of the published settings needs these re-runs, not
truncation.

### 2. The filter is deduplicating, not just forcing depth

Rejecting 60% of step candidates *at random* costs 9 problems against the baseline and wins none
(p=0.004). Rejecting the same fraction *chosen by the relation* costs net 4. The relation recovers
more than half of random rejection's loss (filter − random +0.125, 6 vs 1). So the projection does
identify redundant candidates better than chance — but at τ=0.045 on this weak projection (training
false-merge rate 0.24) it rejects too much to beat plain search. This is the cube's filter floor
(`APPROXIMATE_RELATION_FINDINGS.md`) appearing in an LLM setting: a relation coarse enough that the
filter, and pooling with it, falls below doing nothing — here as a point estimate, not significant.

## 1.5B ProsQA (n=42) — the leak cost pooling four problems

Published configuration (`SECOND_MODEL_FINDINGS.md`) with the fix. Filter's realized rejection rate:
**0.552**.

| arm | solved (published) | solved, leak fixed | expansions/problem | widenings/problem |
|---|---|---|---|---|
| baseline | 11/42 | 11/42 | 13.3 | 0.3 |
| pool | 21/42 | **25/42** | 9.9 | 0.9 |
| filter | 20/42 | 20/42 | 10.3 | 1.0 |
| random rejection @ 0.552 | — | 16/42 | 11.9 | 2.1 |

| comparison | published | leak fixed | discordant | p |
|---|---|---|---|---|
| pool − baseline | +0.238 (p=0.031) | **+0.333** [+0.167, +0.500] | 2 vs 16 | **0.0013** |
| filter − baseline | +0.214 (p=0.023) | +0.214 [+0.048, +0.381] | 2 vs 11 | 0.0225 |
| random − baseline | — | +0.119 [−0.024, +0.262] | 2 vs 7 | 0.18 |
| filter − pool | −0.024 | **−0.119** [−0.262, +0.024] | 7 vs 2 | 0.18 |
| filter − random | — | +0.095 [−0.024, +0.214] | 1 vs 5 | 0.22 |

**The fix behaves exactly as designed.** Against the published run, problem by problem: baseline
42/42 identical (5 of its runs starved and were widened; none changed outcome), filter 42/42
identical, pool **+4 gained, 0 lost**. Pooling keeps the fewest nodes, starves first, and was the
arm the leak cost. With it fixed, pooling's gain over baseline rises to p=0.0013 and pooling now
leads the filter (not significant). So the published "filter and pool tie" on this setting was
partly the leak penalising pooling.

## Across both settings

**The relation beats random rejection.** Rejecting the filter's own fraction of step candidates at
random does worse than letting the relation choose: filter − random is +0.125 (Countdown-3) and
+0.095 (ProsQA); pooled over both settings the filter wins **11 discordant problems to 2
(p=0.022)**. The projection identifies redundant candidates better than chance, so the filter is
deduplicating, not merely pruning.

**Forced depth alone is domain-dependent.** Random rejection *helps* on ProsQA (+0.119, n.s.) and
*hurts* badly on Countdown-3 (−0.225, p=0.004). On ProsQA, about half the filter's gain over
baseline would come from dropping candidates at all; on Countdown-3, dropping candidates at all is
harmful and the relation only partly offsets it.

**Pooling vs filtering: no ordering, and the leak was biased toward the filter.** Leak-fixed,
filter − pool is +0.025 (Countdown-3) and −0.119 (ProsQA); pooled 6 vs 10 discordant (p=0.45).
The leak penalised the arm that starves first, which is pooling, so the published filter-over-pool
margins were tilted toward the filter. The one published filter win — 7B Countdown, +0.263 — is the
setting most exposed to that, and it is the one not yet re-run.

**Truncation is not a substitute for the fix.** Baseline truncation (the earlier compute-matching)
predicted ProsQA pool +0.357 / filter +0.286 and Countdown-3 pool +0.050 / filter +0.075; the fix
gives +0.333 / +0.214 and −0.125 / −0.100. It overstated the filter on both settings and got
Countdown-3's sign wrong.

## 7B Countdown (n=19, seeds 70/71/72) — the filter-over-pool win survives, but mostly as forced depth

Run after the corrected fix, with your other jobs throttled around it. Filter's realized rejection
rate: **0.628**.

| arm | solved (published) | solved, leak fixed | expansions/problem | widenings/problem |
|---|---|---|---|---|
| baseline | 5/19 | 5/19 | 14.7 | 0.0 |
| pool | 11/19 | 11/19 | 8.9 | 2.8 |
| filter | 16/19 | 16/19 | 7.1 | 0.7 |
| random rejection @ 0.628 | — | **13/19** | 9.1 | 1.1 |

| comparison | Δ | 95% CI | discordant | p |
|---|---|---|---|---|
| pool − baseline | +0.316 | [−0.053, +0.684] | 4 vs 10 | 0.18 |
| filter − baseline | +0.579 | [+0.263, +0.842] | 2 vs 13 | 0.0074 |
| **random − baseline** | **+0.421** | [+0.105, +0.737] | 2 vs 10 | **0.039** |
| filter − pool | +0.263 | [+0.105, +0.474] | 0 vs 5 | 0.0625 |
| filter − random | +0.158 | [−0.053, +0.368] | 1 vs 4 | 0.38 |

**The leak cost nothing here.** All three arms reproduce their published outcome on all 19 problems;
pooling widened 53 times without gaining a solve. The concern that this — the project's one
filter-over-pool win — was the setting most exposed to the pooling leak is refuted: the +0.263
stands.

**But most of the filter's gain is forced depth.** Rejecting the filter's own fraction of
candidates *at random* recovers +0.421 of the filter's +0.579 over baseline (~73%), significantly,
and solves more than pooling does (13 vs 11). On this setting, simply discarding candidates —
concentrating the budget on fewer branches — is worth more than pooling their statistics. The
relation's choice of *which* candidates to discard adds a further +0.158 (4 vs 1, n.s.).

Across all three settings now run, the filter beats random rejection **15 discordant problems to 3
(p=0.0075)**: the relation is consistently better than chance at choosing what to drop. How much
dropping *at all* is worth varies by setting: −0.225 (1.5B Countdown-3), +0.119 (1.5B ProsQA),
+0.421 (7B Countdown).

## 7B ProsQA (n=21) — unchanged by the fix; random rejection costs a quarter of the gain

Filter's realized rejection rate: **0.641**. All three arms reproduce their published outcome on all
21 problems.

| arm | solved | vs baseline |
|---|---|---|
| baseline | 5/21 | — |
| pool | 18/21 | +0.619 (p=0.0002) |
| filter | 19/21 | +0.667 (p=0.0001) |
| random rejection @ 0.641 | 14/21 | +0.429 (p=0.023) |

filter − pool +0.048 (2 vs 1, n.s.); filter − random **+0.238** (5 vs 0, p=0.063).

## All four settings, leak fixed

| setting | baseline | pool | filter | random @ filter's rate | filter − pool | filter − random (discordant) |
|---|---|---|---|---|---|---|
| 7B ProsQA (n=21) | 5 | 18 | 19 | 14 | +0.048 | 5 vs 0 |
| 7B Countdown (n=19) | 5 | 11 | 16 | 13 | +0.263 | 4 vs 1 |
| 1.5B ProsQA (n=42) | 11 | **25** | 20 | 16 | −0.119 | 5 vs 1 |
| 1.5B Countdown-3 (n=40) | 15 | 10 | 11 | 6 | +0.025 | 6 vs 1 |

1. **The leak changed one setting.** Every arm in every setting reproduces its published outcome
   problem for problem, except pooling on 1.5B ProsQA (+4 problems, 0 lost). The published
   conclusions survive the fix; the one filter-over-pool win (7B Countdown) stands.
2. **The relation beats chance in every setting**: filter over random rejection at the filter's own
   rate, **20 discordant problems to 3 (p=0.0005)**. The filter deduplicates; it does not merely prune.
3. **Dropping candidates at all is setting-dependent**: random rejection vs baseline is +0.429,
   +0.421, +0.119 and −0.225. Where it helps most (both 7B settings) it accounts for 64–73% of the
   filter's gain; on 1.5B Countdown-3 it is harmful and the filter falls below baseline.
4. **No ordering between pool and filter**: +0.048, +0.263, −0.119, +0.025 — one filter win, one
   (n.s.) pooling lead, two ties.
