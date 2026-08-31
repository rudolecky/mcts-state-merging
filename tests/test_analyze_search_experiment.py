import numpy as np

from mcts_phase0.analyze_search_experiment import bootstrap_accuracy_delta_ci, mcnemar_exact


def test_mcnemar_exact_matches_known_reference():
    # classic textbook example: b=10, c=2 discordant pairs -> exact two-sided
    # p-value from the binomial(n=12, p=0.5) distribution is a well-known value
    p = mcnemar_exact(10, 2)
    assert abs(p - 0.0386) < 0.001


def test_mcnemar_exact_symmetric_in_b_and_c():
    assert mcnemar_exact(10, 2) == mcnemar_exact(2, 10)


def test_mcnemar_exact_one_when_no_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_small_p_for_lopsided_split():
    assert mcnemar_exact(15, 0) < 0.001


def test_bootstrap_delta_ci_recovers_known_positive_effect():
    rng = np.random.default_rng(0)
    n = 40
    baseline = [False] * 10 + [True] * 30  # 75% baseline
    treatment = [True] * 38 + [False] * 2   # 95% treatment
    point, lo, hi = bootstrap_accuracy_delta_ci(baseline, treatment, n_boot=1000, seed=0)
    assert point > 0.15
    assert lo > 0  # CI should exclude zero for this lopsided a difference


def test_bootstrap_delta_ci_straddles_zero_for_identical_arms():
    baseline = [True, False] * 20
    treatment = [True, False] * 20
    point, lo, hi = bootstrap_accuracy_delta_ci(baseline, treatment, n_boot=1000, seed=0)
    assert point == 0.0
    assert lo <= 0 <= hi
