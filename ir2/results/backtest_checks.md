# Backtest correctness checks (main run)

Recomputed from the raw normalized text, re-phrased with each cutoff's phraser (independent of the document × term matrices the backtest uses). Script: `scripts/phase3_checks.py`.

## 1. No post-T text in training

Every training abstract of every cutoff has first and latest version before T: **PASS**.

## 2. Candidates are new

4,787 distinct top-50 terms (word2vec seed 0 and frequency, both candidate sets, all 28 scored pairs) were searched in every pre-T abstract with an indicator. Terms found there: **0** → **PASS**.

## 3. "Found" is real

### Hero pair: `learning_to_rank`, T = 2020, candidates `phrases` (word2vec top 20, seed 0)

For each term: post-T cs.IR abstracts containing both the term and an indicator. ✓ requires ≥ 3; the recorded year must be the year of the 3rd.

| rank | term | recorded | recount | first 3 abstracts (arXiv id, year) | check |
|:-:|---|:-:|:-:|---|:-:|
| 1 | `multi_armed_bandit` | ✗ | 2 | 2108.03440 (2021), 2311.04528 (2023) | PASS |
| 2 | `personalized_ranking` | ✓ 2024 | 3 | 2102.07279 (2021), 2107.13876 (2021), 2406.15983 (2024) | PASS |
| 3 | `matrix_factorisation` | ✗ | 0 | — | PASS |
| 4 | `fairness_aware` | ✗ | 1 | 2503.21092 (2025) | PASS |
| 5 | `root_mean_square_error` | ✗ | 0 | — | PASS |
| 6 | `directly_optimizes` | ✗ | 2 | 2011.00550 (2020), 2511.00530 (2025) | PASS |
| 7 | `latent_factor` | ✗ | 1 | 2206.05368 (2022) | PASS |
| 8 | `outperforms_state_of_the_art` | ✓ 2023 | 9 | 2207.11785 (2022), 2303.04712 (2023), 2304.11043 (2023) | PASS |
| 9 | `meta_learning` | ✗ | 1 | 2409.19548 (2024) | PASS |
| 10 | `negative_sampling` | ✓ 2024 | 7 | 2105.04769 (2021), 2307.14906 (2023), 2405.07920 (2024) | PASS |
| 11 | `recommendation_algorithms` | ✓ 2024 | 5 | 2005.12977 (2020), 2108.03440 (2021), 2405.00824 (2024) | PASS |
| 12 | `large_margin` | ✓ 2025 | 3 | 2206.00473 (2022), 2407.07181 (2024), 2507.02255 (2025) | PASS |
| 13 | `page_rank` | ✗ | 0 | — | PASS |
| 14 | `linear_regression` | ✗ | 1 | 2002.07651 (2020) | PASS |
| 15 | `bandit_feedback` | ✗ | 0 | — | PASS |
| 16 | `state_of_the_art_baselines` | ✓ 2026 | 3 | 2002.09841 (2020), 2511.00530 (2025), 2609.35041 (2026) | PASS |
| 17 | `rating_prediction` | ✗ | 0 | — | PASS |
| 18 | `ctr_prediction` | ✓ 2022 | 5 | 2007.06434 (2020), 2208.06164 (2022), 2210.10725 (2022) | PASS |
| 19 | `euclidean_distance` | ✗ | 0 | — | PASS |
| 20 | `link_prediction` | ✗ | 1 | 2409.16590 (2024) | PASS |

20/20 rows agree with the recount.

### Random sample: 60 top-50 terms (word2vec seed 0) from all scored pairs

18 recorded as found, 42 as not found. Agreement with the recount: **60/60** → **PASS**.

## 4. Random equals the base rate

Over 56 (pair, candidate set) rows, |random P@50 − base rate| is at most 0.0049 (mean 0.0013): **PASS** (threshold 0.01).

## 5. Seeds differ

word2vec P@50 is identical across the 3 seeds in 3/56 (pair, candidate set) rows; the mean std across seeds is 0.027. The seeds are applied, and the spread is small. Identical values in a few rows are expected: P@50 moves in steps of 0.02.
