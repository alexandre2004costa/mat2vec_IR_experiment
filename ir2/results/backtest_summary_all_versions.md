# Backtest results: all methods (all-versions comparison run)

Precision@k = share of the top-k candidates that co-occur with a target indicator in ≥ 3 cs.IR abstracts from T to 2026. word2vec rows: mean over scored pairs, ± std across 3 seeds. "Frequency-matched" = expected P@50 if the same top-50 were replaced by random candidates of the same pre-T frequency decile; the last column is what a method adds beyond frequency.

### Candidates: `all` — 28 scored (target, T) pairs

| method | P@10 | P@20 | P@50 | P@50 frequency-matched | P@50 − freq-matched |
|---|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.240 ± 0.011 | 0.262 ± 0.009 | 0.269 ± 0.001 | 0.167 ± 0.001 | 0.102 ± 0.001 |
| word2vec input·input | 0.092 ± 0.009 | 0.102 ± 0.012 | 0.102 ± 0.003 | 0.073 ± 0.000 | 0.028 ± 0.002 |
| SPECTER2 abstract means | 0.496 | 0.479 | 0.442 | 0.173 | 0.269 |
| 2nd-order co-occurrence | 0.800 | 0.777 | 0.747 | 0.334 | 0.413 |
| frequency | 0.454 | 0.452 | 0.504 | 0.339 | 0.165 |
| random (1000 draws) | 0.129 | 0.130 | 0.130 | — | — |

Mean base rate (share of all candidates found): 0.131.

### Candidates: `phrases` — 28 scored (target, T) pairs

| method | P@10 | P@20 | P@50 | P@50 frequency-matched | P@50 − freq-matched |
|---|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.283 ± 0.022 | 0.253 ± 0.009 | 0.239 ± 0.005 | 0.114 ± 0.000 | 0.125 ± 0.005 |
| word2vec input·input | 0.173 ± 0.008 | 0.152 ± 0.002 | 0.142 ± 0.005 | 0.076 ± 0.001 | 0.067 ± 0.005 |
| SPECTER2 abstract means | 0.479 | 0.427 | 0.349 | 0.110 | 0.239 |
| 2nd-order co-occurrence | 0.575 | 0.530 | 0.454 | 0.180 | 0.274 |
| frequency | 0.375 | 0.330 | 0.271 | 0.198 | 0.072 |
| random (1000 draws) | 0.098 | 0.099 | 0.098 | — | — |

Mean base rate (share of all candidates found): 0.098.

## Status of every (target, T) pair

`pre_T_target_docs_ir`: pre-T cs.IR abstracts with an indicator (the validity rule uses this). `pre_T_target_docs`: the same over all training text (cs.IR ∪ cs.CL).

| target                         |    T | status                              |   pre_T_target_docs_ir |   pre_T_target_docs |   post_T_target_docs | target_tokens                                                                                                              |   n_candidates_all |   base_rate_all |   n_candidates_phrases |   base_rate_phrases |
|:-------------------------------|-----:|:------------------------------------|-----------------------:|--------------------:|---------------------:|:---------------------------------------------------------------------------------------------------------------------------|-------------------:|----------------:|-----------------------:|--------------------:|
| query_expansion                | 2016 | scored                              |                     65 |                  71 |                  438 | query_expansion relevance_feedback                                                                                         |               6639 |           0.161 |                    789 |               0.120 |
| recommender_systems            | 2016 | scored                              |                    377 |                 393 |                 8729 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               4189 |           0.547 |                    490 |               0.394 |
| learning_to_rank               | 2016 | fewer than 20 pre-T cs.IR abstracts |                     19 |                  20 |                  563 | learning_to_rank lambdamart                                                                                                |               7642 |           0.261 |                    882 |               0.178 |
| dense_retrieval                | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  824 | nan                                                                                                                        |               8452 |           0.368 |                    965 |               0.238 |
| conversational_search          | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  370 | nan                                                                                                                        |               8452 |           0.235 |                    965 |               0.146 |
| retrieval_augmented_generation | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                 2136 | nan                                                                                                                        |               8452 |           0.557 |                    965 |               0.366 |
| query_expansion                | 2017 | scored                              |                     79 |                  85 |                  424 | query_expansion query_reformulation relevance_feedback                                                                     |               7844 |           0.130 |                   1122 |               0.095 |
| recommender_systems            | 2017 | scored                              |                    520 |                 551 |                 8586 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               4728 |           0.478 |                    676 |               0.339 |
| learning_to_rank               | 2017 | scored                              |                     33 |                  36 |                  549 | learning_to_rank lambdamart                                                                                                |               8694 |           0.207 |                   1234 |               0.157 |
| dense_retrieval                | 2017 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  824 | nan                                                                                                                        |               9911 |           0.331 |                   1370 |               0.223 |
| conversational_search          | 2017 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                  369 | nan                                                                                                                        |               9860 |           0.203 |                   1364 |               0.124 |
| retrieval_augmented_generation | 2017 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                 2135 | nan                                                                                                                        |               9884 |           0.511 |                   1367 |               0.348 |
| query_expansion                | 2018 | scored                              |                     93 |                 102 |                  410 | query_expansion query_reformulation relevance_feedback                                                                     |               9443 |           0.100 |                   1635 |               0.078 |
| recommender_systems            | 2018 | scored                              |                    743 |                 791 |                 8363 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               5434 |           0.405 |                    923 |               0.301 |
| learning_to_rank               | 2018 | scored                              |                     49 |                  56 |                  533 | learning_to_rank lambdamart listwise                                                                                       |              10076 |           0.154 |                   1690 |               0.114 |
| dense_retrieval                | 2018 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   3 |                  824 | nan                                                                                                                        |              11642 |           0.282 |                   1927 |               0.196 |
| conversational_search          | 2018 | fewer than 20 pre-T cs.IR abstracts |                      4 |                   6 |                  366 | nan                                                                                                                        |              11485 |           0.161 |                   1907 |               0.102 |
| retrieval_augmented_generation | 2018 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                 2135 | nan                                                                                                                        |              11777 |           0.461 |                   1944 |               0.318 |
| query_expansion                | 2019 | scored                              |                    116 |                 128 |                  387 | query_expansion query_reformulation relevance_feedback                                                                     |              11470 |           0.070 |                   2304 |               0.050 |
| recommender_systems            | 2019 | scored                              |                   1083 |                1170 |                 8023 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               6410 |           0.315 |                   1245 |               0.243 |
| learning_to_rank               | 2019 | scored                              |                     86 |                  96 |                  496 | learning_to_rank lambdamart listwise                                                                                       |              11782 |           0.097 |                   2313 |               0.071 |
| dense_retrieval                | 2019 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   4 |                  824 | nan                                                                                                                        |              14056 |           0.239 |                   2721 |               0.166 |
| conversational_search          | 2019 | fewer than 20 pre-T cs.IR abstracts |                      7 |                  12 |                  363 | nan                                                                                                                        |              13699 |           0.123 |                   2672 |               0.076 |
| retrieval_augmented_generation | 2019 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   2 |                 2135 | nan                                                                                                                        |              14181 |           0.405 |                   2739 |               0.276 |
| query_expansion                | 2020 | scored                              |                    138 |                 153 |                  365 | query_expansion query_reformulation relevance_feedback                                                                     |              14195 |           0.049 |                   3231 |               0.034 |
| recommender_systems            | 2020 | scored                              |                   1561 |                1694 |                 7545 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               7790 |           0.230 |                   1700 |               0.191 |
| learning_to_rank               | 2020 | scored                              |                    123 |                 142 |                  459 | learning_to_rank lambdamart listwise                                                                                       |              14260 |           0.061 |                   3181 |               0.038 |
| dense_retrieval                | 2020 | fewer than 20 pre-T cs.IR abstracts |                      1 |                  14 |                  823 | dual_encoder                                                                                                               |              16812 |           0.188 |                   3706 |               0.132 |
| conversational_search          | 2020 | fewer than 20 pre-T cs.IR abstracts |                     17 |                  24 |                  353 | nan                                                                                                                        |              16457 |           0.086 |                   3669 |               0.051 |
| retrieval_augmented_generation | 2020 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   2 |                 2135 | nan                                                                                                                        |              17332 |           0.350 |                   3815 |               0.241 |
| query_expansion                | 2021 | scored                              |                    169 |                 193 |                  334 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              17478 |           0.031 |                   4300 |               0.023 |
| recommender_systems            | 2021 | scored                              |                   2161 |                2383 |                 6945 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               9426 |           0.152 |                   2138 |               0.129 |
| learning_to_rank               | 2021 | scored                              |                    180 |                 206 |                  402 | learning_to_rank lambdamart listwise                                                                                       |              17276 |           0.033 |                   4211 |               0.025 |
| dense_retrieval                | 2021 | fewer than 20 pre-T cs.IR abstracts |                     10 |                  39 |                  814 | dual_encoder                                                                                                               |              19971 |           0.136 |                   4792 |               0.096 |
| conversational_search          | 2021 | scored                              |                     50 |                  63 |                  320 | conversational_search clarifying_questions                                                                                 |              19420 |           0.047 |                   4710 |               0.026 |
| retrieval_augmented_generation | 2021 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   4 |                 2135 | rag                                                                                                                        |              21050 |           0.297 |                   5035 |               0.211 |
| query_expansion                | 2022 | scored                              |                    194 |                 221 |                  309 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              21114 |           0.022 |                   5536 |               0.015 |
| recommender_systems            | 2022 | scored                              |                   2906 |                3221 |                 6200 | recommender recommenders recommendation recommendations collaborative_filtering                                            |              11293 |           0.100 |                   2683 |               0.097 |
| learning_to_rank               | 2022 | scored                              |                    232 |                 263 |                  350 | learning_to_rank lambdamart listwise                                                                                       |              20605 |           0.020 |                   5345 |               0.016 |
| dense_retrieval                | 2022 | scored                              |                     68 |                 130 |                  756 | dense_retrieval dense_retriever dense_retrievers dual_encoder bi_encoder bi_encoders                                       |              22396 |           0.079 |                   5768 |               0.055 |
| conversational_search          | 2022 | scored                              |                     87 |                 112 |                  283 | conversational_search clarifying_questions                                                                                 |              22539 |           0.024 |                   5883 |               0.015 |
| retrieval_augmented_generation | 2022 | fewer than 20 pre-T cs.IR abstracts |                      5 |                  20 |                 2131 | rag                                                                                                                        |              24504 |           0.244 |                   6306 |               0.177 |
| query_expansion                | 2023 | scored                              |                    215 |                 250 |                  288 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              25001 |           0.016 |                   6742 |               0.011 |
| recommender_systems            | 2023 | scored                              |                   3755 |                4187 |                 5351 | recommender recommenders recommendation recommendations collaborative_filtering                                            |              13257 |           0.060 |                   3129 |               0.069 |
| learning_to_rank               | 2023 | scored                              |                    279 |                 312 |                  303 | learning_to_rank ranknet lambdamart listwise                                                                               |              24312 |           0.013 |                   6479 |               0.011 |
| dense_retrieval                | 2023 | scored                              |                    153 |                 274 |                  671 | dense_retrieval dense_retriever dense_retrievers dense_passage_retrieval dual_encoder dual_encoders bi_encoder bi_encoders |              24907 |           0.041 |                   6592 |               0.030 |
| conversational_search          | 2023 | scored                              |                    126 |                 161 |                  244 | conversational_search clarifying_questions                                                                                 |              26095 |           0.013 |                   7042 |               0.008 |
| retrieval_augmented_generation | 2023 | fewer than 20 pre-T cs.IR abstracts |                     16 |                  70 |                 2120 | retrieval_augmented rag                                                                                                    |              27511 |           0.188 |                   7368 |               0.145 |

## P@50 per pair, candidates `all` (word2vec: mean of 3 seeds)

| target | T | w2v_out | w2v_in | specter2 | cooc2 | frequency | random | base_rate |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| conversational_search | 2021 | 0.21 | 0.04 | 0.24 | 0.78 | 0.64 | 0.05 | 0.05 |
| conversational_search | 2022 | 0.09 | 0.00 | 0.08 | 0.64 | 0.48 | 0.02 | 0.02 |
| conversational_search | 2023 | 0.07 | 0.02 | 0.04 | 0.48 | 0.36 | 0.01 | 0.01 |
| dense_retrieval | 2022 | 0.43 | 0.09 | 0.42 | 0.86 | 0.74 | 0.08 | 0.08 |
| dense_retrieval | 2023 | 0.14 | 0.05 | 0.14 | 0.60 | 0.64 | 0.04 | 0.04 |
| learning_to_rank | 2017 | 0.53 | 0.11 | 0.82 | 0.96 | 0.66 | 0.21 | 0.21 |
| learning_to_rank | 2018 | 0.47 | 0.08 | 0.68 | 0.96 | 0.66 | 0.15 | 0.15 |
| learning_to_rank | 2019 | 0.27 | 0.07 | 0.60 | 0.88 | 0.44 | 0.10 | 0.10 |
| learning_to_rank | 2020 | 0.16 | 0.02 | 0.34 | 0.80 | 0.46 | 0.06 | 0.06 |
| learning_to_rank | 2021 | 0.13 | 0.02 | 0.28 | 0.56 | 0.30 | 0.03 | 0.03 |
| learning_to_rank | 2022 | 0.08 | 0.02 | 0.24 | 0.54 | 0.20 | 0.02 | 0.02 |
| learning_to_rank | 2023 | 0.17 | 0.02 | 0.12 | 0.44 | 0.24 | 0.01 | 0.01 |
| query_expansion | 2016 | 0.27 | 0.11 | 0.66 | 0.92 | 0.76 | 0.16 | 0.16 |
| query_expansion | 2017 | 0.31 | 0.09 | 0.56 | 0.94 | 0.72 | 0.13 | 0.13 |
| query_expansion | 2018 | 0.27 | 0.09 | 0.44 | 0.88 | 0.66 | 0.10 | 0.10 |
| query_expansion | 2019 | 0.20 | 0.05 | 0.30 | 0.80 | 0.58 | 0.07 | 0.07 |
| query_expansion | 2020 | 0.11 | 0.05 | 0.20 | 0.70 | 0.58 | 0.05 | 0.05 |
| query_expansion | 2021 | 0.15 | 0.09 | 0.16 | 0.56 | 0.36 | 0.03 | 0.03 |
| query_expansion | 2022 | 0.14 | 0.04 | 0.20 | 0.48 | 0.42 | 0.02 | 0.02 |
| query_expansion | 2023 | 0.11 | 0.02 | 0.16 | 0.48 | 0.36 | 0.02 | 0.02 |
| recommender_systems | 2016 | 0.63 | 0.45 | 1.00 | 1.00 | 0.82 | 0.55 | 0.55 |
| recommender_systems | 2017 | 0.65 | 0.38 | 0.98 | 1.00 | 0.78 | 0.48 | 0.48 |
| recommender_systems | 2018 | 0.53 | 0.35 | 0.96 | 0.96 | 0.66 | 0.40 | 0.41 |
| recommender_systems | 2019 | 0.37 | 0.25 | 0.82 | 0.88 | 0.52 | 0.32 | 0.31 |
| recommender_systems | 2020 | 0.33 | 0.19 | 0.72 | 0.80 | 0.36 | 0.23 | 0.23 |
| recommender_systems | 2021 | 0.33 | 0.13 | 0.54 | 0.72 | 0.34 | 0.16 | 0.15 |
| recommender_systems | 2022 | 0.19 | 0.05 | 0.44 | 0.70 | 0.22 | 0.10 | 0.10 |
| recommender_systems | 2023 | 0.16 | 0.00 | 0.24 | 0.60 | 0.14 | 0.06 | 0.06 |

## P@50 per pair, candidates `phrases` (word2vec: mean of 3 seeds)

| target | T | w2v_out | w2v_in | specter2 | cooc2 | frequency | random | base_rate |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| conversational_search | 2021 | 0.13 | 0.09 | 0.12 | 0.42 | 0.18 | 0.02 | 0.03 |
| conversational_search | 2022 | 0.05 | 0.01 | 0.04 | 0.34 | 0.16 | 0.01 | 0.01 |
| conversational_search | 2023 | 0.05 | 0.03 | 0.00 | 0.20 | 0.08 | 0.01 | 0.01 |
| dense_retrieval | 2022 | 0.31 | 0.11 | 0.36 | 0.62 | 0.38 | 0.05 | 0.06 |
| dense_retrieval | 2023 | 0.10 | 0.04 | 0.14 | 0.30 | 0.24 | 0.03 | 0.03 |
| learning_to_rank | 2017 | 0.42 | 0.25 | 0.66 | 0.70 | 0.46 | 0.16 | 0.16 |
| learning_to_rank | 2018 | 0.41 | 0.21 | 0.48 | 0.62 | 0.38 | 0.11 | 0.11 |
| learning_to_rank | 2019 | 0.33 | 0.12 | 0.34 | 0.44 | 0.28 | 0.07 | 0.07 |
| learning_to_rank | 2020 | 0.26 | 0.06 | 0.30 | 0.38 | 0.24 | 0.04 | 0.04 |
| learning_to_rank | 2021 | 0.16 | 0.09 | 0.18 | 0.24 | 0.22 | 0.02 | 0.02 |
| learning_to_rank | 2022 | 0.08 | 0.03 | 0.12 | 0.22 | 0.14 | 0.02 | 0.02 |
| learning_to_rank | 2023 | 0.14 | 0.06 | 0.08 | 0.16 | 0.10 | 0.01 | 0.01 |
| query_expansion | 2016 | 0.24 | 0.11 | 0.38 | 0.56 | 0.36 | 0.12 | 0.12 |
| query_expansion | 2017 | 0.19 | 0.07 | 0.28 | 0.60 | 0.36 | 0.10 | 0.10 |
| query_expansion | 2018 | 0.20 | 0.12 | 0.30 | 0.50 | 0.32 | 0.08 | 0.08 |
| query_expansion | 2019 | 0.12 | 0.07 | 0.20 | 0.48 | 0.28 | 0.05 | 0.05 |
| query_expansion | 2020 | 0.08 | 0.06 | 0.20 | 0.38 | 0.34 | 0.03 | 0.03 |
| query_expansion | 2021 | 0.15 | 0.09 | 0.14 | 0.24 | 0.26 | 0.02 | 0.02 |
| query_expansion | 2022 | 0.12 | 0.07 | 0.12 | 0.20 | 0.16 | 0.01 | 0.01 |
| query_expansion | 2023 | 0.13 | 0.05 | 0.10 | 0.18 | 0.12 | 0.01 | 0.01 |
| recommender_systems | 2016 | 0.58 | 0.50 | 0.88 | 0.82 | 0.56 | 0.39 | 0.39 |
| recommender_systems | 2017 | 0.57 | 0.40 | 0.84 | 0.82 | 0.56 | 0.34 | 0.34 |
| recommender_systems | 2018 | 0.43 | 0.36 | 0.86 | 0.78 | 0.46 | 0.30 | 0.30 |
| recommender_systems | 2019 | 0.43 | 0.29 | 0.80 | 0.70 | 0.38 | 0.24 | 0.24 |
| recommender_systems | 2020 | 0.35 | 0.27 | 0.66 | 0.64 | 0.26 | 0.19 | 0.19 |
| recommender_systems | 2021 | 0.27 | 0.16 | 0.50 | 0.40 | 0.14 | 0.13 | 0.13 |
| recommender_systems | 2022 | 0.25 | 0.17 | 0.42 | 0.42 | 0.10 | 0.09 | 0.10 |
| recommender_systems | 2023 | 0.15 | 0.10 | 0.26 | 0.36 | 0.06 | 0.07 | 0.07 |
