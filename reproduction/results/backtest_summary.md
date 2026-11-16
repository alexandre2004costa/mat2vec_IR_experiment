# Backtest results: all methods

Precision@k = share of the top-k candidates that co-occur with a target indicator in ≥ 3 cs.IR abstracts from T to 2026. word2vec rows: mean over scored pairs, ± std across 3 seeds. "Frequency-matched" = expected P@50 if the same top-50 were replaced by random candidates of the same pre-T frequency decile; the last column is what a method adds beyond frequency.

### Candidates: `all` — 28 scored (target, T) pairs

| method | P@10 | P@20 | P@50 | P@50 frequency-matched | P@50 − freq-matched |
|---|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.252 ± 0.011 | 0.277 ± 0.010 | 0.279 ± 0.006 | 0.178 ± 0.002 | 0.101 ± 0.005 |
| word2vec input·input | 0.094 ± 0.005 | 0.094 ± 0.011 | 0.104 ± 0.003 | 0.077 ± 0.000 | 0.027 ± 0.003 |
| SPECTER2 abstract means | 0.529 | 0.491 | 0.467 | 0.184 | 0.283 |
| 2nd-order co-occurrence | 0.800 | 0.791 | 0.760 | 0.348 | 0.412 |
| frequency | 0.457 | 0.475 | 0.513 | 0.353 | 0.160 |
| random (1000 draws) | 0.138 | 0.137 | 0.137 | — | — |

Mean base rate (share of all candidates found): 0.137.

### Candidates: `phrases` — 28 scored (target, T) pairs

| method | P@10 | P@20 | P@50 | P@50 frequency-matched | P@50 − freq-matched |
|---|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.298 ± 0.027 | 0.280 ± 0.019 | 0.254 ± 0.002 | 0.121 ± 0.001 | 0.133 ± 0.002 |
| word2vec input·input | 0.180 ± 0.005 | 0.158 ± 0.006 | 0.155 ± 0.003 | 0.080 ± 0.000 | 0.075 ± 0.003 |
| SPECTER2 abstract means | 0.479 | 0.429 | 0.362 | 0.114 | 0.248 |
| 2nd-order co-occurrence | 0.596 | 0.561 | 0.475 | 0.186 | 0.289 |
| frequency | 0.368 | 0.337 | 0.281 | 0.205 | 0.076 |
| random (1000 draws) | 0.103 | 0.102 | 0.102 | — | — |

Mean base rate (share of all candidates found): 0.102.

## Status of every (target, T) pair

`pre_T_target_docs_ir`: pre-T cs.IR abstracts with an indicator (the validity rule uses this). `pre_T_target_docs`: the same over all training text (cs.IR ∪ cs.CL).

| target                         |    T | status                              |   pre_T_target_docs_ir |   pre_T_target_docs |   post_T_target_docs | target_tokens                                                                                                              |   n_candidates_all |   base_rate_all |   n_candidates_phrases |   base_rate_phrases |
|:-------------------------------|-----:|:------------------------------------|-----------------------:|--------------------:|---------------------:|:---------------------------------------------------------------------------------------------------------------------------|-------------------:|----------------:|-----------------------:|--------------------:|
| query_expansion                | 2016 | scored                              |                     65 |                  71 |                  438 | query_expansion relevance_feedback                                                                                         |               6330 |           0.166 |                    717 |               0.121 |
| recommender_systems            | 2016 | scored                              |                    359 |                 373 |                 8729 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               4026 |           0.557 |                    451 |               0.397 |
| learning_to_rank               | 2016 | fewer than 20 pre-T cs.IR abstracts |                     17 |                  18 |                  563 | learning_to_rank lambdamart                                                                                                |               7405 |           0.273 |                    821 |               0.185 |
| dense_retrieval                | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  824 | nan                                                                                                                        |               8134 |           0.375 |                    890 |               0.238 |
| conversational_search          | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  370 | nan                                                                                                                        |               8134 |           0.242 |                    890 |               0.146 |
| retrieval_augmented_generation | 2016 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                 2136 | nan                                                                                                                        |               8134 |           0.567 |                    890 |               0.372 |
| query_expansion                | 2017 | scored                              |                     78 |                  84 |                  424 | query_expansion query_reformulation relevance_feedback                                                                     |               7546 |           0.135 |                   1058 |               0.098 |
| recommender_systems            | 2017 | scored                              |                    495 |                 522 |                 8586 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               4558 |           0.489 |                    640 |               0.347 |
| learning_to_rank               | 2017 | scored                              |                     31 |                  34 |                  549 | learning_to_rank lambdamart                                                                                                |               8420 |           0.216 |                   1172 |               0.163 |
| dense_retrieval                | 2017 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   0 |                  824 | nan                                                                                                                        |               9585 |           0.338 |                   1293 |               0.226 |
| conversational_search          | 2017 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                  369 | nan                                                                                                                        |               9535 |           0.209 |                   1287 |               0.127 |
| retrieval_augmented_generation | 2017 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                 2135 | nan                                                                                                                        |               9558 |           0.520 |                   1290 |               0.348 |
| query_expansion                | 2018 | scored                              |                     92 |                 101 |                  410 | query_expansion query_reformulation relevance_feedback                                                                     |               9061 |           0.104 |                   1515 |               0.080 |
| recommender_systems            | 2018 | scored                              |                    703 |                 749 |                 8363 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               5276 |           0.420 |                    876 |               0.309 |
| learning_to_rank               | 2018 | scored                              |                     47 |                  53 |                  533 | learning_to_rank lambdamart listwise                                                                                       |               9749 |           0.162 |                   1579 |               0.118 |
| dense_retrieval                | 2018 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   3 |                  824 | nan                                                                                                                        |              11235 |           0.289 |                   1796 |               0.200 |
| conversational_search          | 2018 | fewer than 20 pre-T cs.IR abstracts |                      4 |                   6 |                  366 | nan                                                                                                                        |              11078 |           0.166 |                   1777 |               0.104 |
| retrieval_augmented_generation | 2018 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   1 |                 2135 | nan                                                                                                                        |              11370 |           0.471 |                   1813 |               0.327 |
| query_expansion                | 2019 | scored                              |                    112 |                 123 |                  387 | query_expansion query_reformulation relevance_feedback                                                                     |              11062 |           0.074 |                   2168 |               0.054 |
| recommender_systems            | 2019 | scored                              |                   1014 |                1092 |                 8023 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               6230 |           0.332 |                   1186 |               0.255 |
| learning_to_rank               | 2019 | scored                              |                     74 |                  84 |                  496 | learning_to_rank lambdamart listwise                                                                                       |              11492 |           0.108 |                   2198 |               0.078 |
| dense_retrieval                | 2019 | fewer than 20 pre-T cs.IR abstracts |                      0 |                   3 |                  824 | nan                                                                                                                        |              13604 |           0.248 |                   2562 |               0.172 |
| conversational_search          | 2019 | fewer than 20 pre-T cs.IR abstracts |                      6 |                  11 |                  363 | nan                                                                                                                        |              13260 |           0.130 |                   2517 |               0.081 |
| retrieval_augmented_generation | 2019 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   2 |                 2135 | nan                                                                                                                        |              13694 |           0.415 |                   2573 |               0.284 |
| query_expansion                | 2020 | scored                              |                    136 |                 150 |                  365 | query_expansion query_reformulation relevance_feedback                                                                     |              13622 |           0.052 |                   3023 |               0.036 |
| recommender_systems            | 2020 | scored                              |                   1471 |                1596 |                 7545 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               7490 |           0.242 |                   1593 |               0.191 |
| learning_to_rank               | 2020 | scored                              |                    114 |                 132 |                  459 | learning_to_rank lambdamart listwise                                                                                       |              13774 |           0.067 |                   3000 |               0.045 |
| dense_retrieval                | 2020 | fewer than 20 pre-T cs.IR abstracts |                      1 |                  11 |                  823 | dual_encoder                                                                                                               |              16302 |           0.199 |                   3520 |               0.140 |
| conversational_search          | 2020 | fewer than 20 pre-T cs.IR abstracts |                     17 |                  24 |                  353 | nan                                                                                                                        |              15827 |           0.089 |                   3450 |               0.053 |
| retrieval_augmented_generation | 2020 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   2 |                 2135 | nan                                                                                                                        |              16702 |           0.360 |                   3593 |               0.243 |
| query_expansion                | 2021 | scored                              |                    167 |                 189 |                  334 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              16694 |           0.033 |                   4068 |               0.024 |
| recommender_systems            | 2021 | scored                              |                   2034 |                2233 |                 6945 | recommender recommenders recommendation recommendations collaborative_filtering                                            |               9046 |           0.165 |                   2039 |               0.138 |
| learning_to_rank               | 2021 | scored                              |                    169 |                 194 |                  402 | learning_to_rank lambdamart listwise                                                                                       |              16529 |           0.036 |                   3971 |               0.026 |
| dense_retrieval                | 2021 | fewer than 20 pre-T cs.IR abstracts |                      7 |                  30 |                  814 | dual_encoder                                                                                                               |              19323 |           0.147 |                   4578 |               0.104 |
| conversational_search          | 2021 | scored                              |                     44 |                  55 |                  320 | conversational_search clarifying_questions                                                                                 |              18770 |           0.054 |                   4500 |               0.030 |
| retrieval_augmented_generation | 2021 | fewer than 20 pre-T cs.IR abstracts |                      1 |                   3 |                 2135 | nan                                                                                                                        |              20255 |           0.309 |                   4783 |               0.218 |
| query_expansion                | 2022 | scored                              |                    191 |                 216 |                  309 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              20326 |           0.024 |                   5265 |               0.016 |
| recommender_systems            | 2022 | scored                              |                   2743 |                3044 |                 6200 | recommender recommenders recommendation recommendations collaborative_filtering                                            |              10815 |           0.106 |                   2537 |               0.101 |
| learning_to_rank               | 2022 | scored                              |                    220 |                 249 |                  350 | learning_to_rank lambdamart listwise                                                                                       |              19889 |           0.022 |                   5087 |               0.017 |
| dense_retrieval                | 2022 | scored                              |                     54 |                 102 |                  756 | dense_retrieval dense_retriever dense_retrievers dual_encoder bi_encoder bi_encoders                                       |              21910 |           0.089 |                   5565 |               0.061 |
| conversational_search          | 2022 | scored                              |                     82 |                 104 |                  283 | conversational_search clarifying_questions                                                                                 |              21815 |           0.026 |                   5613 |               0.016 |
| retrieval_augmented_generation | 2022 | fewer than 20 pre-T cs.IR abstracts |                      5 |                  18 |                 2131 | rag                                                                                                                        |              23681 |           0.253 |                   6011 |               0.184 |
| query_expansion                | 2023 | scored                              |                    212 |                 244 |                  288 | query_expansion query_reformulation query_rewriting relevance_feedback                                                     |              24113 |           0.017 |                   6471 |               0.012 |
| recommender_systems            | 2023 | scored                              |                   3565 |                3968 |                 5351 | recommender recommenders recommendation recommendations collaborative_filtering                                            |              12815 |           0.066 |                   3038 |               0.074 |
| learning_to_rank               | 2023 | scored                              |                    267 |                 299 |                  303 | learning_to_rank ranknet lambdamart listwise                                                                               |              23487 |           0.015 |                   6230 |               0.013 |
| dense_retrieval                | 2023 | scored                              |                    128 |                 232 |                  671 | dense_retrieval dense_retriever dense_retrievers dense_passage_retrieval dual_encoder dual_encoders bi_encoder bi_encoders |              24373 |           0.047 |                   6425 |               0.035 |
| conversational_search          | 2023 | scored                              |                    119 |                 152 |                  244 | conversational_search clarifying_questions                                                                                 |              25250 |           0.014 |                   6777 |               0.010 |
| retrieval_augmented_generation | 2023 | fewer than 20 pre-T cs.IR abstracts |                     12 |                  53 |                 2120 | retrieval_augmented rag                                                                                                    |              26963 |           0.201 |                   7154 |               0.152 |

## P@50 per pair, candidates `all` (word2vec: mean of 3 seeds)

| target | T | w2v_out | w2v_in | specter2 | cooc2 | frequency | random | base_rate |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| conversational_search | 2021 | 0.23 | 0.01 | 0.20 | 0.82 | 0.66 | 0.05 | 0.05 |
| conversational_search | 2022 | 0.09 | 0.00 | 0.14 | 0.70 | 0.44 | 0.03 | 0.03 |
| conversational_search | 2023 | 0.07 | 0.02 | 0.06 | 0.50 | 0.34 | 0.01 | 0.01 |
| dense_retrieval | 2022 | 0.41 | 0.08 | 0.50 | 0.92 | 0.80 | 0.09 | 0.09 |
| dense_retrieval | 2023 | 0.18 | 0.06 | 0.24 | 0.68 | 0.68 | 0.05 | 0.05 |
| learning_to_rank | 2017 | 0.49 | 0.17 | 0.82 | 1.00 | 0.72 | 0.22 | 0.22 |
| learning_to_rank | 2018 | 0.49 | 0.08 | 0.72 | 0.96 | 0.66 | 0.16 | 0.16 |
| learning_to_rank | 2019 | 0.36 | 0.06 | 0.60 | 0.90 | 0.48 | 0.11 | 0.11 |
| learning_to_rank | 2020 | 0.17 | 0.03 | 0.42 | 0.80 | 0.46 | 0.06 | 0.07 |
| learning_to_rank | 2021 | 0.15 | 0.02 | 0.32 | 0.60 | 0.30 | 0.03 | 0.04 |
| learning_to_rank | 2022 | 0.09 | 0.00 | 0.22 | 0.50 | 0.24 | 0.02 | 0.02 |
| learning_to_rank | 2023 | 0.13 | 0.02 | 0.14 | 0.46 | 0.22 | 0.01 | 0.01 |
| query_expansion | 2016 | 0.31 | 0.10 | 0.70 | 0.92 | 0.76 | 0.17 | 0.17 |
| query_expansion | 2017 | 0.29 | 0.08 | 0.64 | 0.92 | 0.72 | 0.14 | 0.13 |
| query_expansion | 2018 | 0.28 | 0.06 | 0.48 | 0.86 | 0.66 | 0.11 | 0.10 |
| query_expansion | 2019 | 0.21 | 0.05 | 0.30 | 0.84 | 0.56 | 0.07 | 0.07 |
| query_expansion | 2020 | 0.19 | 0.05 | 0.26 | 0.66 | 0.60 | 0.05 | 0.05 |
| query_expansion | 2021 | 0.17 | 0.09 | 0.18 | 0.58 | 0.40 | 0.03 | 0.03 |
| query_expansion | 2022 | 0.15 | 0.06 | 0.18 | 0.50 | 0.42 | 0.02 | 0.02 |
| query_expansion | 2023 | 0.09 | 0.01 | 0.14 | 0.44 | 0.34 | 0.02 | 0.02 |
| recommender_systems | 2016 | 0.60 | 0.48 | 1.00 | 1.00 | 0.82 | 0.56 | 0.56 |
| recommender_systems | 2017 | 0.60 | 0.44 | 0.98 | 1.00 | 0.78 | 0.48 | 0.49 |
| recommender_systems | 2018 | 0.51 | 0.37 | 0.98 | 0.96 | 0.64 | 0.42 | 0.42 |
| recommender_systems | 2019 | 0.43 | 0.23 | 0.88 | 0.92 | 0.50 | 0.33 | 0.33 |
| recommender_systems | 2020 | 0.40 | 0.24 | 0.68 | 0.86 | 0.38 | 0.24 | 0.24 |
| recommender_systems | 2021 | 0.39 | 0.07 | 0.56 | 0.72 | 0.36 | 0.17 | 0.17 |
| recommender_systems | 2022 | 0.18 | 0.03 | 0.46 | 0.66 | 0.24 | 0.11 | 0.11 |
| recommender_systems | 2023 | 0.14 | 0.02 | 0.28 | 0.60 | 0.18 | 0.07 | 0.07 |

## P@50 per pair, candidates `phrases` (word2vec: mean of 3 seeds)

| target | T | w2v_out | w2v_in | specter2 | cooc2 | frequency | random | base_rate |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| conversational_search | 2021 | 0.14 | 0.08 | 0.14 | 0.48 | 0.26 | 0.03 | 0.03 |
| conversational_search | 2022 | 0.05 | 0.00 | 0.06 | 0.36 | 0.18 | 0.02 | 0.02 |
| conversational_search | 2023 | 0.05 | 0.04 | 0.02 | 0.26 | 0.10 | 0.01 | 0.01 |
| dense_retrieval | 2022 | 0.34 | 0.11 | 0.44 | 0.70 | 0.46 | 0.06 | 0.06 |
| dense_retrieval | 2023 | 0.14 | 0.07 | 0.16 | 0.38 | 0.24 | 0.04 | 0.03 |
| learning_to_rank | 2017 | 0.40 | 0.27 | 0.70 | 0.70 | 0.42 | 0.16 | 0.16 |
| learning_to_rank | 2018 | 0.43 | 0.21 | 0.48 | 0.66 | 0.40 | 0.12 | 0.12 |
| learning_to_rank | 2019 | 0.43 | 0.15 | 0.40 | 0.48 | 0.28 | 0.08 | 0.08 |
| learning_to_rank | 2020 | 0.27 | 0.06 | 0.28 | 0.36 | 0.28 | 0.04 | 0.04 |
| learning_to_rank | 2021 | 0.13 | 0.05 | 0.24 | 0.26 | 0.24 | 0.03 | 0.03 |
| learning_to_rank | 2022 | 0.10 | 0.03 | 0.10 | 0.24 | 0.18 | 0.02 | 0.02 |
| learning_to_rank | 2023 | 0.14 | 0.06 | 0.10 | 0.18 | 0.10 | 0.01 | 0.01 |
| query_expansion | 2016 | 0.25 | 0.16 | 0.40 | 0.58 | 0.32 | 0.12 | 0.12 |
| query_expansion | 2017 | 0.20 | 0.08 | 0.30 | 0.56 | 0.34 | 0.10 | 0.10 |
| query_expansion | 2018 | 0.15 | 0.09 | 0.30 | 0.50 | 0.30 | 0.08 | 0.08 |
| query_expansion | 2019 | 0.15 | 0.11 | 0.20 | 0.50 | 0.24 | 0.05 | 0.05 |
| query_expansion | 2020 | 0.11 | 0.06 | 0.20 | 0.40 | 0.34 | 0.04 | 0.04 |
| query_expansion | 2021 | 0.14 | 0.11 | 0.16 | 0.24 | 0.24 | 0.02 | 0.02 |
| query_expansion | 2022 | 0.14 | 0.08 | 0.10 | 0.20 | 0.18 | 0.02 | 0.02 |
| query_expansion | 2023 | 0.09 | 0.04 | 0.10 | 0.18 | 0.14 | 0.01 | 0.01 |
| recommender_systems | 2016 | 0.55 | 0.51 | 0.84 | 0.84 | 0.58 | 0.40 | 0.40 |
| recommender_systems | 2017 | 0.59 | 0.49 | 0.82 | 0.80 | 0.56 | 0.35 | 0.35 |
| recommender_systems | 2018 | 0.52 | 0.34 | 0.86 | 0.78 | 0.46 | 0.31 | 0.31 |
| recommender_systems | 2019 | 0.49 | 0.33 | 0.82 | 0.70 | 0.38 | 0.25 | 0.25 |
| recommender_systems | 2020 | 0.38 | 0.25 | 0.66 | 0.60 | 0.24 | 0.19 | 0.19 |
| recommender_systems | 2021 | 0.28 | 0.19 | 0.52 | 0.56 | 0.22 | 0.14 | 0.14 |
| recommender_systems | 2022 | 0.25 | 0.21 | 0.40 | 0.44 | 0.10 | 0.10 | 0.10 |
| recommender_systems | 2023 | 0.19 | 0.17 | 0.34 | 0.36 | 0.08 | 0.08 | 0.07 |
