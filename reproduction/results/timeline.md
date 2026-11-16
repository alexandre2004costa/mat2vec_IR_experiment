# Phase 4: embedding timeline of cs.IR

Subtopics are keyword sets matched on cs.IR abstracts (first-version year; an abstract can belong to several). A point = the mean SPECTER2 vector of a subtopic's abstracts in one year, kept only with ≥ 20 abstracts. Figures: `figures/timeline_specter2_panels.png` (one panel per subtopic) and `figures/timeline_specter2.png` (overview). Caveat: SPECTER2 was pretrained on recent papers.

Similarities below are cosines of centroids **minus the mean cs.IR abstract vector**: raw SPECTER2 centroids all have cosine 0.95–0.99 with each other, because all abstracts share a common direction. Centering does not change the PCA plot.

## A. SPECTER2 centroids

One PCA on all 64 centroids: PC1 explains 44.8% and PC2 20.9% of their variance (65.7% together). The 2-D plot is a projection; distances below are in the full space.

| subtopic | keywords | first year | last year | abstracts (first → last year) | centered cosine first↔last centroid |
|---|---|:-:|:-:|:-:|:-:|
| recommender systems | recommender, recommenders, recommendation, recommendations, collaborative filtering | 2012 | 2026 | 64 → 1388 | 0.43 |
| query expansion & reformulation | query expansion, query reformulation, query rewriting, query suggestion, relevance feedback | 2018 | 2026 | 25 → 111 | 0.55 |
| learning to rank | learning to rank, ranknet, lambdamart, lambdarank, listwise, pairwise ranking | 2017 | 2026 | 20 → 88 | 0.55 |
| dense & neural retrieval | dense retrieval, dense retriever, dense retrievers, dense passage retrieval, dual encoder, dual encoders, bi encoder, bi encoders, neural ranking, neural retrieval, colbert | 2020 | 2026 | 28 → 295 | 0.73 |
| conversational search | conversational search, conversational recommendation, conversational recommender, clarifying question, clarifying questions, conversational information seeking | 2020 | 2026 | 33 → 51 | 0.83 |
| IR evaluation | test collection, test collections, relevance judgments, relevance judgements, relevance assessments, evaluation metrics, evaluation measures, offline evaluation, cranfield | 2015 | 2026 | 22 → 119 | 0.11 |
| LLMs & RAG | large language model, large language models, llm, llms, retrieval augmented, rag | 2022 | 2026 | 29 → 2323 | 0.69 |

### Centered cosine similarity between subtopic centroids, by year

| pair | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| dense & neural retrieval ↔ LLMs & RAG | — | — | — | — | — | — | — | — | — | — | 0.68 | 0.48 | 0.58 | 0.62 | 0.71 |
| query expansion & reformulation ↔ LLMs & RAG | — | — | — | — | — | — | — | — | — | — | 0.45 | 0.59 | 0.70 | 0.80 | 0.78 |
| query expansion & reformulation ↔ dense & neural retrieval | — | — | — | — | — | — | — | — | 0.49 | 0.71 | 0.69 | 0.75 | 0.81 | 0.79 | 0.87 |
| recommender systems ↔ conversational search | — | — | — | — | — | — | — | — | 0.03 | 0.03 | 0.31 | 0.35 | 0.12 | 0.46 | 0.48 |
| learning to rank ↔ dense & neural retrieval | — | — | — | — | — | — | — | — | 0.24 | 0.28 | 0.08 | 0.23 | 0.40 | 0.52 | 0.66 |
| recommender systems ↔ LLMs & RAG | — | — | — | — | — | — | — | — | — | — | -0.47 | -0.19 | -0.25 | -0.13 | 0.05 |

### Points per subtopic and year (abstracts)

| subtopic                        |   2012 |   2013 |   2014 |   2015 |   2016 |   2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |   2026 |
|:--------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| IR evaluation                   |      0 |      0 |      0 |     22 |      0 |     31 |     38 |     50 |     52 |     59 |     74 |     90 |    125 |    147 |    119 |
| LLMs & RAG                      |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |     29 |    358 |   1322 |   2067 |   2323 |
| conversational search           |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |     33 |     37 |     39 |     60 |     68 |     65 |     51 |
| dense & neural retrieval        |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      0 |     28 |     83 |    111 |    145 |    161 |    207 |    295 |
| learning to rank                |      0 |      0 |      0 |      0 |      0 |     20 |     39 |     39 |     60 |     56 |     53 |     70 |     78 |     85 |     88 |
| query expansion & reformulation |      0 |      0 |      0 |      0 |      0 |      0 |     25 |     25 |     34 |     30 |     23 |     44 |     54 |     88 |    111 |
| recommender systems             |     64 |     74 |     75 |     91 |    143 |    223 |    340 |    478 |    600 |    745 |    849 |   1127 |   1382 |   1454 |   1388 |

### Robustness of the 2-D layout

Procrustes disparity between the main layout and a refit on a variant, on their common points (0 = identical shape up to rotation, scaling and reflection; 1 = unrelated).

| variant | common points | disparity |
|---|:-:|:-:|
| without 2026 (partial year) | 57 | 0.000 |
| ≥ 50 abstracts per point | 47 | 0.002 |

## B. word2vec: neighbours and movement across cutoffs (seed 0)

Spaces of different cutoffs are not aligned, so each pair of consecutive cutoffs is aligned by orthogonal Procrustes on the normalized input vectors of their shared vocabulary (HistWords). *Self-similarity* = cosine between a word's aligned vector at T and its vector at the next cutoff; lower = the word's usage moved more. It is compared with the median over all shared words.

| concept | 2016→2017 | 2017→2018 | 2018→2019 | 2019→2020 | 2020→2021 | 2021→2022 | 2022→2023 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `query_expansion` | 0.90 | 0.93 | 0.91 | 0.91 | 0.89 | 0.91 | 0.92 |
| `recommender` | 0.86 | 0.85 | 0.83 | 0.81 | 0.83 | 0.91 | 0.94 |
| `learning_to_rank` | 0.89 | 0.91 | 0.88 | 0.88 | 0.89 | 0.89 | 0.92 |
| `retrieval` | 0.86 | 0.90 | 0.88 | 0.90 | 0.90 | 0.91 | 0.94 |
| *median word* | 0.90 | 0.89 | 0.89 | 0.89 | 0.90 | 0.91 | 0.92 |

Nearest neighbours (input vectors, top 8, terms in the candidate vocabulary):

**`query_expansion`**

| T | neighbours |
|---|---|
| 2016 | bradfordizing, proposes_a_new, reformulating, refining, categorisation, enriches, bayesum, original_query |
| 2017 | aqe, bradfordizing, query_reformulation, proposes_a_new, enriches, relevance_feedback, reformulating, pseudo |
| 2018 | aqe, bradfordizing, query_reformulation, relevance_feedback, bayesum, finder, vocabulary_mismatch, expansion |
| 2019 | pseudo_relevance_feedback, aqe, bradfordizing, prf, bm25f, term_weighting, gsa, textrank |
| 2020 | aqe, pseudo_relevance_feedback, bradfordizing, prf, expansion_terms, keyphrase_extraction, elasticsearch, passage_retrieval |
| 2021 | pseudo_relevance_feedback, aqe, prf, bradfordizing, search, retrieval, set_expansion, okapi |
| 2022 | aqe, pseudo_relevance_feedback, prf, manifold_ranking, vocabulary_mismatch, bradfordizing, deepimpact, ceqe |
| 2023 | pseudo_relevance_feedback, pseudo_relevance_feedback_prf, term_suggestion, aqe, bradfordizing, query_reformulation, ceqe, deepimpact |

**`recommender`**

| T | neighbours |
|---|---|
| 2016 | recommendation, recommenders, clothes, collaborative_filtering_cf, rating_prediction, tour, recommendations, crs |
| 2017 | recommendation, recommendations, recommenders, recommender_systems, collaborative_filtering, clothes, collaborative_filtering_cf, coined |
| 2018 | recommendation, recommender_systems, recommendations, collaborative_filtering, dlib, recommends, collaborative_filtering_cf, middleware |
| 2019 | recommendation, recommender_systems, recommenders, collaborative_filtering, afel, rec, recommendation_algorithms, recommendations |
| 2020 | recommendation, recommendations, recommender_systems, collaborative_filtering, recommendation_systems, recommenders, recommendation_algorithms, collaborative_filtering_cf |
| 2021 | recommendation, recommendations, recommender_systems, recommenders, collaborative_filtering, recommendation_systems, hybrid_recommender, collaborative_filtering_cf |
| 2022 | recommendation, recommender_systems, recommendations, collaborative_filtering, recommenders, recommendation_systems, collaborative_filtering_cf, personalized_recommendation |
| 2023 | recommendation, recommender_systems, recommendations, recommenders, collaborative_filtering, recommendation_systems, movie_recommender, personalized_recommendation |

**`learning_to_rank`**

| T | neighbours |
|---|---|
| 2016 | bagging, rank_aggregation, adaboost, probfuse, mcf, implicit_feedback, benchmark_data_sets, ccf |
| 2017 | adaboost, l2r, ltr, bagging, lambdamart, rank_aggregation, regularized, mcf |
| 2018 | ltr, l2r, listwise, ranking, hcf, bpr, bm25, bm25f |
| 2019 | ltr, ranking, listwise, personalized_ranking, ranking_functions, rankers, ncr, gemrank |
| 2020 | ranking, letor, unbiased_learning_to_rank, ranking_functions, learning_to_rank_ltr, list_wise, bpr, listwise |
| 2021 | learning_to_rank_ltr, ranking, unbiased_learning_to_rank, list_wise, listwise, letor, plackett, gams |
| 2022 | ranking, learning_to_rank_ltr, letor, list_wise, listwise, plackett, rank_aggregation, ranking_functions |
| 2023 | learning_to_rank_ltr, ranking, listwise, letor, unbiased_learning_to_rank, ltr, list_wise, bayesian_personalized |

**`retrieval`**

| T | neighbours |
|---|---|
| 2016 | cross_media, fusing, bm25, subregions, birds, fusion, passage, probfuse |
| 2017 | cross_media, humming, subregions, cbvr, fusing, retrieving, zettair, eeg |
| 2018 | subregions, cross_media, cbvr, indexing_and_retrieval, humming, search, probfuse, document_retrieval |
| 2019 | subregions, search, mias, retrieving, pseudo_relevance_feedback, cbvr, lucene, probfuse |
| 2020 | search, vitrivr, prior_art, mias, subregions, query, nohis, cross_media |
| 2021 | matching, search, passage_retrieval, document_retrieval, vitrivr, pseudo_relevance_feedback, indexing_and_retrieval, query |
| 2022 | document_retrieval, matching, search, dense_passage, retrieving, vitrivr, retriever_reader, query |
| 2023 | parm, document_retrieval, search, matching, query, vitrivr, retrieving, dense_passage_retrieval |
