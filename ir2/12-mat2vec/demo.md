# Demo queries (testbed, BM25 vs word2vec + SIF trained with the paper's recipe)

Run from the repository root. Indexes are built beforehand (`ir2/testbed_demo.py build`, and `method.py` saves ours); each answer takes < 0.1 s. Interactive mode: `ir2/.venv/bin/python ir2/testbed_demo.py show --dataset scifact` takes a query id or free text. ✓ marks documents labelled relevant; the last lines give each relevant document's rank under BM25 → word2vec.

- **Hero**: `ir2/.venv/bin/python ir2/testbed_demo.py show --dataset scifact --query 212`
  "CR is associated with higher methylation age." The labelled abstract, *Caloric restriction delays age-related methylation drift*, is ranked 9th by BM25 and **1st by word2vec**. "CR" is an abbreviation: BM25 can only match the letters `cr` and puts *Crossover and noncrossover pathways* first. word2vec learned from the corpus that `cr` is used like "caloric restriction", because the two appear in the same contexts. This is the paper's idea at work: meaning comes from context, without labels. Backup: `--query 1191` ("The amount of publicly available DNA data doubles every 10 years"): BM25 rank 631 → word2vec rank 4. It is the same query as the instructor's worked example, where Beyond Abstracts got it to rank 1.
- **Failure**: `ir2/.venv/bin/python ir2/testbed_demo.py show --dataset nfcorpus --query PLAIN-1473`
  "kohlrabi". BM25 ranks the only relevant abstract 1st; word2vec ranks it **2,781st**. The word occurs fewer than 5 times in the corpus, so it has no vector: the query becomes empty and the ranking is arbitrary. 38 of 323 NFCorpus queries fail this way. It is the same lesson as the backtest's "RAG before it had a word". Second failure, with a query the model does know: `--dataset scifact --query 1137` ("TNFAIP3 is a tumor suppressor in glioblastoma"): BM25 1st → word2vec 20th. The gene name, the one discriminating word, is dropped from the vocabulary and averaged away.
- **Wildcard**: proposed on the spot in interactive mode (SciFact claims work best; free text has no relevance labels, so judge by the titles). Backup: "Eating less slows down biological aging": word2vec returns aging research (*The Hallmarks of Aging*, *Extending healthy life span*), BM25 returns eating-behaviour papers (*binge eating*, *weight loss strategies*).

The honest frame: these are hand-picked queries. Over all queries the method loses to BM25 on every dataset (SciFact nDCG@10 0.42 vs 0.68): it wins 29 of 300 SciFact queries and loses 154.

Fallback recording: link (to record before the session).
