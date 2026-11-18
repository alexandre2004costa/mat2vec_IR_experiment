# Demo Information

This is the presentation of the queries that will be presented during the demo. The demo consists of a hero query, a failure query and a wildcard query. Each query includes a backup in case another example is needed.

The comparison that will be displayed in the demo is between BM25, MiniLM and Word2Vec + SIF. The same query and the same dataset will be used to rank the documents that are more similar to the query. 

On one hand, BM25 ranks documents by matching query terms, taking into account how rare those terms are, how often they occur in each document and the document length. 

In addition, MiniLM (`all-MiniLM-L6-v2`) represents queries and documents via a pretrained encoder and ranks documents by cosine similarity. It taskes into account the word order and the context, unlike Word2Vec.

On the other hand, Word2Vec learns word representations from the contexts in which words appear. For this task, documents and queries are represented by weighted averages of their word vectors and ranked by cosine similarity.

Furthermore, SIF gives frequent words less weight through `a / (a + p(w))`, with `a = 0.001` and `p(w)` estimated from the training corpus. SIF also removes the strongest shared direction for both documents and queries, so that documents with similar language patterns (despite different topics) are less connected. 

As a result, the search made by Word2Vec + SIF focus on informative words. However, it is important to rebember that averaging still loses word order and can weaken relationships between terms.


## Demo queries

### Hero query

> *"CR is associated with higher methylation age."*

This query was chosen since it includes an abbreviation, `CR`, in a labelled query whose relevant abstract is titled *Caloric restriction delays age-related methylation drift*.

By BM25, this document is only ranked 9th. This is because BM25 matches `CR` without understanding its meaning. This gives as first result the document titled *Crossover and non crossover pathways in mouse meiosis*.

By MiniLM, the relevant document is at 6th position. Top scores are included for documents that mention ``methylation``, `age` and `associated`. However, when the abreviation is removed, the relevant document places 1st. This suggest that MiniLM might focus on associations that are more literal.

By Word2Vec, the relevant document is ranked 1st. While learning from the corpus, Word2Vec identified `CR` to be close to `caloric restriction`, since both appear in the same contexts. As a consequence, this suggests that it is possible to retrieve knowledge from context alone, without labels.

Observe the query through:

```bash
python pairs/12-mat2vec/testbed_demo.py show --dataset scifact --query 212
```

### Backup hero query

> *"The amount of publicly available DNA data doubles every 10 years."*

This query shows an even more contrasting result. Word2Vec ranks the relevant document, *The EMBL nucleotide sequence database*, at the 4th place, but BM25 places the same abstract nowhere near the top ranking, at the 631st place.

This is the same hero query proposed in ``ir2-testbed/pairs/00-beyond-abstracts
/demo.md``, where Beyond Abstracts ranked the same relevant document as 1st.

In addition, this query suggests that Word2Vec + SIF normalise `10` into a number placeholder, which makes its embedding excluded from the averaged representation. Hence, a limitation of this method is that it looses the timescale of queries.

Surprisingly, MiniLM ranks the document 2nd place, but the score of the 1st document is very close (0.419 vs 0.411).

Observe the query through:

```bash
python pairs/12-mat2vec/testbed_demo.py show --dataset scifact --query 1191
```

### Failure query

> *"kohlrabi"*

For the rankings of the failure query, the tables have turned. Since there is only one labelled relevant abstract, BM25 ranks it 1st. Despite that, Word2Vec cannot score precisely any document, which leads to the actual document being ranked 2781st.

Looking further into the problem, `kohlrabi` occurs only twice in the saved training token counts. As Word2Vec uses `min_count=5`, this word is excluded from its vocabulary. Therefore, no word vector is learned for it. And since the query  only contains that word, there are no other vectors to average, so the resulting query representation is a zero vector. When the search is executed, there is no information that can be used to rank documents above one another. This leads to an arbitrary ranking, which leads to the original corpus order.

Of course, this is not the only query that fails the search for Word2Vec. Inside NFCorpus, 38 of 323 queries have no retained in-vocabulary terms.

What can be learned from this result is that vocabulary coverage matters. Excluding rare words reduces the number of poorly supported vectors, even though it might remove important terms for retrieval. Clearly, adding more domain text would improve frequency of essential words, and also improve their contexts.

One thing is for sure, this same effect occurs in the backtest whenever the word `RAG` is used to predict discoveries: no one used this term in past articles because the term was not coined yet, although researchers used combination of words to provide similar meaning, trying to predict `RAG` will be pratically impossible.

Lastly, although MiniLM can split rare words into subword tokens, it does not necessarily mean that the model understands each token, and return the appropriate document. This is why the relevant document is ranked at 117th.

Observe the query through:

```bash
python pairs/12-mat2vec/testbed_demo.py show --dataset nfcorpus --query PLAIN-1473
```

### Backup failure query

> *"TNFAIP3 is a tumor suppressor in glioblastoma."*

In comparison to the previous query, this query can be represented by Word2Vec. By BM25 and MiniLM, the relevant document, *Targeting A20 Decreases Glioma Stem Cell Survival and Tumor Growth*, is ranked 1st. By Word2Vec, it is ranked 27th.

As seen previously, `TNFAIP3` is excluded from the Word2Vec vocabulary due to being a rare word. Therefore, other query terms are used, such as `tumor_suppressor` and `glioblastoma`. This leads to a broader search, hence the lower ranking.

Observe the query through:

```bash
python pairs/12-mat2vec/testbed_demo.py show --dataset scifact --query 1137
```

### Wildcard query

This query is going to be proposed on the spot. Some things to consider, however:

- Queries that are written without any label in mind (aka free-text queries) will work. But, documents will not show any "Relevant" badge. Thus, to understand the performance of either method, the ranking needs to be evaluated one by one.

- Providing the query ID, will guarantee that both the query and relevance labels are correctly loaded. 

- An empty list of Word2Vec terms means that every document receives the same score.

- A nonempty list means the query can be represented, but does not guarantee a useful ranking.


### Backup wildcard query

If the audience does not select their own queries, a backup query can be mentioned:

> *"Eating less slows down biological aging"*

This query shows a divergence of how the methods search for documents. Word2Vec delivers research related to aging, such as *The Hallmarks of Aging* as 1st and *Extending healthy life span-from yeast to humans* as 3rd. Meanwhile, BM25 returns abstracts focused on eating behaviours. Lastly, MiniLM shows a mixture of both topics, although strongly towards aging related documents.

Observe the query through:

```bash
python pairs/12-mat2vec/testbed_demo.py show --dataset scifact --query "Eating less slows down biological aging"
```

### Final reminder

These selected queries only showcase a tiny part of the whole algorithm comparison. Current evaluation indicates that Word2Vec + SIF still underperforms both BM25 and MiniLM on all three datasets, for both nDCG@10 and Recall@100.


### Fallback recording

> https://drive.google.com/file/d/1-sql0BQsx1srwHF7GOLx4CNEnvkQ-T24/view?usp=sharing