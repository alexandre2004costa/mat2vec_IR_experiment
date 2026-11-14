# Local demo website


## Running instructions
#### 1. Run 

```
conda activate IR2
python -X utf8 ir2/web/server.py
```

#### 2. Open http://127.0.0.1:8765

#### 3. Stop with `Ctrl+C`
___
___

## Website instructions

1. Select a collection
2. Then:
   1. Type either:
      - query string
      - query ID
   2. Or select either:
      - hero query
      - failure query
3. Click `Compare`


> Note: The `Clear` button clears the input and output. 


Other information:
- Relevant badges are available only for known test query IDs. 
- A query with no usable word vector displays an explanation.
- Scores from BM25 and cosine similarity have different scales. Only compare ranks, not score magnitude.

___
___

## Prepare indexes on a fresh checkout

```
git clone --depth 1 https://github.com/kimgerdes/ir2-testbed.git ../ir2-testbed
pip install -r ir2/requirements.txt
pip install nltk rank-bm25
python -X utf8 ir2/testbed/method.py --seed 0
python -X utf8 ir2/testbed/testbed_demo.py build
```

Other information:
- The first comparison always loads the collection, so that later comparisons do not need to reload.

- ArguAna excludes a known query's own document.
