"""Local presentation website. Run: python -X utf8 pairs/12-mat2vec/web/server.py"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SYSTEMS = {}
MINILM_MODEL = None
MINILM_INDEXES = {}
LOCK = threading.Lock()
DATASETS = ('scifact', 'nfcorpus', 'arguana')


class MiniLMUnavailable(RuntimeError):
    pass


def minilm_scores(system, text):
    """Pretrained MiniLM inference; keep document embeddings only in memory."""
    from sentence_transformers import SentenceTransformer
    from testbed_demo import testbed
    global MINILM_MODEL
    if MINILM_MODEL is None:
        try:
            MINILM_MODEL = SentenceTransformer('all-MiniLM-L6-v2',
                device=testbed.device(), local_files_only=True)
        except OSError as error:
            raise MiniLMUnavailable('The pretrained all-MiniLM-L6-v2 model is not available '
                'in the local Hugging Face cache. See web/README.md; no training is required.') from error
        MINILM_MODEL.eval()
    if system.name not in MINILM_INDEXES:
        # Preserve the same document order as BM25 and Word2Vec.
        corpus = {doc_id: system.corpus[doc_id] for doc_id in system.ids}
        MINILM_INDEXES[system.name] = MINILM_MODEL.encode(testbed.doc_texts(corpus),
            batch_size=64, normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True)
    query = MINILM_MODEL.encode([text], normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True)[0]
    # Use the encoder's own preprocessing and truncation, including special tokens.
    inputs = MINILM_MODEL.tokenize([text])
    token_ids = inputs['input_ids'][0][inputs['attention_mask'][0].bool()].tolist()
    tokens = MINILM_MODEL.tokenizer.convert_ids_to_tokens(token_ids)
    return MINILM_INDEXES[system.name] @ query, tokens


def search(dataset, query, k):
    import numpy as np
    from testbed_demo import Systems
    with LOCK:
        if dataset not in SYSTEMS:
            SYSTEMS[dataset] = Systems(dataset)
        system = SYSTEMS[dataset]
        started = time.perf_counter()
        qid = query if query in system.queries else None
        text = system.queries[qid] if qid else query
        labels = system.qrels.get(qid, {}) if qid else {}
        scores = {'bm25': system.bm(text), 'word2vec': system.ours(text),
                  'minilm': minilm_scores(system, text)}
        result = {'query': text, 'query_id': qid, 'dataset': dataset, 'systems': {}}
        for name, (values, terms) in scores.items():
            values = values.copy()
            if dataset == 'arguana' and qid:
                values[np.array(system.ids) == qid] = -np.inf
            order = [int(j) for j in np.argsort(-values, kind='stable') if np.isfinite(values[j])]
            ranks = {system.ids[j]: i + 1 for i, j in enumerate(order)}
            empty = not order or (name != 'minilm' and
                (not terms or not np.any(values[np.isfinite(values)])))
            rows = [] if empty else [dict(id=system.ids[j], rank=i + 1,
                title=system.corpus[system.ids[j]]['title'] or '(Untitled document)',
                text=system.corpus[system.ids[j]]['text'], score=float(values[j]),
                relevance=labels.get(system.ids[j], 0) if qid else None)
                for i, j in enumerate(order[:k])]
            result['systems'][name] = {'terms': terms, 'empty': empty, 'results': rows,
                'relevant_ranks': [{'id': d, 'rank': ranks.get(d), 'title': system.corpus[d]['title']}
                                   for d, grade in labels.items() if grade > 0 and d in system.corpus]}
        result['milliseconds'] = round((time.perf_counter() - started) * 1000)
        return result


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == '/api/results':
                files = [ROOT / 'results' / f'validated_seed{seed}_vs_bm25.json' for seed in range(3)]
                return self.send_json({'runs': [json.loads(p.read_text()) for p in files],
                    'minilm': json.loads((ROOT / 'results_vs_minilm.json').read_text())})
            if url.path == '/api/search':
                args = parse_qs(url.query)
                dataset = args.get('dataset', ['scifact'])[0]
                query = args.get('q', [''])[0].strip()
                k = int(args.get('k', ['10'])[0])
                if dataset not in DATASETS or not query or len(query) > 4000 or not 1 <= k <= 20:
                    return self.send_json({'error': 'Choose a dataset and enter a query (up to 4,000 characters).'}, 400)
                return self.send_json(search(dataset, query, k))
            assets = {
                '/': ('index.html', 'text/html; charset=utf-8'),
                '/index.html': ('index.html', 'text/html; charset=utf-8'),
                '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
                '/script.js': ('script.js', 'text/javascript; charset=utf-8'),
            }
            if url.path in assets:
                filename, content_type = assets[url.path]
                body = Path(__file__).with_name(filename).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_error(404)
        except (ValueError, KeyError):
            self.send_json({'error': 'Invalid query parameters.'}, 400)
        except MiniLMUnavailable as error:
            self.send_json({'error': str(error)}, 503)
        except ImportError as error:
            self.send_json({'error': f'Missing Python dependency: {error}. Run the server in the IR2 environment with the project requirements installed.',
                            'detail': str(error)}, 503)
        except FileNotFoundError as error:
            self.send_json({'error': f'Demo file missing: {error}. See pairs/12-mat2vec/web/README.md.',
                            'detail': str(error)}, 503)
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_json({'error': 'Search failed. See the server terminal for details.'}, 500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Demo website: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
