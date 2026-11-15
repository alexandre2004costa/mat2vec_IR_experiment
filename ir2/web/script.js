const $ = id => document.getElementById(id);

const esc = s => String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
}[c]));

async function get(url) {
    const response = await fetch(url);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || 'Request failed');
    }

    return data;
}

async function compare(event) {
    event?.preventDefault();

    const q = $('query').value.trim();

    if (!q) {
        $('query').focus();
        return;
    }

    $('submit').disabled = true;
    document.querySelectorAll('.presets button').forEach(button => {
        button.disabled = true;
    });

    $('output').hidden = true;
    $('status').textContent = 'Comparing... The first comparison loads the collection into memory.';

    try {
        const params = new URLSearchParams({
            q,
            dataset: $('dataset').value,
            k: 10
        });
        const data = await get('/api/search?' + params);

        $('query-title').textContent = "\"" + data.query + "\"";
        $('labels').textContent = data.query_id
            ? 'Query ' + data.query_id
            : 'Free-text query';
        $('notes').textContent = data.query_id
            ? 'Note: "Relevant" marks documents with a positive testbed relevance label. Unmarked documents may be unjudged.'
            : 'Note: no relevance judgments are available for this prompt.';
        $('notes-mandatory').textContent = 'Note: Scores use different scales, hence compare ranks not magnitudes.'

        for (const name of ['bm25', 'word2vec']) {
            const system = data.systems[name];
            const termsHtml = '<div class="terms"><span>Query terms: ' +
                (system.terms.length ? system.terms.map(term => '<span class="badge">' + esc(term) + '</span>').join(' ')
                : '<span class="badge">none</span>') + 
                '</span></div>';

            const resultsHtml = system.empty
                ? '<div class="card">' +
                    (name === 'word2vec'
                        ? 'No usable query vector. The query might contain no in-vocabulary words.'
                        : 'No matching query terms produced a score.') +
                    ' No ranking is shown.</div>'
                : system.results.map(doc =>
                    '<article class="card">' +
                        '<div class="rank">' +
                            doc.rank.toString().padStart(2, '0') + '  ' +
                            (doc.relevance > 0 ? '<span class="badge">Relevant</span>' : '') +
                        '</div>' +
                        '<h4>' + esc(doc.title) + '</h4>' +
                        '<small>Document ' + esc(doc.id) + '</small>' +
                        '<br>' +
                        '<small>Score ' + doc.score.toFixed(3) + '</small>' +
                         '<br>' +
                        '<details>' +
                            '<summary>Read content</summary>' +
                            '<p>' + esc(doc.text) + '</p>' +
                        '</details>' +
                    '</article>'
                ).join('');

            const relevantHtml = data.query_id && !system.empty
                ? '<div class="relevance"><p>Relevant document ranks</p>' +
                    system.relevant_ranks.map(doc =>
                            '<span>' + (doc.rank?.toString().padStart(2, '0') ?? 'not returned') + '</span>' +
                            '<h4>' + esc(doc.title || doc.id) + '</h4>'
                        ).join('<br>') +
                    '</div>'
                : '';

            $(name).innerHTML = termsHtml + resultsHtml + relevantHtml;
        }

        $('output').hidden = false;
        $('status').textContent = 'Comparison ready after ' + data.milliseconds + ' ms';
    } catch (error) {
        $('status').textContent = error.message;
    } finally {
        $('submit').disabled = false;
        document.querySelectorAll('.presets button').forEach(button => {
            button.disabled = false;
        });
    }
}

$('search').addEventListener('submit', compare);

document.querySelectorAll('[data-query]').forEach(button => {
    button.addEventListener('click', () => {
        $('dataset').value = button.dataset.dataset;
        $('query').value = button.dataset.query;
        compare();
    });
});

$('wildcard').addEventListener('click', () => {
    $('query').value = '';
    $('query').focus();
    $('output').hidden = true;
    $('status').textContent = 'Waiting for a query...';
});

$('dataset').addEventListener('change', () => {
    $('output').hidden = true;
    $('status').textContent = 'Collection changed. Submit again to compare.';
});

get('/api/results')
    .then(data => {
        $('scores').innerHTML = Object.entries(data.runs[0].datasets)
            .map(([name, row]) => {
                const values = data.runs.map(run => run.datasets[name].ndcg10);
                const mean = values.reduce((a, b) => a + b) / values.length;
                const std = Math.sqrt(
                    values.reduce((a, b) => a + (b - mean) ** 2, 0) /
                    (values.length - 1)
                );

                return '<tr>' +
                    '<td>' + esc(name) + '</td>' +
                    '<td>' + row.n_queries.toLocaleString() + '</td>' +
                    '<td>' + mean.toFixed(3) + ' ± ' + std.toFixed(3) + '</td>' +
                    '<td>' + row.baseline.ndcg10.toFixed(3) + '</td>' +
                    '<td>' + data.minilm.datasets[name].baseline.ndcg10.toFixed(3) + '</td>' +
                    '</tr>';
            }).join('');
    })
    .catch(error => {
        $('scores').textContent = 'Saved results unavailable: ' + error.message;
    });
