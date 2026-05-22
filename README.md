# symmetrical-broccoli

**Right Word at the Right Time (rwrt)** — adaptive next-word recommendations for Turkish learners who already know some of the language. Given the words a learner knows, rwrt suggests semantically related words they have not learned yet, using bi-encoder retrieval over a FAISS index and optional cross-encoder reranking.

## How it works

The pipeline has two phases: an **offline index build** and an **online recommendation** loop.

```text
primary_graph.sqlite          OpenSubtitles word frequencies
        │
        ▼
  VocabularyStore             filter by min_weight / max_vocab
        │
        ▼
  EmbeddingIndex              bi-encoder → FAISS (index.faiss)
        │
        ▼
  BiEncoderRetriever            known words → query vector → top-k candidates
        │
        ▼
  CrossEncoderReranker          (optional) rerank candidates → top-n suggestions
```

**Recommendation flow**

1. Load the learner's known words (CLI args, a JSON profile, or a `LearnerProfile` in Python).
2. **Subsample** known words for the query (`query_word_selection`) so the query tracks recent learning rather than only the first high-frequency words the learner picked up.
3. **Build a query vector** from those words (`query_strategy`) — or directly from a topic keyword.
4. **Search FAISS** for semantically similar vocabulary words the learner does not know.
5. Optionally **rerank** the top candidates with a cross-encoder and return the best *n* suggestions.

Each suggestion is a `Candidate` with bi-encoder score, optional cross-encoder score, and corpus frequency.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Place `primary_graph.sqlite` in the project root (gitignored if large). This file is produced from the OpenSubtitles Turkish corpus and contains word-frequency data in a `nodes(word, weight)` table.

Run tests:

```bash
pytest
```

## Quick start

### 1. Build the FAISS index (one-time)

Embeds the vocabulary and writes index artifacts to `data/faiss/`. GPU is recommended for large vocabularies.

```bash
rwrt-build-index --min-weight 10 --max-vocab 50000
```

On first run this downloads the bi-encoder model (`paraphrase-multilingual-mpnet-base-v2`). A full 50k-word build may take several minutes depending on hardware.

### 2. Get suggestions

**One-shot** — pass known words directly:

```bash
rwrt-recommend merhaba teşekkür evet kitap -n 5
```

**With a persistent profile** — known words are saved to JSON and updated over time:

```bash
rwrt-recommend --profile data/learner.json merhaba evet
```

**Interactive session** — pick words to learn in a loop; the profile is saved after each selection:

```bash
rwrt-learn --profile data/learner.json merhaba evet
```

At the prompt, enter a number to mark a suggestion as learned, `r` to refresh, or `q` to quit.

## Commands

Three entry points are installed with the package:

| Command | Purpose |
|---------|---------|
| `rwrt-build-index` | Embed vocabulary and write FAISS index to disk |
| `rwrt-recommend` | Print top-*n* suggestions for a known-word set or profile |
| `rwrt-learn` | Interactive pick-and-learn session (requires `--profile`) |

Legacy script wrappers also exist under `scripts/` but the commands above are preferred.

### Shared flags

Available on all three commands:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `primary_graph.sqlite` | Word-frequency SQLite database |
| `--index-dir` | `data/faiss` | Directory for FAISS artifacts |
| `--min-weight` | `10` | Minimum corpus frequency to include a word |
| `--max-vocab` | none | Cap vocabulary size (most frequent words kept) |
| `--device` | auto | `cuda` or `cpu` |
| `--bi-model` | see config | Bi-encoder model name |
| `--cross-model` | see config | Cross-encoder model name |
| `--query-strategy` | `mean_embedding` | How to build the bi-encoder query vector |
| `--query-word-selection` | `diverse` | How to subsample known words for the query |
| `--max-query-words` | `64` | Max known words used in a query |
| `--topic` | none | Topic keyword (for topic-based modes) |
| `--frequency-boost` | `0.0` | Add `boost × log1p(freq)` to bi-encoder scores before reranking |

### `rwrt-build-index` flags

| Flag | Description |
|------|-------------|
| `--no-save-embeddings` | Skip writing `embeddings.npy` (saves disk space) |

### `rwrt-recommend` flags

| Flag | Default | Description |
|------|---------|-------------|
| `known` | — | Positional known words (Turkish tokens) |
| `-k`, `--retrieve-k` | `200` | Bi-encoder candidates to retrieve before reranking |
| `-n`, `--return-n` | `5` | Final suggestions to return |
| `--bi-only` | off | Skip cross-encoder reranking |
| `--profile` | none | Load/update a JSON learner profile |
| `--no-save-profile` | off | Do not write the profile after running |

### `rwrt-learn` flags

Same as `rwrt-recommend`, except `--profile` is **required** and there is no `--no-save-profile` (the profile is saved after each learned word).

### Examples

```bash
# Bi-encoder only, larger candidate pool
rwrt-recommend merhaba evet -n 10 --bi-only -k 500

# Inverse-frequency query weighting (emphasize rarer known words)
rwrt-recommend --profile data/learner.json --query-strategy inverse_weighted_mean

# Topic-guided retrieval — search near "food" in embedding space
rwrt-recommend --profile data/learner.json --query-strategy topic --topic yemek

# Topic-guided word selection for cross-encoder context
rwrt-recommend --profile data/learner.json \
  --query-word-selection topic --topic seyahat
```

## Python API

```python
from rwrt import LearnerProfile, PipelineConfig, RecommendationPipeline, VocabularyStore

cfg = PipelineConfig(
    min_weight=10,
    max_vocab_size=50_000,
    query_strategy="mean_embedding",
    query_word_selection="diverse",
).resolve()

vocab = VocabularyStore(cfg)
vocab.load()

profile = LearnerProfile.load_or_create("data/learner.json", vocabulary=vocab)
profile.learn("merhaba")
profile.learn("teşekkür")

pipe = RecommendationPipeline(cfg, vocab)
pipe.index.load()                          # skip if index already loaded in memory

suggestions = pipe.recommend(profile)      # list[Candidate]
suggestions = pipe.recommend(
    profile,
    return_n=10,
    use_cross_encoder=False,               # bi-encoder only
    topic_keyword="yemek",                 # for query_strategy="topic"
)

profile.save()
```

Build the index from Python:

```python
index = pipe.index
index.build(save_embeddings=True)
```

## Configuration

`PipelineConfig` controls all pipeline behavior. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `bi_model` | `paraphrase-multilingual-mpnet-base-v2` | Sentence-transformers bi-encoder |
| `cross_model` | `dbmdz/bert-base-turkish-cased` | Cross-encoder for reranking |
| `retrieve_k` | `200` | Candidates retrieved from FAISS |
| `return_n` | `5` | Final suggestions returned |
| `use_cross_encoder` | `True` | Enable cross-encoder reranking |
| `max_query_words` | `64` | Cap on known words fed into queries |

### Query strategies (`query_strategy`)

Controls how the bi-encoder **query vector** is built:

| Strategy | Behavior |
|----------|----------|
| `mean_embedding` | Uniform mean of subsampled known-word embeddings |
| `weighted_mean` | Frequency-weighted mean (high-frequency known words dominate) |
| `inverse_weighted_mean` | Inverse-frequency weights (rarer known words dominate) |
| `topic` | Embed the topic keyword directly; skips word aggregation |

### Query word selection (`query_word_selection`)

Controls **which known words** are subsampled before aggregation (not used when `query_strategy="topic"`):

| Selection | Behavior |
|-----------|----------|
| `diverse` | Mix of middle- and lower-frequency known words plus a small random sample |
| `topic` | Known words whose embeddings are closest to the topic keyword |

Set `topic_keyword` when using either topic mode.

## Data and artifacts

**Input**

- `primary_graph.sqlite` — `nodes(word, weight)` table from OpenSubtitles processing

**Generated** (under `data/faiss/` by default)

| File | Description |
|------|-------------|
| `index.faiss` | FAISS inner-product index over word embeddings |
| `words.txt` | Vocabulary order aligned with the index |
| `embeddings.npy` | Optional raw embedding matrix (speeds up query building) |
| `meta.json` | Build metadata (model, min_weight, vocab size) — validated on load |

If you change `min_weight`, `max_vocab`, or `bi_model`, rebuild the index. Loading an mismatched index raises a clear error.

**Learner profiles**

JSON files with a `known` word list:

```json
{
  "version": 1,
  "known": ["merhaba", "evet", "teşekkür"]
}
```

## Package layout

```text
rwrt/
  __init__.py       # public API
  config.py         # PipelineConfig
  types.py          # Candidate
  vocabulary.py     # VocabularyStore (SQLite word frequencies)
  encoders.py       # lazy bi- / cross-encoder loaders
  index.py          # EmbeddingIndex (FAISS build / load / search)
  query.py          # known-word subsampling (diverse / topic selection)
  retrieve.py       # BiEncoderRetriever
  rerank.py         # CrossEncoderReranker
  learner.py        # LearnerProfile (known-word set, save/load)
  pipeline.py       # RecommendationPipeline
  interactive.py    # interactive learn loop
  cli.py            # rwrt-build-index / rwrt-recommend / rwrt-learn

scripts/            # thin wrappers around cli entry points
tests/              # pytest suite
alpha.ipynb         # benchmark notebook (MRR, Recall@10)
```

## Benchmark notebook

`alpha.ipynb` builds an index, simulates learner profiles (top-*N* known words with the next frequency decile as targets), and compares three baselines:

| Method | Description |
|--------|-------------|
| `frequency` | Top-frequency unknown words |
| `bi_only` | Bi-encoder + FAISS, no reranking |
| `bi_cross` | Bi-encoder retrieval + cross-encoder rerank |

Metrics: MRR and Recall@10. Results are saved to `data/faiss_alpha/benchmark_results.json`.

Tune `MAX_VOCAB`, `PROFILE_SIZES`, and `FORCE_REBUILD` at the top of the notebook before a full run.
