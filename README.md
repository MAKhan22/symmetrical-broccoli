# symmetrical-broccoli

**Right Word at the Right Time (rwrt)** — adaptive next-word recommendations for Turkish learners using bi-encoder retrieval and cross-encoder reranking.

## Package layout

```text
rwrt/
  config.py       # PipelineConfig
  vocabulary.py   # SQLite word-frequency store
  encoders.py     # Lazy bi- / cross-encoder loaders
  index.py        # FAISS embedding index (build / load / search)
  retrieve.py     # BiEncoderRetriever
  rerank.py       # CrossEncoderReranker
  learner.py      # LearnerProfile (known-word set, save/load)
  pipeline.py     # RecommendationPipeline
  cli.py          # Console entry points
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Place `primary_graph.sqlite` in the project root (gitignored if large).

## Usage

Build the FAISS index (one-time, GPU recommended):

```bash
rwrt-build-index --min-weight 10 --max-vocab 50000
```

Run recommendations:

```bash
rwrt-recommend merhaba teşekkür evet kitap -n 5

# Persistent learner profile
rwrt-recommend --profile data/learner.json merhaba evet
```

From Python:

```python
from rwrt import LearnerProfile, PipelineConfig, RecommendationPipeline, VocabularyStore

cfg = PipelineConfig(min_weight=10, max_vocab_size=50_000).resolve()
vocab = VocabularyStore(cfg)
vocab.load()

profile = LearnerProfile.load_or_create("data/learner.json", vocabulary=vocab)
profile.learn("merhaba")
profile.learn("teşekkür")

pipe = RecommendationPipeline(cfg, vocab)
pipe.index.load()
suggestions = pipe.recommend(profile)

profile.save()
```

## Data

- `primary_graph.sqlite` — `nodes(word, weight)` from OpenSubtitles processing
- `data/faiss/` — built index artifacts (`index.faiss`, `words.txt`, optional `embeddings.npy`)
