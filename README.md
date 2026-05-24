# symmetrical-broccoli

**Right Word at the Right Time (rwrt)** — adaptive next-word recommendations for Turkish learners who already know some of the language. Given the words a learner knows, rwrt suggests semantically related words they have not learned yet, using bi-encoder retrieval over a FAISS index, optional cross-encoder reranking, and an optional OpenRouter-powered tutor that explains each word.

## How it works

```text
                         OFFLINE (once)
primary_graph.sqlite ──► VocabularyStore ──► EmbeddingIndex ──► data/faiss/
  (word frequencies)      (filter/cap)        (bi-encoder + FAISS)

                         ONLINE (each session)
LearnerProfile ──► BiEncoderRetriever ──► CrossEncoderReranker ──► suggestions
 (known words)      (FAISS search)         (optional rerank)

                         CHAT TUI (rwrt-chat)
pick a suggestion ──► OpenRouter LLM ──► EN/TR explanation + examples + Q&A
                   ──► word added to profile
```

**Recommendation pipeline**

1. Load the learner's known words (CLI args, a JSON profile, or `LearnerProfile` in Python).
2. **Subsample** known words for the query (`query_word_selection`) — a diverse mix of middle- and lower-frequency words so suggestions evolve as the learner grows, rather than staying anchored on their first high-frequency words.
3. **Build a query vector** (`query_strategy`) from those words, or directly from a topic keyword.
4. **Search FAISS** for semantically similar words the learner does not know.
5. Optionally **rerank** candidates with a cross-encoder and return the top *n* suggestions.

Each suggestion is a `Candidate` with bi-encoder score, optional cross-encoder score, and corpus frequency.

**Chat TUI** (`rwrt-chat`) wraps the same pipeline: you pick a suggestion, the LLM explains it in English and Turkish with example sentences, you can ask follow-up questions, then the word is saved to your profile.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Data** — place `primary_graph.sqlite` in the project root (gitignored if large). Built from the OpenSubtitles Turkish corpus; contains a `nodes(word, weight)` table.

**OpenRouter** (for `rwrt-chat` only):

```bash
cp .env.example .env
# edit .env → OPENROUTER_API_KEY=sk-or-v1-...
```

**Tests:**

```bash
pytest
```

## Quick start

### 1. Build the FAISS index (one-time)

GPU recommended for large vocabularies. Downloads the bi-encoder on first run.

```bash
rwrt-build-index --min-weight 10 --max-vocab 50000
```

Artifacts are written to `data/faiss/` (`index.faiss`, `words.txt`, `meta.json`, optional `embeddings.npy`).

### 2. Start the chat TUI (recommended)

```bash
rwrt-chat --profile data/learner.json merhaba teşekkür evet
```

| Key | Action |
|-----|--------|
| `4` | Study suggestion 4 — LLM explains the word |
| `1k, 2k, 3k, 4` | Mark 1–3 known immediately; study 4 |
| `1k, 2k, 3k` | Mark 1–3 known only (skip LLM) |
| Enter (after explanation) | Mark studied word learned, save profile |
| ask a question | Follow-up chat about the word |
| `b` | Back to suggestions without learning |
| `r` | Refresh suggestions |
| `q` | Quit |

### 3. Other ways to get suggestions

**One-shot** — no profile, print and exit:

```bash
rwrt-recommend merhaba teşekkür evet kitap -n 5
```

**Profile, no LLM** — pick words in a simple loop:

```bash
rwrt-learn --profile data/learner.json merhaba evet
```

## Commands

| Command | Purpose |
|---------|---------|
| `rwrt-build-index` | Embed vocabulary → FAISS index on disk |
| `rwrt-recommend` | Print top-*n* suggestions (one-shot or profile) |
| `rwrt-learn` | Interactive pick-and-learn loop (no LLM) |
| `rwrt-chat` | Full TUI: recommendations + OpenRouter tutor |

Equivalent wrappers live in `scripts/` (`build_index.py`, `recommend.py`, `learn.py`, `chat.py`).

### Which command?

| Goal | Command |
|------|---------|
| First-time setup | `rwrt-build-index` |
| Daily learning with explanations | `rwrt-chat` |
| Quick suggestion check | `rwrt-recommend` |
| Add words without API calls | `rwrt-learn` |
| Benchmark / ablation | `alpha.ipynb` |

### Shared pipeline flags

Available on `rwrt-recommend`, `rwrt-learn`, and `rwrt-chat` (and partially on `rwrt-build-index`):

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `primary_graph.sqlite` | Word-frequency SQLite database |
| `--index-dir` | `data/faiss` | FAISS artifact directory |
| `--min-weight` | `10` | Minimum corpus frequency |
| `--max-vocab` | none | Cap vocabulary size |
| `--device` | auto | `cuda` or `cpu` |
| `--bi-model` | see below | Bi-encoder model |
| `--cross-model` | see below | Cross-encoder model |
| `--query-strategy` | `mean_embedding` | How to build the query vector |
| `--query-word-selection` | `diverse` | How to subsample known words |
| `--max-query-words` | `64` | Max known words in a query |
| `--topic` | none | Topic keyword (for topic modes) |
| `--frequency-boost` | `0.0` | `boost × log1p(freq)` added to bi scores |

### Per-command flags

**`rwrt-build-index`**

| Flag | Description |
|------|-------------|
| `--no-save-embeddings` | Skip `embeddings.npy` (saves disk) |

**`rwrt-recommend`**

| Flag | Default | Description |
|------|---------|-------------|
| `known` | — | Positional known words |
| `-k`, `--retrieve-k` | `200` | FAISS candidates before reranking |
| `-n`, `--return-n` | `5` | Final suggestions |
| `--bi-only` | off | Skip cross-encoder |
| `--profile` | none | Load/update JSON profile |
| `--no-save-profile` | off | Don't write profile after run |

**`rwrt-learn`** — same as `rwrt-recommend`, but `--profile` is required and the profile is saved after each learned word.

**`rwrt-chat`** — same as `rwrt-learn`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--env-file` | `.env` | Path to OpenRouter env file |

### OpenRouter environment

| Variable | Required | Default |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | yes | — |
| `OPENROUTER_MODEL` | no | `openrouter/free` |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` |
| `OPENROUTER_TIMEOUT_S` | no | `60` |
| `OPENROUTER_APP_TITLE` | no | `rwrt` |

### Examples

```bash
# Chat with topic-guided retrieval
rwrt-chat --profile data/learner.json --query-strategy topic --topic yemek

# Emphasize rarer known words in the query vector
rwrt-recommend --profile data/learner.json --query-strategy inverse_weighted_mean

# Fast bi-encoder-only suggestions
rwrt-recommend merhaba evet -n 10 --bi-only -k 500

# Rebuild index after changing min_weight or bi_model
rwrt-build-index --min-weight 10 --max-vocab 50000 --bi-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

## Configuration

### Defaults (`PipelineConfig`)

| Field | Default |
|-------|---------|
| `bi_model` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `cross_model` | `dbmdz/bert-base-turkish-cased` |
| `retrieve_k` | `200` |
| `return_n` | `5` |
| `use_cross_encoder` | `True` |
| `max_query_words` | `64` |
| `query_strategy` | `mean_embedding` |
| `query_word_selection` | `diverse` |

### Query strategies (`query_strategy`)

How the bi-encoder **query vector** is built:

| Strategy | Behavior |
|----------|----------|
| `mean_embedding` | Uniform mean of subsampled known-word embeddings |
| `weighted_mean` | Frequency-weighted mean (high-freq known words dominate) |
| `inverse_weighted_mean` | Inverse-frequency weights (rarer known words dominate) |
| `topic` | Embed `--topic` directly; skips word aggregation |

### Query word selection (`query_word_selection`)

Which known words are subsampled before aggregation (skipped when `query_strategy=topic`):

| Selection | Behavior |
|-----------|----------|
| `diverse` | Mix of middle + lower frequency known words + small random sample |
| `topic` | Known words closest to the topic keyword in embedding space |

`query_strategy=topic` searches FAISS near the topic. `query_word_selection=topic` picks known words near the topic to build a mean query. They can be combined with `--topic yemek`.

## Python API

```python
from rwrt import LearnerProfile, PipelineConfig, RecommendationPipeline, VocabularyStore

cfg = PipelineConfig(min_weight=10, max_vocab_size=50_000).resolve()
vocab = VocabularyStore(cfg)
vocab.load()

profile = LearnerProfile.load_or_create("data/learner.json", vocabulary=vocab)
profile.learn("merhaba")

pipe = RecommendationPipeline(cfg, vocab)
pipe.index.load()

suggestions = pipe.recommend(profile, return_n=5)
suggestions = pipe.recommend(
    profile,
    use_cross_encoder=False,
    topic_keyword="yemek",
)
profile.save()
```

**Index build:**

```python
pipe.index.build(save_embeddings=True)
```

**LLM tutor:**

```python
from rwrt.llm import LLMConfig, OpenRouterClient

llm = OpenRouterClient(LLMConfig.from_env())
explanation = llm.explain_word("merhaba", known_words=profile.known)
answer = llm.follow_up("merhaba", [{"role": "assistant", "content": explanation}], "Is it formal?")
```

**Chat session programmatically:**

```python
from rwrt.llm import LLMConfig, OpenRouterClient
from rwrt.tui import run_chat_session

run_chat_session(profile, pipe, OpenRouterClient(LLMConfig.from_env()), return_n=5)
```

## Data and artifacts

**Input:** `primary_graph.sqlite` — `nodes(word, weight)`

**Generated** (`data/faiss/` by default):

| File | Description |
|------|-------------|
| `index.faiss` | FAISS inner-product index |
| `words.txt` | Vocabulary aligned with index rows |
| `embeddings.npy` | Optional embedding cache |
| `meta.json` | Build metadata — validated on load |

Rebuild the index if you change `min_weight`, `max_vocab`, or `bi_model`. A mismatch raises a clear error pointing to `rwrt-build-index`.

**Learner profile** (`data/learner.json`):

```json
{
  "version": 1,
  "known": ["merhaba", "evet", "teşekkür"]
}
```

## Package layout

```text
rwrt/
  config.py         PipelineConfig
  vocabulary.py     VocabularyStore (SQLite)
  encoders.py       lazy bi- / cross-encoder loaders
  index.py          EmbeddingIndex (FAISS)
  query.py          known-word subsampling (diverse / topic)
  retrieve.py       BiEncoderRetriever
  rerank.py         CrossEncoderReranker
  learner.py        LearnerProfile
  pipeline.py       RecommendationPipeline
  interactive.py    rwrt-learn loop
  llm.py            OpenRouter client
  tui.py            rwrt-chat TUI
  cli.py            all entry points

scripts/            CLI wrappers
tests/              pytest (77 tests)
alpha.ipynb         benchmark notebook
.env.example        OpenRouter template
```

## Benchmark notebook

`alpha.ipynb` simulates learner profiles (top-*N* known words, next frequency decile as targets) and compares:

| Method | Description |
|--------|-------------|
| `frequency` | Top-frequency unknown words |
| `bi_only` | Bi-encoder + FAISS |
| `bi_cross` | Bi-encoder + cross-encoder rerank |

Metrics: MRR and Recall@10. Results → `data/faiss_alpha/benchmark_results.json`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FAISS index not found` | Run `rwrt-build-index` first |
| `Index metadata does not match` | Rebuild index with current `--min-weight` / `--max-vocab` / `--bi-model` |
| `OPENROUTER_API_KEY is not set` | Copy `.env.example` → `.env` and add your key |
| `None of the known words appear in vocabulary` | Check spelling; word may be below `--min-weight` cutoff |
| `Ignored (not in vocabulary)` | Word not in corpus — try a different spelling or lower `--min-weight` |
| Slow first run | Models download on first use; use `--device cuda` if available |
| CUDA driver warning | Falls back to CPU automatically |
