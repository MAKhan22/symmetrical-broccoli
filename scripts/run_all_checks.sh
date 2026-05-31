#!/usr/bin/env bash
# End-to-end checks for rwrt demo / CI. Run from repo root:
#   bash scripts/run_all_checks.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-}"
BIN_DIR=""
if [[ -z "$PY" ]]; then
  if [[ -x "$HOME/.pyenv/versions/3.12.7/envs/global_main_312/bin/python" ]]; then
    PY="$HOME/.pyenv/versions/3.12.7/envs/global_main_312/bin/python"
    BIN_DIR="$HOME/.pyenv/versions/3.12.7/envs/global_main_312/bin"
  else
    PY="$(command -v python3)"
    BIN_DIR="$(dirname "$PY")"
  fi
else
  BIN_DIR="$(dirname "$PY")"
fi
RWRT_RECOMMEND="${BIN_DIR}/rwrt-recommend"
RWRT_LEARN="${BIN_DIR}/rwrt-learn"
RWRT_CHAT="${BIN_DIR}/rwrt-chat"

MPNET_DIR="data/faiss/sentence_transformers_paraphrase_multilingual_mpnet_base_v2"
DISTIL_DIR="data/faiss/sentence_transformers_distiluse_base_multilingual_cased_v2"
E5_DIR="data/faiss/intfloat_multilingual_e5_small"
COMMON=(--min-weight 10 --max-vocab 50000)
PROFILE="data/learner.json"
PROFILE_CHAT="data/learner_chat_test.json"
rm -f "$PROFILE_CHAT"
PASS=0
FAIL=0
SKIP=0

log() { printf '\n=== %s ===\n' "$1"; }
ok()  { PASS=$((PASS + 1)); echo "[PASS] $1"; }
bad() { FAIL=$((FAIL + 1)); echo "[FAIL] $1"; }
skip(){ SKIP=$((SKIP + 1)); echo "[SKIP] $1"; }

run() {
  local name="$1"
  shift
  log "$name"
  if "$@"; then
    ok "$name"
    return 0
  else
    bad "$name"
    return 1
  fi
}

# --- unit tests ---
run "pytest (full suite)" "$PY" -m pytest tests/ -q --tb=no || true

# --- data layout ---
log "FAISS index layout"
for d in "$MPNET_DIR" "$DISTIL_DIR" "$E5_DIR"; do
  if [[ -f "$d/index.faiss" && -f "$d/meta.json" ]]; then
    ok "index present: $d"
  else
    bad "index missing: $d"
  fi
done
if [[ -f data/faiss/index.faiss ]]; then
  ok "flat data/faiss/index.faiss exists"
else
  skip "flat data/faiss/index.faiss (use per-model subdirs)"
fi

if [[ -f primary_graph.sqlite ]]; then
  ok "primary_graph.sqlite present"
else
  bad "primary_graph.sqlite missing"
fi

# --- rwrt-recommend (all encoders) ---
run "rwrt-recommend mpnet" \
  "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$MPNET_DIR" \
  merhaba teşekkür evet -n 3

run "rwrt-recommend distiluse" \
  "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$DISTIL_DIR" \
  --bi-model sentence-transformers/distiluse-base-multilingual-cased-v2 \
  merhaba teşekkür evet -n 3

run "rwrt-recommend e5-small" \
  "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$E5_DIR" \
  --bi-model intfloat/multilingual-e5-small \
  merhaba teşekkür evet -n 3

run "rwrt-recommend + profile" \
  "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$MPNET_DIR" \
  --profile "$PROFILE" merhaba teşekkür evet -n 3

run "rwrt-recommend bi-only" \
  "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$MPNET_DIR" --bi-only \
  merhaba evet -n 3

# cross-encoder is slow; optional
log "rwrt-recommend cross_encoder (may take 1–2 min)"
if "$RWRT_RECOMMEND" "${COMMON[@]}" --index-dir "$MPNET_DIR" \
  --reranker-type cross_encoder merhaba evet -n 2 2>&1 | tail -5; then
  ok "rwrt-recommend cross_encoder"
else
  bad "rwrt-recommend cross_encoder"
fi

# --- script wrappers ---
run "scripts/recommend.py" \
  "$PY" scripts/recommend.py \
  "${COMMON[@]}" --index-dir "$MPNET_DIR" \
  merhaba evet -n 2

# --- learn TUI ---
log "rwrt-learn (pick 1, quit)"
if printf '1\nq\n' | "$RWRT_LEARN" \
  "${COMMON[@]}" --index-dir "$MPNET_DIR" \
  --profile "$PROFILE" merhaba teşekkür evet 2>&1 | grep -q "Learned:"; then
  ok "rwrt-learn pick word"
else
  bad "rwrt-learn pick word"
fi

# --- chat TUI (needs OPENROUTER_API_KEY in .env) ---
if [[ -f .env ]] && grep -q 'OPENROUTER_API_KEY=.' .env 2>/dev/null; then
  log "rwrt-chat (study word 1, mark learned, quit)"
  CHAT_LOG="$(mktemp)"
  printf '1\n\nq\n' | "$RWRT_CHAT" \
    "${COMMON[@]}" --index-dir "$MPNET_DIR" \
    --profile "$PROFILE_CHAT" merhaba teşekkür evet >"$CHAT_LOG" 2>&1 || true
  if grep -qE 'Added to profile|already known' "$CHAT_LOG"; then
    ok "rwrt-chat LLM study flow"
  else
    bad "rwrt-chat LLM study flow"
    tail -8 "$CHAT_LOG" >&2 || true
  fi
  rm -f "$CHAT_LOG"

  rm -f "$PROFILE_CHAT"
  log "rwrt-chat (1k mark known, quit)"
  CHAT_LOG="$(mktemp)"
  printf '1k\nq\n' | "$RWRT_CHAT" \
    "${COMMON[@]}" --index-dir "$MPNET_DIR" \
    --profile "$PROFILE_CHAT" merhaba teşekkür evet >"$CHAT_LOG" 2>&1 || true
  if grep -q 'Marked known' "$CHAT_LOG"; then
    ok "rwrt-chat mark known (1k)"
  else
    bad "rwrt-chat mark known (1k)"
    tail -8 "$CHAT_LOG" >&2 || true
  fi
  rm -f "$CHAT_LOG"
  rm -f "$PROFILE_CHAT"
else
  skip "rwrt-chat live LLM (.env / OPENROUTER_API_KEY)"
fi

# --- README default path (expected fail unless flat index built) ---
log "README default index-dir (informational)"
if "$RWRT_RECOMMEND" merhaba evet -n 1 --max-vocab 50000 2>/dev/null; then
  ok "README flat index path works"
else
  skip "README flat index path (use --index-dir $MPNET_DIR)"
fi

printf '\n========== SUMMARY ==========\n'
echo "PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
[[ "$FAIL" -eq 0 ]]
