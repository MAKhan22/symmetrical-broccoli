from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.interactive import run_learn_session
from rwrt.learner import LearnerProfile
from rwrt.pipeline import RecommendationPipeline
from rwrt.vocabulary import VocabularyStore


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="rwrt — Right Word at the Right Time")
    p.add_argument(
        "--db",
        type=Path,
        default=Path("primary_graph.sqlite"),
        help="Path to word-frequency SQLite database",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/faiss"),
        help="Directory for FAISS index artifacts",
    )
    p.add_argument(
        "--min-weight",
        type=int,
        default=10,
        help="Minimum token frequency to include in vocabulary",
    )
    p.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Cap vocabulary size (most frequent words kept)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    return p


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bi-model",
        type=str,
        default=None,
        help="Sentence-transformers bi-encoder model name",
    )
    parser.add_argument(
        "--cross-model",
        type=str,
        default=None,
        help="Cross-encoder model name for reranking",
    )
    parser.add_argument(
        "--query-strategy",
        choices=("mean_embedding", "weighted_mean", "inverse_weighted_mean", "topic"),
        default=None,
        help="How to build the bi-encoder query vector",
    )
    parser.add_argument(
        "--max-query-words",
        type=int,
        default=None,
        help="Cap known words used in bi-encoder and cross-encoder queries",
    )
    parser.add_argument(
        "--frequency-boost",
        type=float,
        default=None,
        help="Add boost * log1p(freq) to bi-encoder scores before reranking",
    )
    parser.add_argument(
        "--query-word-selection",
        choices=("diverse", "topic"),
        default=None,
        help="How to subsample known words for queries (default: diverse)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic keyword for query_word_selection=topic",
    )


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        db_path=args.db,
        index_dir=args.index_dir,
        min_weight=args.min_weight,
        max_vocab_size=args.max_vocab,
    )
    if args.device:
        cfg.device = args.device
    if getattr(args, "bi_model", None):
        cfg.bi_model = args.bi_model
    if getattr(args, "cross_model", None):
        cfg.cross_model = args.cross_model
    if getattr(args, "query_strategy", None):
        cfg.query_strategy = args.query_strategy
    if getattr(args, "max_query_words", None) is not None:
        cfg.max_query_words = args.max_query_words
    if getattr(args, "frequency_boost", None) is not None:
        cfg.frequency_boost = args.frequency_boost
    if getattr(args, "query_word_selection", None):
        cfg.query_word_selection = args.query_word_selection
    if getattr(args, "topic", None):
        cfg.topic_keyword = args.topic
    if getattr(args, "retrieve_k", None) is not None:
        cfg.retrieve_k = args.retrieve_k
    if getattr(args, "return_n", None) is not None:
        cfg.return_n = args.return_n
    if getattr(args, "bi_only", False):
        cfg.use_cross_encoder = False
    cfg.resolve()
    return cfg


def build_index_main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    _add_pipeline_args(parser)
    parser.add_argument(
        "--no-save-embeddings",
        action="store_true",
        help="Skip writing embeddings.npy (saves disk space)",
    )
    args = parser.parse_args(argv)

    cfg = _config_from_args(args)

    vocab = VocabularyStore(cfg)
    vocab.load()
    print(f"Vocabulary size: {len(vocab):,} words (min_weight={cfg.min_weight})")

    index = EmbeddingIndex(cfg, vocab)
    index.build(save_embeddings=not args.no_save_embeddings)
    print(f"Index written to {cfg.index_dir}")
    return 0


def recommend_main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    _add_pipeline_args(parser)
    parser.add_argument(
        "known",
        nargs="*",
        help="Words the learner already knows",
    )
    parser.add_argument("-k", "--retrieve-k", type=int, default=200)
    parser.add_argument("-n", "--return-n", type=int, default=5)
    parser.add_argument(
        "--bi-only",
        action="store_true",
        help="Skip cross-encoder reranking",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="JSON learner profile to load/update",
    )
    parser.add_argument(
        "--no-save-profile",
        action="store_true",
        help="Do not write --profile after running",
    )
    args = parser.parse_args(argv)

    cfg = _config_from_args(args)

    vocab = VocabularyStore(cfg)
    vocab.load()

    if args.profile is not None:
        profile = LearnerProfile.load_or_create(args.profile, vocabulary=vocab)
        if args.known:
            added, rejected = profile.add_many(args.known)
            if rejected:
                print(f"Ignored (not in vocabulary): {', '.join(rejected)}", file=sys.stderr)
            if added:
                print(f"Added {added} word(s) to profile.", file=sys.stderr)
    else:
        profile = None
        if not args.known:
            print("Provide known words or --profile.", file=sys.stderr)
            return 1

    known = profile if profile is not None else set(args.known)
    if profile is not None and len(profile) == 0:
        print("Profile has no known words.", file=sys.stderr)
        return 1

    index = EmbeddingIndex(cfg, vocab)
    index.load()

    pipe = RecommendationPipeline(cfg, vocab, index=index)
    results = pipe.recommend(known, topic_keyword=cfg.topic_keyword)

    if args.profile is not None and not args.no_save_profile:
        profile.save(args.profile)

    known_set = profile.known if profile is not None else set(args.known)
    label = ", ".join(sorted(known_set))
    print(f"\nSuggestions ({len(known_set)} known): {label}\n")
    for i, c in enumerate(results, 1):
        bi = f"{c.bi_score:.4f}" if c.bi_score is not None else "—"
        cross = f"{c.cross_score:.4f}" if c.cross_score is not None else "—"
        freq = c.frequency if c.frequency is not None else "—"
        print(f"{i}. {c.word:20}  bi={bi}  cross={cross}  freq={freq}")
    return 0


def learn_main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    _add_pipeline_args(parser)
    parser.add_argument(
        "known",
        nargs="*",
        help="Optional words to seed the profile before the session",
    )
    parser.add_argument("-k", "--retrieve-k", type=int, default=200)
    parser.add_argument("-n", "--return-n", type=int, default=5)
    parser.add_argument(
        "--bi-only",
        action="store_true",
        help="Skip cross-encoder reranking",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="JSON learner profile to load/update (required)",
    )
    args = parser.parse_args(argv)

    cfg = _config_from_args(args)

    vocab = VocabularyStore(cfg)
    vocab.load()

    profile = LearnerProfile.load_or_create(args.profile, vocabulary=vocab)
    if args.known:
        added, rejected = profile.add_many(args.known)
        if rejected:
            print(f"Ignored (not in vocabulary): {', '.join(rejected)}", file=sys.stderr)
        if added:
            print(f"Added {added} word(s) to profile.", file=sys.stderr)

    if len(profile) == 0:
        print(
            "Profile has no known words. Seed with positional words or edit the profile.",
            file=sys.stderr,
        )
        return 1

    index = EmbeddingIndex(cfg, vocab)
    index.load()

    pipe = RecommendationPipeline(cfg, vocab, index=index)

    def save_profile() -> None:
        profile.save(args.profile)

    return run_learn_session(
        profile,
        pipe,
        return_n=cfg.return_n,
        on_save=save_profile,
    )
