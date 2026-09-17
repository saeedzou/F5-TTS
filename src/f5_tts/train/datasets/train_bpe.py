"""
Train a SentencePiece BPE tokenizer from a JSONL manifest.

Usage:
    python train_bpe.py /path/to/manifest.jsonl /path/to/tokenizer \
        --vocab-size 4096

The manifest must contain a ``text`` field on each non-empty JSON line. Other
manifest fields are ignored.
"""

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def read_manifest_texts(manifest_path):
    manifest_path = Path(manifest_path).expanduser().absolute()
    if manifest_path.suffix.lower() != ".jsonl":
        raise ValueError(f"manifest must be a .jsonl file: {manifest_path}")

    texts = []
    with manifest_path.open("r", encoding="utf-8-sig") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(item, dict) or "text" not in item:
                raise ValueError(f"Missing text field on line {line_number}")

            text = str(item["text"])
            if text:
                texts.append(text)

    if not texts:
        raise RuntimeError("No non-empty text entries found in manifest.")
    return texts


def get_args():
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer from a JSONL manifest.")
    parser.add_argument("manifest_path", type=str, help="JSONL manifest containing a text field.")
    parser.add_argument(
        "model_prefix",
        type=str,
        help="Output prefix for the SentencePiece model and vocabulary files.",
    )
    parser.add_argument("--vocab-size", type=int, default=4096, help="Vocabulary size (default: 4096).")
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Character coverage for the tokenizer (default: 1.0).",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=4192,
        help="Maximum number of characters per training sentence (default: 4192).",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Maximum number of sentences to use; 0 uses all sentences (default: 0).",
    )
    parser.add_argument(
        "--user-defined-symbol",
        action="append",
        default=[],
        help="Symbol to reserve as a token; repeat for multiple symbols.",
    )
    parser.add_argument(
        "--control-symbol",
        action="append",
        default=[],
        help="Control symbol to reserve; repeat for multiple symbols.",
    )
    parser.add_argument("--hard-vocab-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-input-sentence", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = get_args()
    texts = read_manifest_texts(args.manifest_path)

    model_prefix = Path(args.model_prefix).expanduser()
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=model_prefix.as_posix(),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        normalization_rule_name="identity",
        byte_fallback=False,
        max_sentence_length=args.max_sentence_length,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,
        hard_vocab_limit=args.hard_vocab_limit,
        user_defined_symbols=args.user_defined_symbol,
        control_symbols=args.control_symbol,
    )

    print(f"Trained BPE tokenizer on {len(texts)} text entries.")
    print(f"Model: {model_prefix}.model")
    print(f"Vocabulary: {model_prefix}.vocab")


if __name__ == "__main__":
    main()