"""
Prepare a dataset from a JSONL manifest.

Usage:
    python prepare_manifest.py /path/to/manifest.jsonl /path/to/output [--pretrain]

Each manifest line must contain:
    {"audio_filepath": "/absolute/path/audio.wav", "duration": 1.23, "text": "Hello."}
"""

import argparse
import json
import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path

from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


sys.path.append(os.getcwd())


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")


def read_manifest(manifest_path):
    manifest_path = Path(manifest_path).expanduser().absolute()
    if manifest_path.suffix.lower() != ".jsonl":
        raise ValueError(f"manifest must be a .jsonl file: {manifest_path}")

    result = []
    durations = []
    text_vocab_set = set()

    with manifest_path.open("r", encoding="utf-8-sig") as manifest_file:
        for line_number, line in enumerate(tqdm(manifest_file, desc="Reading manifest ..."), start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            missing_fields = {"audio_filepath", "duration", "text"} - item.keys()
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Missing field(s) on line {line_number}: {missing}")

            audio_path = Path(str(item["audio_filepath"])).expanduser()
            if not audio_path.is_absolute():
                raise ValueError(
                    f"audio_filepath must be an absolute path (line {line_number}): {audio_path}"
                )
            if not audio_path.is_file():
                print(f"Warning: audio file not found on line {line_number}, skipping: {audio_path}")
                continue

            try:
                duration = float(item["duration"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"duration must be numeric (line {line_number})") from exc
            if duration <= 0:
                print(f"Warning: non-positive duration on line {line_number}, skipping")
                continue

            text = str(item["text"]).strip()
            result.append({"audio_path": audio_path.as_posix(), "text": text, "duration": duration})
            durations.append(duration)
            text_vocab_set.update(text)

    if not result:
        raise RuntimeError("No valid entries found in manifest.")

    return result, durations, text_vocab_set


def save_prepped_dataset(out_dir, result, durations, text_vocab_set, is_pretrain):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    with ArrowWriter(path=(out_dir / "raw.arrow").as_posix()) as writer:
        for item in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(item)
        writer.finalize()

    with (out_dir / "duration.json").open("w", encoding="utf-8") as duration_file:
        json.dump({"duration": durations}, duration_file, ensure_ascii=False)

    vocab_path = out_dir / "vocab.txt"
    if is_pretrain:
        with vocab_path.open("w", encoding="utf-8") as vocab_file:
            for character in sorted(text_vocab_set):
                vocab_file.write(character + "\n")
    else:
        shutil.copy2(PRETRAINED_VOCAB_PATH, vocab_path)

    print(f"For {out_dir.stem}, sample count: {len(result)}")
    print(f"For {out_dir.stem}, vocab size is: {len(text_vocab_set)}")
    print(f"For {out_dir.stem}, total {sum(durations) / 3600:.2f} hours")


def get_args():
    parser = argparse.ArgumentParser(description="Prepare and save a dataset from a JSONL manifest.")
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Input JSONL with audio_filepath, duration, and text fields.",
    )
    parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
    parser.add_argument("--pretrain", action="store_true", help="Build vocab.txt from the manifest text.")
    return parser.parse_args()


def main():
    args = get_args()
    result, durations, text_vocab_set = read_manifest(args.manifest_path)
    save_prepped_dataset(args.out_dir, result, durations, text_vocab_set, args.pretrain)


if __name__ == "__main__":
    main()