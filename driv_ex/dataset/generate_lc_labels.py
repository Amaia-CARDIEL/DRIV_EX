# Copyright (c) 2026 Valeo. All rights reserved.
"""Add ground-truth lane-change labels to a highD textual dataset.

Reads llama_{split}_surrounding_thought.json and produces
llama_{split}_surrounding_thought_with_labels.json, where each entry is the
original text enriched with an integer 'label' field (0=keep lane,
1=left lane change, 2=right lane change).

These labelled files are required by the DRIV-EX evaluation pipeline.
"""
import argparse
import json
from tqdm import tqdm

from driv_ex.dataset import DATA_DIR

HIGHD_DIR = DATA_DIR / "highD_by_lcllm"
INTENTION_PATTERNS = {
    0: 'Intention: "0: Keep lane"',
    1: 'Intention: "1: Left lane change"',
    2: 'Intention: "2: Right lane change"',
}


def extract_label(text: str) -> int:
    for label, pattern in INTENTION_PATTERNS.items():
        if pattern in text:
            return label
    raise ValueError(f"No intention label found in text: {text[:200]!r}")


def process_split(split: str, overwrite: bool) -> None:
    src_path = HIGHD_DIR / f"llama_{split}_surrounding_thought.json"
    dst_path = HIGHD_DIR / f"llama_{split}_surrounding_thought_with_labels.json"

    if dst_path.exists() and not overwrite:
        print(f"[{split}] {dst_path.name} already exists — skipping (use --overwrite to regenerate).")
        return

    print(f"[{split}] Loading {src_path.name} ...")
    with open(src_path) as f:
        src_data = json.load(f)
    print(f"[{split}] {len(src_data)} samples loaded.")

    output = []
    for entry in tqdm(src_data, desc=f"[{split}]"):
        text = entry["text"]
        output.append({"text": text, "label": extract_label(text)})

    assert len(output) == len(src_data), "Length mismatch after processing"

    print(f"[{split}] Saving {dst_path.name} ...")
    with open(dst_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"[{split}] Done. {len(output)} entries saved.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val"],
        default=["val"],
        help="Dataset split(s) to process (default: val).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for split in args.split:
        process_split(split, overwrite=args.overwrite)
