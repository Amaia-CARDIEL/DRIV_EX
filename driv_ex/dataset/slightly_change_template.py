# Copyright (c) 2026 Valeo. All rights reserved.
import argparse
import json
import os
from tqdm import tqdm
from driv_ex import REPO_DIR

DATA_DIR = REPO_DIR / "textual_driving_data" / "highD_by_lcllm"

X_ego_pattern = "The target vehicle is driving on"


def apply_template_fixes(text):
    start_id = text.find(X_ego_pattern)
    X_before_ego = text[:start_id]
    X_after_ego = text[start_id:]

    # misspelling / bad wording
    X_after_ego = X_after_ego.replace(
        "The information of target vehicle is as follow:",
        "The information about the target vehicle is as follows:")
    X_after_ego = X_after_ego.replace(
        "The information of its surrounding vehicles (with a range of 200m) are listed as follow:",
        "The information about its surrounding vehicles (within a range of 200 m) is listed as follows:")
    X_after_ego = X_after_ego.replace("Velocity(km/h)", "Velocity (km/h)")
    X_after_ego = X_after_ego.replace("Accelaration", "Acceleration")

    # coherence
    X_after_ego = X_after_ego.replace("vy = ", "vy=")
    X_after_ego = X_after_ego.replace("ax = ", "ax=")

    # lane positioning
    X_after_ego = X_after_ego.replace("rightmost", "right").replace("leftmost", "left")
    X_after_ego = X_after_ego.replace(" highway, located at the", " highway, in the")

    # vehicle type
    X_after_ego = X_after_ego.replace(" Car", " car").replace(" Truck", " truck")

    # vehicle position
    X_after_ego = X_after_ego.replace("  - Ahead:", "  - Front side:")
    X_after_ego = X_after_ego.replace("  - Behind:", "  - Back side:")
    X_after_ego = X_after_ego.replace("  - Left:", "  - Left side:")
    X_after_ego = X_after_ego.replace("  - Right:", "  - Right side:")

    X_after_ego = X_after_ego.replace("  - Notable feature: Ahead is ", "  - Notable feature: Front side is ")
    X_after_ego = X_after_ego.replace("  - Notable feature: Behind is ", "  - Notable feature: Back side is ")
    X_after_ego = X_after_ego.replace("  - Notable feature: Left is ", "  - Notable feature: Left side is ")
    X_after_ego = X_after_ego.replace("  - Notable feature: Right is ", "  - Notable feature: Right side is ")

    return X_before_ego + X_after_ego


def process_file(filename):
    filepath = DATA_DIR / filename
    print(f"\nProcessing {filename}")
    with open(filepath) as f:
        data = json.load(f)

    new_data = []
    for i in tqdm(range(len(data))):
        new_dico = dict(data[i])
        new_dico["text"] = apply_template_fixes(data[i]["text"])
        new_data.append(new_dico)

    with open(filepath, "w") as outfile:
        json.dump(new_data, outfile, indent=4)
    print(f"Saved {len(new_data)} entries to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits to process (default: train val)"
    )
    args = parser.parse_args()

    for split in args.splits:
        process_file(f"llama_{split}_surrounding_thought.json")
