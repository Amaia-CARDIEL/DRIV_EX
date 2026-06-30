#!/bin/bash

# Generate input-prompt-only versions of the highD textual datasets (train + val).
# Output files are written to textual_driving_data/highD_by_lcllm/.
# Pass --overwrite to regenerate files that already exist.

python ./driv_ex/dataset/generate_input_only_highD_data.py \
    --split train val \
    "$@"
