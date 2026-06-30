#!/bin/bash

# Usage:
#   bash compute_cf_score.sh --results_path <path> [--dab] [--BS_lb <float>]
#
# --results_path: path to the results folder to analyse.
#   - If an absolute path is given, it is used as-is.
#   - Otherwise the path is appended to REPO_DIR/results/cf_algo_results/LC_LLM/.
#   In both cases, the script reads JSON files from a best_results_json/ subfolder.
#   Example: --results_path main_results/my_exp
#            -> reads from REPO_DIR/results/cf_algo_results/LC_LLM/main_results/my_exp/best_results_json/
#
# --dab: set this flag if results come from a DAB-based algorithm (DAB, DAB†).
#        Omit for other algorithms (PEZ, PEZ†, GCG).
#
# --BS_lb: BertScore lower bound threshold for success criteria (default: 0.95).

python ./driv_ex/cf_scoring/compute_cf_score.py "$@"
