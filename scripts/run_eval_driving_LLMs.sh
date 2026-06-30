#!/bin/bash

# Generate labelled val dataset if it does not yet exist
VAL_WITH_LABELS="./textual_driving_data/highD_by_lcllm/llama_val_surrounding_thought_with_labels.json"
if [ ! -f "$VAL_WITH_LABELS" ]; then
    echo "llama_val_surrounding_thought_with_labels.json does not exist => check README's 'Pre-processing textual highD' section"
    exit 1
fi

# ------------------------------------------
# ---- Eval driving Llama-3-8B-Instruct ----
# ------------------------------------------

# Evaluating Llama-3-8B-Instruct on full original highD val
python ./driv_ex/llm_evaluation/evaluation_script.py \
    --LLM Meta-Llama-3-8B-Instruct \
    --FT_type classic_FT \
    --exact_checkpt_list 4400

# Extraction of evaluations for safety-critical scenarios
python ./driv_ex/llm_evaluation/extract_eval_on_crash_subset.py \
    --LLM Meta-Llama-3-8B-Instruct \
    --FT_type classic_FT \
    --ckpt_nb 4400


# ------------------------------------------
# ---- Eval driving Mistral-7B-Instruct ----
# ------------------------------------------

# # Evaluating Mistral-7B-Instruct on full original highD val
# python ./driv_ex/llm_evaluation/evaluation_script.py \
#     --LLM mistralai/Mistral-7B-Instruct-v0.3 \
#     --FT_type classic_FT \
#     --gradient_accumulation_steps 32 \
#     --exact_checkpt_list 1100

# # Extraction of evaluations for safety-critical scenarios
# python ./driv_ex/llm_evaluation/extract_eval_on_crash_subset.py \
#     --LLM mistralai/Mistral-7B-Instruct-v0.3 \
#     --FT_type classic_FT \
#     --gradient_accumulation_steps 32 \
#     --ckpt_nb 1100


# ------------------------------------------
# ---- Eval driving Qwen2.5-7B-Instruct ----
# ------------------------------------------

# # Evaluating Qwen2.5-7B-Instruct on full original highD val
# python ./driv_ex/llm_evaluation/evaluation_script.py \
#     --LLM Qwen/Qwen2.5-7B-Instruct \
#     --FT_type classic_FT \
#     --exact_checkpt_list 8000

# # Extraction of evaluations for safety-critical scenarios
# python ./driv_ex/llm_evaluation/extract_eval_on_crash_subset.py \
#     --LLM Qwen/Qwen2.5-7B-Instruct \
#     --FT_type classic_FT \
#     --ckpt_nb 8000
