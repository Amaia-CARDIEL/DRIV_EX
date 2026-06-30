#!/bin/bash


# Usage: bash ./scripts/run_gcg.sh

# Run GCG with paper config on best
# Llama-3-8B-Instruct driving LLM/Fluency Expert finetunings

python ./driv_ex/cf_algorithms/gcg_script.py \
    --LLM Meta-Llama-3-8B-Instruct \
    --per_device_train_batch_size 8 \
    --ckpt_nb 4400 \
    --ckpt_nb_x_vision 750 \
    --grad_cumul 8 \
    --iter 6 \
    --bf_verbose \
    --gcg_batch_size 512 \
    --gcg_topk 128 \
    --path paper_config