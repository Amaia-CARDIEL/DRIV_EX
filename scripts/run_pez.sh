#!/bin/bash


# Usage: bash ./scripts/run_pez.sh         -> runs PEZ
#        bash ./scripts/run_pez.sh adapt   -> runs PEZ †

# Run PEZ or PEZ † (adapted) with paper config on best
# Llama-3-8B-Instruct driving LLM/Fluency Expert finetunings

if [ $# -eq 0 ]; then
    # PEZ (Wen et al., Hard Prompts Made Easy: Gradient-Based Discrete
    # Optimization for Prompt Tuning and Discovery, NeurIPS 2023)
    python ./driv_ex/cf_algorithms/main_script.py \
        --LLM Meta-Llama-3-8B-Instruct \
        --per_device_train_batch_size 8 \
        --ckpt_nb 4400 \
        --ckpt_nb_x_vision 750 \
        --grad_cumul 8 \
        --iter 150 \
        --lr 0.005 \
        --add_Xo_reg_in_loss --xo_loss_weight 9 \
        --path PEZ_paper_config

elif [ "$1" = "adapt" ]; then
    # PEZ † (adapted by us to the task)
    python ./driv_ex/cf_algorithms/main_script.py \
        --LLM Meta-Llama-3-8B-Instruct \
        --per_device_train_batch_size 8 \
        --ckpt_nb 4400 \
        --ckpt_nb_x_vision 750 \
        --grad_cumul 8 \
        --iter 150 \
        --lr 0.005 \
        --add_Xo_reg_in_loss --xo_loss_weight 8 \
        --proj_subvoc --subvoc_size 25 \
        --path PEZ_adapt_paper_config

else
    echo "Usage: bash run_pez.sh [adapt]"
    exit 1
fi
