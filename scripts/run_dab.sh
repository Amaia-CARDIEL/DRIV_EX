#!/bin/bash


# Usage: bash ./scripts/run_dab.sh         -> runs DAB
#        bash ./scripts/run_dab.sh adapt   -> runs DAB †

# Run DAB or DAB † (adapted) with paper config on best
# Llama-3-8B-Instruct driving LLM/Fluency Expert finetunings

if [ $# -eq 0 ]; then
    # DAB (Pynadath et al., Controlled LLM Decoding via
    # Discrete Auto-regressive Biasing, ICLR 2025)
    python ./driv_ex/cf_algorithms/dab_script.py \
        --LLM Meta-Llama-3-8B-Instruct \
        --per_device_train_batch_size 8 \
        --ckpt_nb 4400 \
        --ckpt_nb_x_vision 750 \
        --grad_cumul 8 \
        --iter 15 \
        --temperature 0.001 \
        --k_val 250 \
        --weight_val 5 \
        --add_Xo_reg_in_loss --xo_loss_weight 5

elif [ "$1" = "adapt" ]; then
    # DAB † (adapted by us to the task)
    python ./driv_ex/cf_algorithms/dab_script.py \
        --LLM Meta-Llama-3-8B-Instruct \
        --per_device_train_batch_size 8 \
        --ckpt_nb 4400 \
        --ckpt_nb_x_vision 750 \
        --grad_cumul 8 \
        --iter 15 \
        --temperature 0.001 \
        --k_val 250 \
        --weight_val 3 \
        --add_Xo_reg_in_bias --Xo_bias_weight_val 3

else
    echo "Usage: bash run_dab.sh [adapt]"
    exit 1
fi
