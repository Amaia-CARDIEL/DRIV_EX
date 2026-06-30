#!/bin/bash


# Usage: bash ./scripts/run_drivex.sh

# Run DRIV-EX with paper config on best
# Llama-3-8B-Instruct driving LLM/Fluency Expert finetunings
python ./driv_ex/cf_algorithms/main_script.py \
    --LLM Meta-Llama-3-8B-Instruct \
	--per_device_train_batch_size 8 \
	--ckpt_nb 4400 \
	--ckpt_nb_x_vision 750 \
	--grad_cumul 8 \
    --iter 15 \
    --dab_reg \
    --lr 0.0075 \
    --weight_val 8 \
    --add_Xo_reg_in_bias --Xo_bias_weight_val 3 \
    --add_Xo_reg_in_loss --xo_loss_weight 7 \
    --proj_subvoc --subvoc_size 50 \
    --path paper_config


# # Run DRIV-EX with paper config on best
# # Mistral-7B-Instruct driving LLM/Fluency Expert finetunings
# python ./driv_ex/cf_algorithms/main_script.py \
#   --LLM mistralai/Mistral-7B-Instruct-v0.3 \
#  	--per_device_train_batch_size 8 \
# 	--ckpt_nb 1100 \
# 	--ckpt_nb_x_vision 500 \
# 	--grad_cumul 32 \
#   --iter 15 \
#   --dab_reg \
#   --lr 0.0075 \
#   --weight_val 8 \
#   --add_Xo_reg_in_bias --Xo_bias_weight_val 3 \
#   --add_Xo_reg_in_loss --xo_loss_weight 7 \
#   --proj_subvoc --subvoc_size 50 \
#   --path paper_config


# # Run DRIV-EX with paper config on best
# # Qwen2.5-7B-Instruct driving LLM/Fluency Expert finetunings
# python ./driv_ex/cf_algorithms/main_script.py \
#   --LLM Qwen/Qwen2.5-7B-Instruct \
# 	--per_device_train_batch_size 8 \
# 	--ckpt_nb 8000 \
# 	--ckpt_nb_x_vision 800 \
# 	--grad_cumul 8 \
#   --iter 15 \
#   --dab_reg \
#   --lr 0.0075 \
#   --weight_val 8 \
#   --add_Xo_reg_in_bias --Xo_bias_weight_val 3 \
#   --add_Xo_reg_in_loss --xo_loss_weight 7 \
#   --proj_subvoc --subvoc_size 50 \
#   --path paper_config