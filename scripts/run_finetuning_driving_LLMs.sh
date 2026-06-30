#!/bin/bash

# Fine-tuning Llama-3-8B-Instruct to be a Driving LLM
python ./driv_ex/llm_finetuning/finetuning_script.py \
    --LLM Meta-Llama-3-8B-Instruct \
    --output_dir ./LCLLM_ckpts/classic_FT \
    --train_files ./textual_driving_data/highD_by_lcllm/llama_train_surrounding_thought.json \
    --validation_files ./textual_driving_data/highD_by_lcllm/llama_val_surrounding_thought.json


# # Fine-tuning Mistral-7B-Instruct to be a Driving LLM
# python ./driv_ex/llm_finetuning/finetuning_script.py \
#     --LLM mistralai/Mistral-7B-Instruct-v0.3 \
#     --output_dir ./LCLLM_ckpts/classic_FT \
#     --train_files ./textual_driving_data/highD_by_lcllm/llama_train_surrounding_thought.json \
#     --validation_files ./textual_driving_data/highD_by_lcllm/llama_val_surrounding_thought.json \
#     --gradient_accumulation_steps 32


# # Fine-tuning Qwen2.5-7B-Instruct to be a Driving LLM
# python ./driv_ex/llm_finetuning/finetuning_script.py \
#     --LLM Qwen/Qwen2.5-7B-Instruct \
#     --output_dir ./LCLLM_ckpts/classic_FT \
#     --train_files ./textual_driving_data/highD_by_lcllm/llama_train_surrounding_thought.json \
#     --validation_files ./textual_driving_data/highD_by_lcllm/llama_val_surrounding_thought.json \
#     --epochs 4