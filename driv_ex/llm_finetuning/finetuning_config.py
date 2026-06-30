# Copyright (c) 2026 Valeo. All rights reserved.
import argparse


def get_args(notebook=False):

    parser = argparse.ArgumentParser(
        description="Args for fine-tuning an LLM with LoRA on the textual highD dataset, either as a Driving LLM or Fluency Expert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #################
    # LC-LLM fine-tuning hyperparameters (Table 1 of their paper)
    #################

    parser.add_argument(
        "--FT_learning_rate", "--lr",
        type=float,
        default=5e-4,
        help="Fine-tuning learning rate"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device batch size for training",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Per-device batch size for evaluation",
    )

    parser.add_argument(
        "--num_train_epochs", "--epochs",
        type=int,
        default=2,  # 4 for Qwen
        help="Number of full passes over the training dataset",
    )

    parser.add_argument(
        "--lora_r", "--r",
        type=int,
        default=64,
        help="Rank param (r) when fine-tuning with LoRA",
    )

    parser.add_argument(
        "--lora_alpha", "--a",
        type=float,
        default=16,
        help="LoRA alpha scaling factor",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,  # 32 for Mistral
        help="Number of gradient accumulation steps before each optimizer update (effective batch size = per_device × grad_accum)",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=600,
        help="Number of linear learning-rate warmup steps at the start of training",
    )

    parser.add_argument(
        "--load_in_bits",
        type=int,
        default=8,
        help="Quantization precision in bits for loading the base model (4 or 8)",
    )

    # other fine-tuning hyperparameters

    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"./LCLLM_ckpts/classic_FT",
        help="Directory where fine-tuned checkpoints will be saved",
    )

    parser.add_argument(
        "--resume_from_checkpoint", "--resume",
        action="store_true",
        default=None,
        help="Resume training from the latest checkpoint in --output_dir",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of workers to use"
        )

    #################
    # Model arguments (model_args)
    #################

    parser.add_argument(
        "--HF_name", "--n", "--LLM",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="LLM to use",
    )

    parser.add_argument(
        "--HF_token_path",
        type=str,
        default="./driv_ex/LLM_token.json",
        help="Path to HuggingFace's token with proper rights to access chosen LLM (in working directory by default)",
    )

    parser.add_argument(
        "--main_folder",
        type=str,
        default=None,
        help="Override for the base directory (uses REPO_DIR by default)",
    )

    # parser.add_argument(
    #     "--cache_dir",
    #     type=str,
    #     default="./LLM_cache/",
    #     help="Where to store the pretrained models downloaded from huggingface",
    # )

    parser.add_argument(
        "--modules_to_FT",
        type=str,
        default="like_lcllm",
        choices=[None, "last_4_only", "lm_head_only", "like_lcllm"],
        help="Choice of the modules in the LLM to fine-tune with LoRA",
    )


    #################
    # Data training arguments (data_args)
    #################


    parser.add_argument(
        "--train_files",
        type=str,
        default="./textual_highD/full_sets/llama_train_surrounding_thought.json",
        help="Training dataset.",
    )

    parser.add_argument(
        "--validation_files",
        type=str,
        default="./textual_highD/full_sets/llama_val_surrounding_thought.json",
        help="Val dataset.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1000,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Max number of input tokens given to the LLM",
    )

    parser.add_argument(
        "--FT_on_inputs",
        action="store_true",
        default=False,
        help="Compute the loss on driving scene description, required for Fluency expert finetuning (default: loss computed on driving action tokens only to train a driving LLM)",
    )

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args