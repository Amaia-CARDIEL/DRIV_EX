# Copyright (c) 2026 Valeo. All rights reserved.
"""Argument parser for evaluation_script.py: LLM checkpoint evaluation on the highD
validation set (0-shot baseline + fine-tuned checkpoints, single or sweep)."""
import argparse


def get_args(notebook=False):

    parser = argparse.ArgumentParser(
        description="Args to evaluate a driving LLM on the highD validation set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model identity ---

    parser.add_argument(
        "--HF_name", "--n", "--LLM",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name to evaluate",
    )

    parser.add_argument(
        "--eval_quantization",
        type=str,
        default="8bit",
        help="Quantization to use when loading the model",
    )

    parser.add_argument(
        "--FT_type",
        type=str,
        default="classic_FT",
        choices=["classic_FT", "digit_bias_FT", "vehicle_bias_FT"],
        help="Type of fine-tuning to choose checkpoints from",
    )

    parser.add_argument(
        "--bias_type",
        type=str,
        default=None,
        choices=["vehicle_bias", "digit_bias", "just_vy_in_out", "vy_in_out"],
        help="Bias type used when fine-tuning, to locate the correct checkpoint folder",
    )

    parser.add_argument(
        "--reverse_bias",
        action="store_true",
        default=False,
        help="Whether to use the reverse-bias variant of the fine-tuned checkpoint",
    )

    # --- Checkpoint location (long internal naming) ---

    parser.add_argument(
        "--FT_learning_rate",
        type=float,
        default=0.0005,
        help="Fine-tuning learning rate (used to reconstruct the checkpoint folder name)",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device train batch size (used to reconstruct the checkpoint folder name)",
    )

    parser.add_argument(
        "--grad_cumul", "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (used to reconstruct the checkpoint folder name)",
    )

    parser.add_argument(
        "--lora_r", "--r",
        type=int,
        default=64,
        help="LoRA rank r (used to reconstruct the checkpoint folder name when non-default)",
    )

    parser.add_argument(
        "--lora_alpha", "--a",
        type=float,
        default=16,
        help="LoRA alpha (used to reconstruct the checkpoint folder name when non-default)",
    )

    parser.add_argument(
        "--ckpt_nb",
        type=int,
        default=4400,
        help="Checkpoint step number to evaluate (for single-checkpoint eval)",
    )

    # --- Checkpoint sweep controls ---

    parser.add_argument(
        "--ckpt_multiple",
        "--ckpt_mult",
        type=int,
        default=None,
        help="Eval only checkpoints whose step is a multiple of this value",
    )

    parser.add_argument(
        "--exact_checkpt_list",
        type=int,
        nargs="+",
        default=None,
        help="Exact list of checkpoint step numbers to evaluate",
    )

    parser.add_argument(
        "--last_checkpt",
        action="store_true",
        default=False,
        help="Evaluate only the last checkpoint in the folder",
    )

    parser.add_argument(
        "--min_checkpt",
        type=int,
        default=0,
        help="Minimum checkpoint step to include in the sweep",
    )

    parser.add_argument(
        "--max_checkpt",
        type=int,
        default=5000,
        help="Maximum checkpoint step to include in the sweep",
    )

    # --- Eval scope ---

    parser.add_argument(
        "--dont_skip_0_shot_eval",
        action="store_true",
        default=False,
        help="Also evaluate the 0-shot (non-fine-tuned) base model",
    )

    parser.add_argument(
        "--only_0_shot",
        action="store_true",
        default=False,
        help="Evaluate only the 0-shot base model, skip fine-tuned checkpoints",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        default=False,
        help="Run on a single mini-batch for quick debugging",
    )

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args
