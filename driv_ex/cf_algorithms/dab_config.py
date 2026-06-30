# Copyright (c) 2026 Valeo. All rights reserved.
import argparse




def get_args(notebook=False):

    parser = argparse.ArgumentParser(
        description="Args for the DAB (Discrete Auto-regressive Biasing) algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model / checkpoint ---

    parser.add_argument(
        "--HF_name", "--n", "--LLM",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name of the driving LLM to explain",
    )

    parser.add_argument(
        "--FT_type",
        type=str,
        default="classic_FT",
        choices=["classic_FT", "digit_bias_FT", "vehicle_bias_FT"],
        help="Fine-tuning variant to select the driving LLM checkpoint",
    )

    parser.add_argument(
        "--eval_quantization",
        type=str,
        default="8bit",
        help="Quantization precision for loading the driving LLM (8bit, 4bit, or None)",
    )

    parser.add_argument(
        "--ckpt_nb",
        type=int,
        default=4400,
        help="Checkpoint step for the Y-generation (driving LLM) adapter",
    )

    parser.add_argument(
        "--ckpt_nb_x_vision",
        type=int,
        default=750,
        help="Checkpoint step for the X-vision (fluency) adapter",
    )

    parser.add_argument(
        "--FT_learning_rate",
        type=float,
        default=0.0005,
        help="Fine-tuning learning rate matching the checkpoint's training config",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per-device batch size matching the fine-tuned checkpoint's training config",
    )

    parser.add_argument(
        "--grad_cumul", "--gradient_accumulation_steps",
        type=int,
        default=8,  # 32 for Mistral
        help="Gradient accumulation steps matching the fine-tuned checkpoint's training config",
    )

    # --- Dataset / scenario selection ---

    parser.add_argument(
        "--dataset", "--data",
        type=str,
        default="lc_llm_highD",
        choices=["lc_llm_highD"],
        help="Dataset/benchmark to run the algorithm on",
    )

    parser.add_argument(
        "--bias_type",
        type=str,
        default=None,
        choices=["vehicle_bias", "digit_bias", "just_vy_in_out", "vy_in_out"],
        help="Use the desired bias variant on the validation data used in DAB",
    )

    parser.add_argument(
        "--reverse_bias",
        action="store_true",
        default=False,
        help="Use the reverse-bias variant on the validation data used in DAB",
    )

    parser.add_argument(
        "--label_choice_method",
        type=str,
        default="impose_crash",
        choices=["impose_crash", "avoid_crash", "rd_among_possible"],
        help="How to select the target counterfactual label: impose a crash-inducing label, avoid a crash label, or pick randomly among plausible ones",
    )

    parser.add_argument(
        "--max_data", "--max_samples",
        type=int,
        default=None,
        help="Maximum number of scenarios to process (processes all if None)",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=0,
        help="Index of the first scenario to process (for range-based parallelization)",
    )

    parser.add_argument(
        "--seed_xtg",
        type=int,
        default=10,
        help="Random seed for target label (X_T_G) selection",
    )

    parser.add_argument(
        "--seed_subdataset",
        type=int,
        default=10,
        help="Random seed for sub-dataset sampling (used with --label_choice_method rd_among_possible)",
    )

    parser.add_argument(
        "--remove_ineligible_4_3_1",
        action="store_true",
        default=False,
        help="Skip scenarios not eligible for the controlled bias experiment in Sec. 4.3.1",
    )

    # --- DAB algorithm ---

    parser.add_argument(
        "--num_steps", "--it", "--iter",
        type=int,
        default=15,
        help="Number of DAB biasing iterations per scenario",
    )

    parser.add_argument(
        "--proposal_temp", "--temp", "--temperature",
        type=float,
        default=0.001,
        help="Temperature for token proposal sampling during DAB generation (lower = more peaked)",
    )

    parser.add_argument(
        "--onehot",
        action="store_true",
        default=False,
        help="Use a hard one-hot (vs. soft) token representation when computing the DLP gradient",
    )

    parser.add_argument(
        "--seed_dlp",
        type=int,
        default=40,
        help="Random seed for DLP token distribution sampling",
    )

    parser.add_argument(
        "--dlp_unmask",
        action="store_true",
        default=False,
        help="Allow DLP to propose changes at positions that were previously frozen",
    )

    parser.add_argument(
        "--k_val",
        type=int,
        default=250,
        help="Top-k vocabulary candidates per token position in DAB stage 1",
    )

    parser.add_argument(
        "--weight_val",
        type=float,
        default=1,
        help="Scalar weight on the main objective in the DAB bias update",
    )

    parser.add_argument(
        "--logit_weight_val",
        type=int,
        default=1,
        help="Scalar weight on the logit-level objective in the DAB bias update",
    )

    # --- X_o regularization (optional) ---

    parser.add_argument(
        "--use_Xo_topk_ids",
        action="store_true",
        default=False,
        help="Restrict DAB token candidates to the top-k nearest neighbors from the original X_o tokens",
    )

    parser.add_argument(
        "--use_Xo_topk_S2",
        action="store_true",
        default=False,
        help="Apply the X_o top-k token constraint in DAB stage 2",
    )

    parser.add_argument(
        "--k_val_s2",
        type=int,
        default=250,
        help="Top-k vocabulary candidates per token position in DAB stage 2 (only used if --use_Xo_topk_S2)",
    )

    parser.add_argument(
        "--add_Xo_reg_in_loss",
        action="store_true",
        default=False,
        help="Add X_o similarity regularization to the DAB loss to stay close to the original scene",
    )

    parser.add_argument(
        "--xo_loss_weight",
        type=float,
        default=5,
        help="Weight for the X_o similarity regularization term in the loss (only used if --add_Xo_reg_in_loss)",
    )

    parser.add_argument(
        "--add_Xo_reg_in_bias",
        action="store_true",
        default=False,
        help="Add X_o regularization directly to the DAB token bias update",
    )

    parser.add_argument(
        "--Xo_bias_weight_val",
        type=float,
        default=3,
        help="Scalar weight for the X_o similarity term in the DAB bias update (only used if --add_Xo_reg_in_bias)",
    )

    # --- Output / template control ---

    parser.add_argument(
        "--optim_part",
        type=str,
        default="all",
        choices=["ego", "sv", "all"],
        help="Part of the scene prompt to optimize: ego-vehicle only, surrounding vehicles only, or both",
    )

    parser.add_argument(
        "--fix_template",
        action="store_true",
        default=False,
        help="Reject candidate counterfactuals that violate the expected output template structure",
    )

    parser.add_argument(
        "--fix_all_non_learnable_indices",
        action="store_true",
        default=False,
        help="Freeze all token positions that are not explicitly marked as learnable",
    )

    parser.add_argument(
        "--save_txt",
        action="store_true",
        default=False,
        help="Also save generated sequences as plain text files alongside JSON results",
    )

    parser.add_argument(
        "--small_scorer",
        action="store_true",
        default=False,
        help="Use a smaller model for BERTScore computation (faster but less accurate)",
    )

    # --- Brute-force specifics (not released) ---

    parser.add_argument(
        "--N_sampling", "--N",
        type=int,
        default=100,
        help="Number of token candidates sampled per position in the brute-force search",
    )

    parser.add_argument(
        "--close_range",
        action="store_true",
        default=False,
        help="Restrict brute-force candidates to a semantically close token range",
    )

    parser.add_argument(
        "--single_tok",
        action="store_true",
        default=False,
        help="Test single-token substitutions only, without multi-token combinations",
    )

    parser.add_argument(
        "--max_bs",
        type=int,
        default=12,
        help="Maximum number of candidates evaluated in a single batched LLM forward pass",
    )

    parser.add_argument(
        "--bf_verbose",
        action="store_true",
        default=False,
        help="Print per-candidate scoring details during the brute-force search",
    )

    # --- Refinement mode (not released) ---

    parser.add_argument(
        "--refine_mode",
        action="store_true",
        default=False,
        help="Resume a previous run and refine already found counterfactuals",
    )

    parser.add_argument(
        "--folder_to_refine", "--path",
        type=str,
        default=None,
        help="Path to the results folder to resume refinement from (used with --refine_mode)",
    )

    parser.add_argument(
        "--eval_tok_change_mode",
        action="store_true",
        default=False,
        help="Evaluate token-change statistics only, without running the full search",
    )

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args
