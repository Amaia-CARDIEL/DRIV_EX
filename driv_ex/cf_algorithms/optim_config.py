# Copyright (c) 2026 Valeo. All rights reserved.
import argparse


def boolean_flag(arg) -> bool:
    """Add a boolean flag to argparse parser."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("true", "1", "yes", "y"):
        return True
    elif arg.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Expected 'true'/'false' or '1'/'0', but got '{arg}'")


def None_or_int(arg):

    if arg is None:
        return None
    elif arg.lower() in ["none"]:
        return None
    else:
        return int(arg)



def get_args(notebook=False):

    parser = argparse.ArgumentParser(
        description="Args for the DRIV-EX gradient-based counterfactual search (main algorithm) and the PEZ / GCG baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- GCG-specific ---

    parser.add_argument(
        "--gcg_batch_size", "--gcg_B",
        type=int,
        default=512,
        help="Number of candidate token substitutions to evaluate per GCG step",
    )

    parser.add_argument(
        "--gcg_topk",
        type=int,
        default=128,
        help="Top-k vocabulary candidates considered per token position in GCG",
    )

    parser.add_argument(
        "--gcg_seed",
        type=int,
        default=42,
        help="Random seed for GCG candidate selection",
    )

    parser.add_argument(
        "--bf_verbose",
        action="store_true",
        default=False,
        help="Print per-candidate scoring details during brute-force candidate evaluation",
    )

    parser.add_argument(
        "--single_tok",
        action="store_true",
        default=False,
        help="Evaluate single-token substitutions only, without multi-token combinations",
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
        choices=["classic_FT", "digit_bias_FT", "vehicle_bias_FT", "just_vy_FT", "vy_in_out_FT"],
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

    # --- Optimization loop ---

    parser.add_argument(
        "--num_steps", "--it", "--iter",
        type=int,
        default=15,
        help="Number of optimization steps per scenario",
    )

    parser.add_argument(
        "--lr", "--adam_learning_rate",
        type=float,
        default=0.01,
        help="Adam learning rate for the embedding optimization",
    )

    parser.add_argument(
        "--seed_pez", "--s",
        type=int,
        default=42,
        help="Random seed for DRIV-EX/PEZ algorithms initialization",
    )

    parser.add_argument(
        "--weight_decay", "--wd",
        type=float,
        default=0.0,
        help="L2 weight decay for the embedding optimizer",
    )

    parser.add_argument(
        "--dont_normalize_loss",
        action="store_true",
        default=False,
        help="Use sum (unnormalized) loss instead of per-token normalized loss",
    )

    parser.add_argument(
        "--init",
        type=str,
        default="input_toks",
        choices=["input_toks", "rand_voc", "semi_rand_voc", "rand_full"],
        help="How to initialize the learnable token embeddings",
    )

    # --- Token-related control ---

    parser.add_argument(
        '--modif_pos', '--m',
        type=int,
        nargs='+',
        default=[0],
        help="Token indices that are allowed to change. [0] means all positions except BOS.",
    )

    parser.add_argument(
        "--optim_part",
        type=str,
        default="all",
        choices=["ego", "sv", "all"],
        help="Part of the scene prompt to optimize: ego-vehicle only, surrounding vehicles only, or both",
    )

    parser.add_argument(
        "--one_by_one", "--obo", "--1by1",
        type=str,
        default=None,
        choices=["biggest_norm", "seq_order", "seq_inv_order", "re_init_all_but_one", "re_init_hard"],
        help="Update tokens one at a time using the specified selection strategy (None = update all simultaneously)",
    )

    parser.add_argument(
        "--reinit_hard_mode", "--reinit_hard", "--hard",
        action="store_true",
        default=False,
        help="Reinitialize embeddings to hard token values when a better combination is found (only if K>1)",
    )

    parser.add_argument(
        "--reinit_mult",
        type=int,
        default=1,
        help="Multiplier on the number of token reinitialization candidates to consider",
    )

    parser.add_argument(
        "--onehot",
        action="store_true",
        default=False,
        help="Use a hard one-hot (vs. soft) token representation when computing the gradient",
    )

    # --- X_o regularization ---

    parser.add_argument(
        "--proj_subvoc",
        action="store_true",
        default=False,
        help="Project gradient update steps onto a restricted sub-vocabulary",
    )

    parser.add_argument(
        "--subvoc_size",
        type=int,
        default=250,
        help="Size of the restricted sub-vocabulary used for gradient projection (only used if --proj_subvoc)",
    )

    parser.add_argument(
        "--sub_verbose",
        action="store_true",
        default=False,
        help="Print verbose details during sub-vocabulary projection",
    )

    parser.add_argument(
        "--add_Xo_reg_in_loss",
        action="store_true",
        default=False,
        help="Add X_o similarity regularization to the optimization loss to stay close to the original scene",
    )

    parser.add_argument(
        "--xo_loss_weight",
        type=float,
        default=7,
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

    parser.add_argument(
        "--weight_val",
        type=float,
        default=8,
        help="Scalar weight on the main objective in the DAB bias update",
    )

    parser.add_argument(
        "--logit_weight_val",
        type=int,
        default=1,
        help="Scalar weight on the logit-level objective in the DAB bias update",
    )

    parser.add_argument(
        "--use_Xo_topk_S2",
        action="store_true",
        default=False,
        help="Apply the X_o top-k token constraint in DAB regularization stage 2",
    )

    parser.add_argument(
        "--k_val",
        type=int,
        default=250,
        help="Top-k vocabulary candidates per token position for the DAB regularization term",
    )

    # --- DAB regularization on top of gradient search ---

    parser.add_argument(
        "--dab_reg",
        action="store_true",
        default=False,
        help="Enable DAB-style regularization on top of the gradient-based optim => go from PEZ to DRIV-EX",
    )

    parser.add_argument(
        "--dab_reg_reinit_hard",
        action="store_true",
        default=False,
        help="Use hard token reinitialization when the DAB regularization finds a better candidate",
    )

    # --- Output / recording ---

    parser.add_argument(
        "--result_save_path", "--path",
        type=str,
        default="./results/",
        help="Root directory where results are saved",
    )

    parser.add_argument(
        "--save_filename",
        type=str,
        default=None,
        help="Custom name for the result/TensorBoard file (auto-generated if None)",
    )

    parser.add_argument(
        "--dont_record_metrics", "--dont_record", "--dont_save",
        action="store_true",
        default=False,
        help="Disable all metric recording (TensorBoard, decoded text, JSON)",
    )

    parser.add_argument(
        "--no_tensorboard_nor_deco", "--no_t_no_deco",
        action="store_true",
        default=False,
        help="Disable TensorBoard logging and decoded-text saving (JSON results still written)",
    )

    parser.add_argument(
        "--dont_record_ppl_loss",
        action="store_true",
        default=False,
        help="Skip perplexity loss logging during evaluation steps (saves time)",
    )

    parser.add_argument(
        "--small_scorer",
        action="store_true",
        default=False,
        help="Use a smaller model for BERTScore computation (faster but less accurate)",
    )

    parser.add_argument(
        "--DEBUG_GRAD",
        action="store_true",
        default=False,
        help="Print gradient evolution and projected nearest-neighbour tokens at each step",
    )

    # --- Experiment seed / multi-run ---

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for token re-initialization and random vocabulary sampling (used when --init rand_voc or --one_by_one re_init_all_but_one)",
    )

    parser.add_argument(
        "--expe_seed",
        type=int,
        default=None,
        help="Experiment seed controlling random initialization and sampling across runs",
    )

    # ------------------------
    # ---- Legacy config ----
    # ------------------------

    # --- Legacy (using K>1 as the number Nearest-neighbour projection) ---

    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="Number of top K nearest neighbor to use during optim projection",
    )

    parser.add_argument(
        "--max_bs",
        type=int,
        default=200,
        help="Maximum number of candidates evaluated in a single batched LLM forward pass",
    )

    parser.add_argument(
        "--grad_weighting",
        type=str,
        default="P_XTG",
        choices=["P_XTG", "cos_sim"],
        help="How to weight gradients when aggregating K>1 nearest-neighbour candidates",
    )

    parser.add_argument(
        "--top_proba_trunc",
        type=int,
        default=None,
        help="When using P_XTG gradient weighting, keep only this many top-probability candidates (None = keep all)",
    )

    parser.add_argument(
        "--proba_lower_threshold",
        type=float,
        default=None,
        help="When using P_XTG gradient weighting, discard candidates below this probability threshold (None = keep all)",
    )

    # --- Legacy (task) ---

    parser.add_argument(
        "--dataset_idx", "--idx",
        type=int,
        default=0,
        help="Dataset index to process in single-sample evaluation mode",
    )

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args
