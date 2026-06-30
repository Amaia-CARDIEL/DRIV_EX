# Copyright (c) 2026 Valeo. All rights reserved.
import os
import json
import argparse
from tqdm import tqdm

from driv_ex import REPO_DIR
from driv_ex.utils.driving_metrics import get_label_from_full_scenario, is_lc_label_and_traj_coherent
from driv_ex.utils.shared_base_utils import resolve_ckpt_path


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract LLM evaluation on the safety-critical crash subset from full val-set eval results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--HF_name", "--LLM", type=str, default="Meta-Llama-3-8B-Instruct",
                        help="LLM name (HuggingFace ID or short name)")
    parser.add_argument("--FT_type", type=str, default="classic_FT",
                        help="Fine-tuning type used when running evaluation")
    parser.add_argument("--ckpt_nb", type=int, default=4400,
                        help="Checkpoint number to extract results for")
    parser.add_argument("--eval_quantization", type=str, default="8bit",
                        help="Quantization used during evaluation ('8bit', '4bit', or None)")
    parser.add_argument("--FT_learning_rate", type=float, default=0.0005,
                        help="Fine-tuning learning rate used when running evaluation")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Per-device training batch size used when running evaluation")
    parser.add_argument("--grad_cumul", "--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps used when running evaluation")
    parser.add_argument("--bias_type", type=str, default=None,
                        help="Bias type (set to None for unbiased / base model evaluation)")
    parser.add_argument("--reverse_bias", action="store_true", default=False,
                        help="Whether the bias was reversed")
    return parser.parse_args()


def main(args):

    print("Recup crash eval of", args.HF_name, args.FT_type, "ckpt", args.ckpt_nb)

    # load crash data (GT crash subset: 809 safety-critical scenarios)
    crash_data_load_path = (
        REPO_DIR / "textual_driving_data" / "highD_val_crash_data"
        / "full_crash_subset" / "crash_data_from_GT_data.json"
    )
    save_filename = "eval_on_809_crash_data_from_GT_data.json"

    with open(crash_data_load_path) as f:
        crash_data = json.load(f)["val"]

    # build llm_full_name and eval path — must match the naming used by evaluation_script.py
    HF_name = args.HF_name.replace("/", "_")
    if args.eval_quantization == "8bit":
        quant_str = "quant_8"
    elif args.eval_quantization == "4bit":
        quant_str = "quant_4"
    else:
        quant_str = ""
    llm_full_name = f"{HF_name}_{quant_str}_lr_{args.FT_learning_rate}_bs_{args.per_device_train_batch_size}_grad_cumul_{args.grad_cumul}"

    if args.FT_type == "just_vy_FT":
        assert args.bias_type == "just_vy_in_out"
    elif args.FT_type in ("vy_in_out_free_front_FT", "vy_in_out_FT"):
        assert args.bias_type == "vy_in_out"

    data_type = "base_data" if args.bias_type is None else args.bias_type
    if args.reverse_bias:
        data_type = data_type.replace("bias", "reverse_bias")

    FT_type_var = f"{args.FT_type}_on_{data_type}"
    resolved = resolve_ckpt_path(args.FT_type, args.HF_name, llm_full_name, args.ckpt_nb, args.grad_cumul, args.FT_learning_rate, args.per_device_train_batch_size)
    simple_name = os.path.basename(resolved)
    if not simple_name.startswith("checkpoint-"):
        eval_path = REPO_DIR / "results" / "llm_evaluation_results" / FT_type_var / f"{simple_name}.json"
    else:
        eval_path = REPO_DIR / "results" / "llm_evaluation_results" / FT_type_var / llm_full_name / f"checkpoint-{args.ckpt_nb}.json"

    with open(eval_path) as f:
        llm_eval_data = json.load(f)["gen_scenario_and_eval"]
    assert len(llm_eval_data) == 24000

    # define save path
    save_path = (
        REPO_DIR / "textual_driving_data" / "highD_val_crash_data" / "LLM_eval_on_crash_subset"
        / f"{llm_full_name}_{args.FT_type}_ckpt_{args.ckpt_nb}_on_{data_type}" / "all_eval"
    )
    os.makedirs(save_path, exist_ok=True)

    llm_on_crash_data = []
    for crash_id, crash_dico in enumerate(tqdm(crash_data)):
        data_id = int(crash_dico["data_id"])
        eval_dico = llm_eval_data[data_id]

        if "maneuver_eval" in eval_dico:
            infered_maneuver = eval_dico["maneuver_eval"]
        else:
            infered_maneuver = eval_dico["man_eval_multi_sec"]["4_sec"]

        infered_scenario = eval_dico["full_scenario"]

        if "lc_traj_coherence" in eval_dico:
            lc_traj_coherence = eval_dico["lc_traj_coherence"]
            infered_lc_label = get_label_from_full_scenario(infered_scenario)
        else:
            lc_traj_coherence, infered_lc_label = is_lc_label_and_traj_coherent(
                text=infered_scenario, return_lc_label=True)

        llm_on_crash_data.append({
            "data_id": data_id,
            "crash_id": crash_id,
            "crash_lc_label": crash_dico["goal_label"],
            "crash_scenario": crash_dico["scenario"],
            "infered_coherence": lc_traj_coherence,
            "infered_maneuver_eval": infered_maneuver,
            "infered_lc_label": infered_lc_label,
            "infered_scenario": infered_scenario,
        })

    assert len(llm_on_crash_data) == 809 and len(llm_on_crash_data) == len(crash_data)

    with open(os.path.join(save_path, save_filename), "w") as outfile:
        json.dump(llm_on_crash_data, outfile, indent=4)

    print(f"Saved {len(llm_on_crash_data)} entries to {save_path / save_filename}")


if __name__ == "__main__":
    args = get_args()
    main(args)
