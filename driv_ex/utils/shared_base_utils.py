# Copyright (c) 2026 Valeo. All rights reserved.
"""Experiment setup shared across all counterfactual search algorithms: checkpoint path
resolution (long internal vs. simple released names), data-subset selection for
crash scenarios, scenario skip logic, and source/target label assignment."""
import os, json
import random
from driv_ex import REPO_DIR
from driv_ex.dataset import DATA_DIR
from driv_ex.utils.parsing_utils import parse_vehicle_data, parse_vehicle_data_to_count_bias_changes
from driv_ex.utils.LLM_utils import get_valid_vocab, transform_class_into_voc_id, get_eos_token


# Exact (HF_name, FT_type, grad_cumul, ckpt_nb) -> simple released checkpoint folder name.
# Only the checkpoints we publicly release have a simple alias; any other combination
# (different grad_cumul, different ckpt_nb, biased FT types, etc.) must use the long path.
_RELEASED_CKPTS = {
    # key: (HF_name, FT_type, grad_cumul, ckpt_nb, lr, bs)
    ("Meta-Llama-3-8B-Instruct",           "classic_FT",  8,  4400, 0.0005, 8): "Driving_LLM_Llama-3-8B-Instruct",
    ("mistralai/Mistral-7B-Instruct-v0.3", "classic_FT",  32, 1100, 0.0005, 8): "Driving_LLM_Mistral-7B-Instruct",
    ("Qwen/Qwen2.5-7B-Instruct",           "classic_FT",  8,  8000, 0.0005, 8): "Driving_LLM_Qwen2.5-7B-Instruct",
    ("Meta-Llama-3-8B-Instruct",           "X_vision_FT", 8,  750,  0.0005, 8): "Fluency_expert_Llama-3-8B-Instruct",
    ("mistralai/Mistral-7B-Instruct-v0.3", "X_vision_FT", 32, 500,  0.0005, 8): "Fluency_expert_Mistral-7B-Instruct",
    ("Qwen/Qwen2.5-7B-Instruct",           "X_vision_FT", 8,  800,  0.0005, 8): "Fluency_expert_Qwen2.5-7B-Instruct",
}


def resolve_ckpt_path(FT_type, HF_name_raw, long_subdir_name, ckpt_nb, grad_cumul, lr, bs):
    """Resolve the checkpoint path, preferring simple released names over long internal names.

    For the publicly released checkpoints (exact match on HF_name, FT_type, grad_cumul,
    ckpt_nb, lr and bs), tries the simple name first:
      REPO_DIR/LCLLM_ckpts/{FT_type}/{simple_name}   e.g. Driving_LLM_Llama-3-8B-Instruct

    Falls back to the long internal path in all other cases:
      REPO_DIR/LCLLM_ckpts/{FT_type}/{long_subdir_name}/checkpoint-{ckpt_nb}

    Args:
        HF_name_raw: raw args.HF_name value (e.g. 'mistralai/Mistral-7B-Instruct-v0.3')
        long_subdir_name: processed directory name (e.g. 'mistralai_Mistral-7B-Instruct-v0.3_quant_8_lr_0.0005_bs_8_grad_cumul_32')
        lr: FT learning rate (args.FT_learning_rate)
        bs: per-device train batch size (args.per_device_train_batch_size)
    """
    simple_name = _RELEASED_CKPTS.get((HF_name_raw, FT_type, int(grad_cumul), int(ckpt_nb), float(lr), int(bs)))
    if simple_name is not None:
        simple_path = REPO_DIR / "LCLLM_ckpts" / FT_type / simple_name
        if simple_path.exists():
            return str(simple_path)
    return str(REPO_DIR / "LCLLM_ckpts" / FT_type / long_subdir_name / f"checkpoint-{ckpt_nb}")


def retrieve_ckpt_specific_subset(args, no_config, no_config_data):

    llm_full_name, llm_ft_ckpts, data_type = define_llm_full_naming(args, no_config, no_config_data)

    crash_info = None

    if args.label_choice_method in ["impose_crash", "avoid_crash"]:

        eval_llm_core_folder = DATA_DIR / "highD_val_crash_data" / "LLM_eval_on_crash_subset"
        eval_llm_on_crash_path = os.path.join(eval_llm_core_folder, f"{llm_full_name}_{args.FT_type}_ckpt_{args.ckpt_nb}_on_{data_type}", "all_eval")
        eval_llm_on_crash_set = os.path.join(eval_llm_on_crash_path, "eval_on_809_crash_data_from_GT_data.json")
        with open(eval_llm_on_crash_set) as f:
            crash_info = json.load(f)

        if args.label_choice_method == "impose_crash":
            crash_info = [dico for dico in crash_info if dico["infered_maneuver_eval"]=="good" and dico["infered_lc_label"]!=dico["crash_lc_label"]]
        if args.label_choice_method == "avoid_crash":
            crash_info = [dico for dico in crash_info if dico["infered_maneuver_eval"]=="collision" and dico["infered_lc_label"]==dico["crash_lc_label"]]
        # print("Total nb of 'crash data' before slicing if needed:", len(crash_info))

        max_data = len(crash_info) if args.max_data is None else args.max_data
        sampled_idx = list(range(args.min_samples, min(len(crash_info), max_data)))
        crash_info = [e for it, e in enumerate(crash_info) if it in sampled_idx]
        sampled_idx = [dico["data_id"] for dico in crash_info]

    elif args.label_choice_method == "rd_among_possible":
        random.seed(args.seed_subdataset)
        sampled_idx = random.sample(list(range(4000, 4100)) + list(range(0, 100)) + list(range(2000, 2100)), args.max_data)
        sampled_idx.sort()

    return sampled_idx, crash_info, llm_ft_ckpts


def define_llm_full_naming(args, no_config, no_config_data=None):

    llm_full_name = f"{args.HF_name.replace('/', '_')}_quant_{args.eval_quantization.replace('bit', '')}_lr_{args.FT_learning_rate}_bs_{args.per_device_train_batch_size}_grad_cumul_{args.grad_cumul}"
    if no_config:
        llm_ft_ckpts = f"{llm_full_name}_{args.FT_type}_ckpt_{no_config_data['ckpt_nb']}_ckpt_x_vision_{no_config_data['ckpt_nb_x_vision']}"
    else:
        llm_ft_ckpts = f"{llm_full_name}_{args.FT_type}_ckpt_{args.ckpt_nb}_ckpt_x_vision_{args.ckpt_nb_x_vision}"

    data_type = "base_data" if args.bias_type is None else args.bias_type
    data_type = data_type.replace("bias", "reverse_bias") if args.reverse_bias else data_type

    if args.bias_type is not None:
        llm_ft_ckpts += f"_on_{data_type}"

    return llm_full_name, llm_ft_ckpts, data_type


def skip_scenario(args, root_folder, save_filename, real_data_it, X_vision, X_T_G_decoded, tokenizer, dab_algo=False):

    skip=False
    if dab_algo:
        if os.path.exists(os.path.join(root_folder, save_filename)):
            print(f"val already done on data n°{real_data_it}")
            skip=True

    else:
        if os.path.exists(os.path.join(root_folder, "best_results_json", save_filename+'.json')):
            print(f"val already done on data n°{real_data_it}")
            skip=True

    if "Mistral" in args.HF_name:
        sv_data = parse_vehicle_data(X_vision)['surrounding_vehicles']
        if len(sv_data)>6:
            print("Too many surrounding vehicles, lead to OOM => continue")
            skip=True

    if  "- No surrounding vehicles within 200m." in X_vision or "The information about its surrounding vehicles (within a range of 200 m) is listed as follows:\n [/INST]" in X_vision:
        print("skip data as no information on other vehicles around ego car")
        skip=True

    # remove ineligible data for controlled experiment in Sec 4.3.1
    if args.remove_ineligible_4_3_1:
        result_init, _ = parse_vehicle_data_to_count_bias_changes(X_vision, tokenizer) # result is a dict
        if args.FT_type == "vehicle_bias_FT":
            ego_type_init = result_init['ego_type']
            if X_T_G_decoded == 1 and ego_type_init == "truck":
                print("eligible for controlled experiment")
            elif X_T_G_decoded == 2 and ego_type_init == "car":
                print("eligible for controlled experiment")
            else:
                print("not eligible for controlled experiment => continue")
                skip=True
        else:
            raise Exception("not implem")

    return skip


def set_source_vs_target_token(args, crash_info, sampled_data_it, real_data_it, label, X_sys, X_vision, test_label_switch=True):

    assert crash_info[sampled_data_it]["data_id"] == real_data_it
    X_T_o = crash_info[sampled_data_it]["infered_lc_label"]
    if args.label_choice_method in ["avoid_crash", "rd_among_possible"]:
        X_T_G_decoded = set_xtg_given_xto(label, X_sys+X_vision, method="rd_among_possible", seed_xtg=args.seed_xtg)
    elif args.label_choice_method == "impose_crash":
        X_T_G_decoded = crash_info[sampled_data_it]["crash_lc_label"]

    if test_label_switch:
        assert X_T_G_decoded != X_T_o
        print(f"Label switch goal: {X_T_o} => {X_T_G_decoded}")

    X_T_G = transform_class_into_voc_id(
        HF_name=args.HF_name,
        label=X_T_G_decoded
        )

    return X_T_o, X_T_G, X_T_G_decoded



def set_xtg_given_xto(label, text, method, seed_xtg):
    # 0 no LC
    # 1 left LC
    # 2 right LC
    X_start = "The target vehicle is driving on a "
    X_start_id = text.find(X_start) + len(X_start)
    # print("text where to find if left or right lane to exclude target classes:\n", text[X_start_id:])
    other_classes = [0,1,2]
    other_classes.remove(label)
    if method == "random":
        random.seed(seed_xtg)
        xtg = random.choice(other_classes)
    elif method == "rd_among_possible":
        # if "highway, located at the rightmost lane." in text[X_start_id:]:
        if "highway, in the right lane." in text[X_start_id:]:
            if 2 in other_classes:
                other_classes.remove(2)
        # if "highway, located at the leftmost lane." in text[X_start_id:]:
        if "highway, in the left lane." in text[X_start_id:]:
            if 1 in other_classes:
                other_classes.remove(1)
        random.seed(seed_xtg)
        xtg = random.choice(other_classes)
    elif method == "rd_among_impossible":
        # if "highway, located at the rightmost lane." in text[X_start_id:]:
        if "highway, in the right lane." in text[X_start_id:]:
            xtg=2
        # if "highway, located at the leftmost lane." in text[X_start_id:]:
        if "highway, in the left lane." in text[X_start_id:]:
            xtg=1
        else:
            xtg=None
    else:
        raise Exception("target label selection method not implemented")
    return xtg



