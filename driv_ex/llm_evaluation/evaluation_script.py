# Copyright (c) 2026 Valeo. All rights reserved.
import math
import random
import time
import os, json, sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from driv_ex.dataset.textual_highD_dataset import textual_highD_dataset
from driv_ex.utils.LLM_utils import load_HF_model_tok, load_FT_model_via_base_model, gen_answer
from driv_ex.llm_evaluation.evaluation_config import get_args
from driv_ex.utils.driving_metrics import *
from driv_ex.utils.shared_base_utils import resolve_ckpt_path
from driv_ex import REPO_DIR
from driv_ex.dataset import DATA_DIR
from collections import Counter


def compute_and_save_metrics(y_true, y_pred, y_true_traj, result_folder, subfolder, c_no_label, all_gen_results, all_full_scenarios, failed_gen):

    # metric computation: Precision, Recall, F1 (macro, micro, detailed per-class report)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    per_class_report = classification_report(y_true, y_pred, target_names=["0: Keep lane", "1: Left LC", "2: Right LC"], labels=[0,1,2])

    # saving results to json
    result_json = {
        "macro_avg":{
            "P":precision_macro,
            "R":recall_macro,
            "F1":f1_macro
        },
        "micro_avg":{
            "P":precision_micro,
            "R":recall_micro,
            "F1":f1_micro
        },
        "per_class_report":per_class_report,
        "nb_data_out_of_class": c_no_label,
        "out_of_class_gen":failed_gen
    }

    if len(all_gen_results)>0:

        # perform crash eval
        gen_scenario_and_eval, maneuver_list=[], []
        pred_traj_lists=[]
        all_traj_extracted = True

        for scene_str, GT_list in zip(all_full_scenarios, y_true_traj):

            try:
                pred_traj = extract_trajectory(scene_str)
            except:
                pred_traj = None
                print(f"failure to extract_trajectory from \n{scene_str}")
                all_traj_extracted = False
            pred_traj_lists.append(pred_traj)

            if pred_traj is not None:
                try:
                    rmse_lon, rmse_lat = compute_trajectory_rmse(GT_list, pred_traj)
                except:
                    print(f"failure to compute_trajectory_rmse from \nGT_list={GT_list}\npred_traj={pred_traj}")
                    rmse_lon, rmse_lat = None, None
            else:
                rmse_lon, rmse_lat = None, None

            try:
                man_eval = evaluate_maneuver(scene_str, simulation_duration=4.0)
            except:
                print(f"failure to evaluate_maneuver (4 secs) from \n{scene_str}")
                man_eval=None

            try:
                man_eval_multi_sec = evaluate_maneuver_multi_sec(scene_str)
            except:
                print(f"failure to evaluate_maneuver_multi_sec from \n{scene_str}")
                man_eval_multi_sec=None

            try:
                lc_traj_coherence = is_lc_label_and_traj_coherent(scene_str)
            except:
                lc_traj_coherence = None

            gen_scenario_and_eval.append(
                {
                    "full_scenario": scene_str,
                    "lc_traj_coherence": lc_traj_coherence,
                    "maneuver_eval": man_eval,
                    "man_eval_multi_sec": man_eval_multi_sec,
                    "rmse_lon":rmse_lon,
                    "rmse_lat":rmse_lat
                    }
                )
            maneuver_list.append(man_eval)
        result_json[f"maneuver_stats"] = dict(Counter(maneuver_list))

        if all_traj_extracted:
            try:
                dataset_rmse_lon, dataset_rmse_lat = compute_dataset_trajectory_rmse(y_true_traj, pred_traj_lists)
            except:
                dataset_rmse_lon, dataset_rmse_lat = None, None
                print(f"failure to compute_dataset_trajectory_rmse from \ny_true_traj={y_true_traj}\npred_traj_lists={pred_traj_lists}")

            result_json[f"dataset_rmse_lon"] = dataset_rmse_lon
            result_json[f"dataset_rmse_lat"] = dataset_rmse_lat

        result_json[f"gen_scenario_and_eval"] = gen_scenario_and_eval

    os.makedirs(result_folder, exist_ok = True)
    with open(os.path.join(result_folder, f"{subfolder}.json"), "w") as outfile:
        json.dump(result_json, outfile, indent=4)

    return None


def get_checkpoint_list_and_paths_for_eval(path_to_ckpt_folder, result_folder, ckpt_multiple=None, exact_checkpt_list=None, last_checkpt=False, min_checkpt=0, max_checkpt=10000):

    # get the list of all checkpoint subfolders to evaluate

    # list subfolders in checkpoint folder
    path_to_ckpt_folder = str(path_to_ckpt_folder)
    if "Llama-2" in path_to_ckpt_folder:
        folder_with_checkpoints = path_to_ckpt_folder.split("/")[-1]
    else:
        folder_with_checkpoints = path_to_ckpt_folder.replace("/", "_")

    checkpoint_subfolders = [f for f in os.listdir(path_to_ckpt_folder) if "checkpoint" in f]
    print(
        f"\n{len(checkpoint_subfolders)} checkpoints in total in folder",
        folder_with_checkpoints,
    )

    # list those for which we have already done an evaluation
    if os.path.exists(result_folder):
        print(f"result folder '{result_folder}' already exists")
        checkpoint_subfolders_already_eval = [f.replace(".json", "") for f in os.listdir(result_folder) if "checkpoint" in f]
        checkpoint_subfolders = [elem for elem in checkpoint_subfolders if elem not in checkpoint_subfolders_already_eval]
    else:
        print(f"result folder '{result_folder}' does not exist => mkdir it")
        os.makedirs(result_folder, exist_ok = True)

    # reorder checkpoints
    checkpoint_subfolders_int = [int(e.replace("checkpoint-", "")) for e in checkpoint_subfolders]
    if exact_checkpt_list is not None:
        checkpoint_subfolders_int = [e for e in exact_checkpt_list if e in checkpoint_subfolders_int]
    elif last_checkpt:
        checkpoint_subfolders_int = [max([e for e in checkpoint_subfolders_int])]
    else:
        checkpoint_subfolders_int = [
            e for e in checkpoint_subfolders_int if e >= min_checkpt and e <= max_checkpt
        ]  # cut if too many checkpoints
        checkpoint_subfolders_int.sort()

    if ckpt_multiple is not None:
        checkpoint_subfolders_int = [e for e in checkpoint_subfolders_int if e % ckpt_multiple == 0]

    checkpoint_subfolders = ["checkpoint-" + str(e) for e in checkpoint_subfolders_int]
    print(
        f"{len(checkpoint_subfolders)} checkpoints not yet evaluated:\n",
        checkpoint_subfolders,
    )

    return checkpoint_subfolders


def from_gen_to_pred_class(result):

    # result is a list of strings

    c_no_label_batch = 0
    y_pred_batch=[]
    failed_gen = []
    correct_gen=[]

    for string in result:

        if '0: Keep lane' in string:
            y_pred_batch.append(0)
            correct_gen.append(True)
        elif '1: Left lane change' in string:
            y_pred_batch.append(1)
            correct_gen.append(True)
        elif '2: Right lane change' in string:
            y_pred_batch.append(2)
            correct_gen.append(True)
        else:
            y_pred_batch.append(4)
            c_no_label_batch +=1
            failed_gen.append(f"'{string}'")
            correct_gen.append(False)

    return y_pred_batch, c_no_label_batch, failed_gen, correct_gen


def full_gen_and_eval(HF_name, model, tokenizer, val_dataloader, y_true, y_true_traj, result_folder, subfolder, FT_type, gen_until_end, DEBUG=False):

    y_pred = []
    c_no_label = 0
    nb_batch=0
    c_fail=0
    all_gen_results = []
    all_full_scenarios = []
    failed_gen=[]
    for batch_input, batch_label in tqdm(val_dataloader):

        s=time.time()
        torch.manual_seed(42)
        result, full_scenario = gen_answer(
            HF_name=HF_name,
            model=model,
            tokenizer=tokenizer,
            prompt=batch_input,
            FT_type=FT_type,
            gen_until_end=gen_until_end
            )

        if gen_until_end:
            all_gen_results.extend(result)
            all_full_scenarios.extend(full_scenario)

        y_pred_batch, c_no_label_batch, batch_failed_gen, batch_correct_gen_bools = from_gen_to_pred_class(result)
        c_no_label+=c_no_label_batch
        y_pred.extend(y_pred_batch)
        failed_gen.extend(batch_failed_gen)

        if nb_batch == 0:
            print("\nFirst 5 gen:")
            for it, res_sent in enumerate(result[:5]):
                print(f"\nGen n°{it+1}\n", res_sent)

        if c_no_label_batch>0:
            for res_sent, correct_gen_bool in zip(result, batch_correct_gen_bools):
                if not correct_gen_bool:
                    c_fail+=1
                    print(f"\nFailed gen (no lc label extracted) n°{c_fail}\n{res_sent}")

        nb_batch+=1

        if DEBUG:
            y_true = np.array([0,1,1,2])
            y_pred = [0,1,2,2]
            print("y_true:", y_true)
            print("y_pred:", np.array(y_pred))
            y_true_traj = y_true_traj[:len(y_pred)]
            break

    # compute and save metrics
    compute_and_save_metrics(y_true, np.array(y_pred), y_true_traj, result_folder, subfolder, c_no_label, all_gen_results, all_full_scenarios, failed_gen)

    return None


def get_batch_size(args, FT_type, quantization):

    HF_name = args.HF_name

    if quantization == "8bit":
        if args.DEBUG:
            batch_size = 4
        else:
            if HF_name == "Meta-Llama-3-8B-Instruct":
                batch_size = 30
            elif "Mistral-7B-Instruct-v0.3" in HF_name:
                batch_size = 30
            elif "Qwen2.5" in HF_name:
                batch_size = 30
            elif HF_name == "Llama-2-13b-chat-hf":
                batch_size = 20
            else:
                raise Exception(f"best batch size not pre-computed for this quantization {quantization} + {HF_name}")
    else:
        raise Exception(f"best batch size not pre-computed for this quantization {quantization} + {HF_name}")

    return batch_size


def get_ground_truth_data():
    # extract y_true from dataset
    p = DATA_DIR / "highD_by_lcllm" / "llama_val_surrounding_thought_with_labels.json"
    with open(p, 'r') as f:
        y = json.load(f)
    y_true = np.array([e['label'] for e in y]) # LC labels
    y_true_traj = [extract_trajectory(e['text']) for e in y]
    return y_true, y_true_traj


def get_ckpt_and_result_filepath(args, HF_name, quant_str):
    """Returns (path_to_ckpt_folder, result_folder, is_flat_ckpt).

    is_flat_ckpt=True means path_to_ckpt_folder IS the checkpoint directly (simple released
    naming, no checkpoint-XXXX subfolders). is_flat_ckpt=False means path_to_ckpt_folder
    contains checkpoint-XXXX subfolders (long internal naming).
    """
    ckpt_filename = f"{HF_name}_{quant_str}_lr_{args.FT_learning_rate}_bs_{args.per_device_train_batch_size}_grad_cumul_{args.grad_cumul}"
    if args.lora_r != 64:
        ckpt_filename += f"_r_{args.lora_r}"
    if args.lora_alpha != 16:
        ckpt_filename += f"_a_{args.lora_alpha}"

    if args.bias_type is not None:
        bias_str = args.bias_type
        if args.reverse_bias:
            bias_str = bias_str.replace("_bias", "_reverse_bias")
    else:
        bias_str = "base_data"

    FT_type_var = f"{args.FT_type}_on_{bias_str}"

    # Try simple released checkpoint path first (exact match required)
    resolved = resolve_ckpt_path(args.FT_type, args.HF_name, ckpt_filename, args.ckpt_nb, args.grad_cumul, args.FT_learning_rate, args.per_device_train_batch_size)
    simple_name = os.path.basename(resolved)
    is_flat_ckpt = not simple_name.startswith("checkpoint-")

    if is_flat_ckpt:
        # Released checkpoint: results sit directly under FT_type_var/, named after the simple alias
        result_folder = REPO_DIR / "results" / "llm_evaluation_results" / FT_type_var
        path_to_ckpt_folder = resolved
    else:
        result_folder = REPO_DIR / "results" / "llm_evaluation_results" / FT_type_var / ckpt_filename
        path_to_ckpt_folder = REPO_DIR / "LCLLM_ckpts" / args.FT_type / ckpt_filename

    return path_to_ckpt_folder, result_folder, is_flat_ckpt


def main(args):

    # whether to gen full answer to do crash eval on results
    do_crash_eval=True

    # extract ground truth
    y_true, y_true_traj = get_ground_truth_data()

    # load base model
    if "Llama-2" in args.HF_name: HF_name = args.HF_name if "/" not in args.HF_name else args.HF_name.split("/")[-1]
    else: HF_name = args.HF_name.replace("/", "_")

    quantization=args.eval_quantization # default is "8bit", else "4bit", None
    if quantization == "8bit": quant_str = "quant_8"
    elif quantization == "4bit": quant_str = "quant_4"
    else: quant_str = ""

    base_model, tokenizer, _, _ = load_HF_model_tok(
        HF_name=args.HF_name,
        eval_mode=True,
        freeze_params=True,
        FT_mode=False,
        timing=True,
        quantization=quantization,
        return_text_embedding=False,
        FT_LoRA_folder=None
        )

    if "Mistral-7B" in HF_name:
        tokenizer.padding_side = 'left'

    # get dataloader with proper batch size
    val_dataset = textual_highD_dataset(
        HF_name=args.HF_name,
        bias_type=args.bias_type,
        input_prompt_only=True,
        with_reverse_bias=True if args.reverse_bias else False,
        with_labels=True,
        split_X_and_Y=False,
        )

    # # just to visualize one data of each class
    # for i in [1, 2001, 4001]:
    #     print("\nval data n°", i)
    #     print(val_dataset[i][0])

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=get_batch_size(args, args.FT_type, quantization),
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )

    # eval 0 shot LLMs
    if args.FT_type == "classic_FT" and args.dont_skip_0_shot_eval:

        zero_shot_result_folder = REPO_DIR / "results" / "llm_evaluation_results" / "0_shot_LLMs"
        zero_shot_filename = f"0_shot_{HF_name}_{quant_str}_results"
        if not os.path.exists(os.path.join(zero_shot_result_folder, f"{zero_shot_filename}.json")):

            print(f"eval of {HF_name} 0 shot")
            full_gen_and_eval(
                HF_name=HF_name,
                model=base_model,
                tokenizer=tokenizer,
                val_dataloader=val_dataloader,
                y_true=y_true,
                y_true_traj=y_true_traj,
                result_folder=zero_shot_result_folder,
                subfolder=zero_shot_filename,
                FT_type="0_shot",
                gen_until_end=True,
                DEBUG=args.DEBUG
                )


    # eval fine-tuned checkpoints
    if not args.only_0_shot:

        path_to_ckpt_folder, result_folder, is_flat_ckpt = get_ckpt_and_result_filepath(args, HF_name, quant_str)

        if is_flat_ckpt:
            # Simple released checkpoint: the folder itself is the checkpoint (no checkpoint-XXXX subdir)
            subfolder = os.path.basename(str(path_to_ckpt_folder))
            print(f"\nLoading simple released checkpoint: {path_to_ckpt_folder}")
            model = load_FT_model_via_base_model(
                base_model=base_model,
                complete_FT_path=str(path_to_ckpt_folder),
                timing=False
            )
            full_gen_and_eval(
                HF_name=HF_name,
                model=model,
                tokenizer=tokenizer,
                val_dataloader=val_dataloader,
                y_true=y_true,
                y_true_traj=y_true_traj if do_crash_eval else None,
                result_folder=result_folder,
                subfolder=subfolder,
                FT_type=args.FT_type,
                gen_until_end=True if do_crash_eval else False,
                DEBUG=args.DEBUG
                )
        else:
            checkpoint_subfolders = get_checkpoint_list_and_paths_for_eval(
                path_to_ckpt_folder=path_to_ckpt_folder,
                result_folder=result_folder,
                ckpt_multiple=args.ckpt_multiple,
                exact_checkpt_list=args.exact_checkpt_list,
                last_checkpt=args.last_checkpt,
                min_checkpt=args.min_checkpt,
                max_checkpt=args.max_checkpt
                )

            for subfolder in checkpoint_subfolders:

                # loading lora adapters on top of base model
                print(f"\nAdd LoRA's {subfolder} to {HF_name} with {args.FT_type}")
                model = load_FT_model_via_base_model(
                    base_model=base_model,
                    complete_FT_path=os.path.join(path_to_ckpt_folder, subfolder),
                    timing=False
                )

                # getting pred from LLMs + compute / save metrics
                full_gen_and_eval(
                    HF_name=HF_name,
                    model=model,
                    tokenizer=tokenizer,
                    val_dataloader=val_dataloader,
                    y_true=y_true,
                    y_true_traj=y_true_traj if do_crash_eval else None,
                    result_folder=result_folder,
                    subfolder=subfolder,
                    FT_type=args.FT_type,
                    gen_until_end=True if do_crash_eval else False,
                    DEBUG=args.DEBUG
                    )


    return None


if __name__ == "__main__":
    args = get_args()
    main(args)