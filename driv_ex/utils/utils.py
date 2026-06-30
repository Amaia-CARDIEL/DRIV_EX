# Copyright (c) 2026 Valeo. All rights reserved.
"""General experiment bookkeeping: seeding, result-folder and writer setup, naming
conventions, optimization counters, optimizer reinitialization, and TensorBoard
metric-logging helpers."""
import os
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from driv_ex import REPO_DIR
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return None

def get_emb_dim(args):
    if "Llama-3-8B" in args.HF_name or "Mistral-7B" in args.HF_name:
        emb_dim = 4096
    elif "Llama-2" in args.HF_name:
        emb_dim = 5120
    elif "Qwen" in args.HF_name:
        emb_dim = 3584
    else:
        raise Exception(f"emb dim for {args.HF_name} not yet implemented")
    return emb_dim


def define_root_folder(args, label_choice_method="impose_crash", llm_ft_ckpts=None, algo="pez"):

    core_folder = REPO_DIR
    HF_name = llm_ft_ckpts if llm_ft_ckpts is not None else args.HF_name

    dataset_name="highD"
    sub_crash_dataset = "gt_crash_data"
    root_folder = args.result_save_path if "/home" in args.result_save_path else os.path.join(core_folder, "results", "cf_algo_results", "LC_LLM", f"{algo}_results", f"{dataset_name}", f"{sub_crash_dataset}", f"{label_choice_method}", f"optim_{args.optim_part}", f"{HF_name}", args.result_save_path)

    return root_folder


def get_counters_and_booleans(args, nb_optim_toks):

    count = {}

    # nb forwards
    if args.K > 1:
        total_nb_forwards = nb_optim_toks * (args.K-1) + 1
        count["iter_up_to"] = total_nb_forwards//args.max_bs if total_nb_forwards % args.max_bs == 0 else total_nb_forwards//args.max_bs + 1
        print(f"{total_nb_forwards} forwards for a gradient computation, done in {count['iter_up_to']} iterations")
    else:
        total_nb_forwards = 1

    # Init counters & bools

    best_prob_X_T_G = -1 # 0
    best_prob_X_T_G_for_all_K_nn = -1 #0

    count["c_nb_forward"] = 0
    # count["smallest_loss"] = 999
    # count["best_text"] = ""
    # count["rank_X_T_G_for_best_prompt"] = voc_size

    if args.DEBUG_GRAD:
        count["t_forward"], count["same_grad"] = 0, 0
    if args.one_by_one in ["biggest_norm", "seq_order", "seq_inv_order"]:
        count["update_step"] = 0
    if args.one_by_one in ["re_init_all_but_one", "re_init_hard"] or args.dab_reg:
        count["need_reinit"] = False
        count["c_reinit_but_no_grad_change"], count["c_no_reinit_but_grad_change"] = 0, 0

    # reinit mode
    if args.reinit_hard_mode and args.K>1: # or (args.dab_reg and args.dab_reg_reinit_hard):
        count["reinit_hard"] = True

    else:
        count["reinit_hard"] = False
    count["reinit_hard_count"] = 0
    count["count_generate_did_not_give_eos_tok_id"]=0

    return total_nb_forwards, best_prob_X_T_G, best_prob_X_T_G_for_all_K_nn, count

def reinit_optimizer(args, optimizer, optim_embeds, keep_state = True):

    if keep_state:
        optimizer_state_dict = optimizer.state_dict()
        optimizer = torch.optim.AdamW([optim_embeds], lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        optimizer = torch.optim.AdamW([optim_embeds], lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

# functions related to tensorboard's writer

def define_writer(args, dataset_idx=None, expe_seed=None, label_choice_method="impose_crash", llm_ft_ckpts=None, algo="pez"):

    if not args.dont_record_metrics:
        # Ensure saving folders exist
        root_folder = define_root_folder(
            args, label_choice_method=label_choice_method, llm_ft_ckpts=llm_ft_ckpts, algo=algo
            )

        print("Saving results in folder:", root_folder)
        os.makedirs(os.path.join(root_folder, "best_results_json"), exist_ok=True)
        if not args.no_tensorboard_nor_deco:
            os.makedirs(os.path.join(root_folder, "tensorboards"), exist_ok=True)
            os.makedirs(os.path.join(root_folder, "decoded_text"), exist_ok=True)

        # Define log name and filename

        if args.save_filename is None:

            # specify the positions of the tokens that can be learnt
            tok_modif_str = "_update"
            consecutive_ints = sorted(args.modif_pos) == list(range(min(args.modif_pos), max(args.modif_pos)+1))
            if args.modif_pos == [0]:
                tok_modif_str += f"_all"
            elif len(args.modif_pos) == 1:
                tok_modif_str += f"_{args.modif_pos[0]}"
            elif consecutive_ints:
                tok_modif_str += f"_{min(args.modif_pos)}_to_{max(args.modif_pos)}"
            else:
                for modif_tok in args.modif_pos:
                    tok_modif_str += f"_{modif_tok}"

            # specify weight decay if different from 0 only
            wd_str = f"_WD_{args.weight_decay}" if args.weight_decay != 0 else ""

            # specify how many K nn used + with which aggregation method if use K>1 nn gradients
            if args.K == 1:
                K_grad = ""
            else:
                best_top_str = f"_top_{args.top_proba_trunc}" if args.top_proba_trunc is not None else ""
                lb_str = f"_>_{args.proba_lower_threshold}" if args.proba_lower_threshold is not None else ""
                K_grad = f"_K_{args.K}_by_{args.grad_weighting}{best_top_str}{lb_str}"

            # specify how learnable tokens are initialized
            init_str = "" if args.init == "input_toks" else f"_{args.init}_s_{args.seed}"

            # specify if use of one_by_one variant
            one_by_one_str = "_1by1_" + args.one_by_one if args.one_by_one is not None else ""

            sum_loss_str = "_sum_loss" if args.dont_normalize_loss else ""
            reinit_hard_str = "_Reinit_Hard" if args.reinit_hard_mode else ""

            # specify on which data idx we did the optim
            dataset_idx_str = args.dataset_idx if dataset_idx is None else dataset_idx
            dataset_idx_str = f"Idx_{dataset_idx_str}_"

            # specify seed if use experimental seed
            expe_seed_str = "" if expe_seed is None else f"_expe_seed_{expe_seed}"

            # print
            print("\n\nStart optim loop for dataset idx", dataset_idx_str, " " + expe_seed_str.strip('_'))

            # specify nb of run and final filename
            log_name = f"{dataset_idx_str}LR_{args.lr}{wd_str}_iters_{args.num_steps}{K_grad}{tok_modif_str}{one_by_one_str}{init_str}{sum_loss_str}{reinit_hard_str}{expe_seed_str}".strip("_")
            # if os.path.exists(os.path.join(root_folder, "tensorboards", log_name)):
            #     log_name_ori = log_name
            #     run=1
            #     while os.path.exists(os.path.join(root_folder, "tensorboards", log_name)):
            #         log_name = log_name_ori + f"_run_{run}"
            #         run +=1

            save_filename = log_name.strip("/")

        else:
            save_filename = args.save_filename

        log_dir = os.path.join(root_folder, "tensorboards", save_filename)
        if os.path.exists(log_dir):
            # raise Exception(f"{log_dir} already exists")
            print(f"{log_dir} already exists")

        # define writer
        if not args.no_tensorboard_nor_deco:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer=None
    else:
        writer, save_filename, root_folder = None, None, None
    return writer, save_filename, root_folder








def add_metric(x, y, name, writer):
    writer.add_scalar(
        name,
        y,
        x,
    )
    return None


def writing_main_metrics(writer, step, loss, Prob_X_T_G, rank_X_T_G):

    writer.add_scalar(
        f"0) Loss",
        loss, #loss.item(),
        step,
    )

    writer.add_scalar(
        f"1) P(X_T_G|...) in %",
        Prob_X_T_G,
        step,
    )

    writer.add_scalar(
        f"2) X_T_G rank",
        rank_X_T_G,
        step,
    )

    writer.add_scalar(
        f"2) X_T_G rank (log scale)",
        math.log(rank_X_T_G+1),
        step,
    )

    return None


def writing_grad_norm_metrics(writer, step, optim_embeds):

    with torch.no_grad():

        batch_mean_embed_grad = torch.mean(optim_embeds.grad.detach().clone(), dim=0, keepdim = False) # size (seq_len, emb_dim)
        embed_grad_norm =  torch.norm(batch_mean_embed_grad, dim=-1, keepdim=False) # size seq_len

        writer.add_scalar(
            f"Grad norm (mean for all tokens)",
            torch.mean(embed_grad_norm).item(),
            step,
            )

        writer.add_scalar(
            f"Grad norm (max for all tokens)",
            torch.max(embed_grad_norm).item(),
            step,
            )

        # per token
        for tok_it, tok_wise_grad_norm in enumerate(embed_grad_norm):
            writer.add_scalar(
                f"Grad norm for tok n°{tok_it}",
                tok_wise_grad_norm.item(),
                step,
                )

    return None