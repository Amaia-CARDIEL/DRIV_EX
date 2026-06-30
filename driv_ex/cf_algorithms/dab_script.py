# Copyright (c) 2026 Valeo. All rights reserved.
import os
import re
import time
import json
import random
from tqdm import tqdm
import torch
import numpy as np
from driv_ex.dataset.textual_highD_dataset import textual_highD_dataset
from driv_ex.cf_algorithms.dab_config import get_args
from driv_ex import REPO_DIR
from driv_ex.dataset import DATA_DIR
import torch.nn.functional as F
from driv_ex.utils.dab_utils import *
from driv_ex.utils.shared_base_utils import *
from driv_ex.utils.shared_optim_utils import get_fix_token_indices, nn_project_fct
from driv_ex.utils.LLM_utils import semantic_similarity_bertscore
from driv_ex.utils.driving_metrics import min_conditional_tok_prob


def define_filename_and_folder(args, max_data, num_steps, label_choice_method, total_config, llm_ft_ckpts):

    if total_config["use_Xo_topk_ids"] and total_config["use_Xo_topk_S2"]:
       raise Exception("cannot have both use_Xo_topk_ids and use_Xo_topk_S2")

    ## Define save_dir
    data_type = args.bias_type if args.bias_type is not None else "classic_data"
    data_type = data_type.replace("bias", "reverse_bias") if args.reverse_bias else data_type
    dataset="highD"
    sub_crash_dataset = "gt_crash_data"
    save_dir = os.path.join(REPO_DIR, "results", "cf_algo_results", "LC_LLM", "dab_results", f"{dataset}", f"{sub_crash_dataset}", f"{label_choice_method}", f"optim_{args.optim_part}", f"{llm_ft_ckpts}", f"eval_on_full_{data_type}_nb_steps_{num_steps}_temp_{args.proposal_temp}_dlp_seed_{args.seed_dlp}")

    if args.k_val != 250:
        save_dir+=f"_k_{args.k_val}"
    if args.logit_weight_val !=1:
        save_dir+=f"_logit_{args.logit_weight_val}"
    save_dir+=f"_w_{total_config['weight_val']}"

    # Define filename and update folder if needed
    Xo_string=""
    if total_config["add_Xo_reg_in_bias"]:
        Xo_string += f"_xo_bias_{args.Xo_bias_weight_val}"
    if total_config["add_Xo_reg_in_loss"]:
        Xo_string += f"_xo_loss_{args.xo_loss_weight}"

    filename=f"target_loss{Xo_string}_temp_{args.proposal_temp}_disc_w_{total_config['weight_val']}_k_{args.k_val}_dlp_seed_{args.seed_dlp}"
    save_dir+=Xo_string

    if total_config["Xo_only_in_bias"]:
        filename=f"Xo_only_in_bias_disc_w_{total_config['weight_val']}_k_{args.k_val}"
    if total_config["use_Xo_topk_ids"]:
        filename += "_Xo_topk"
        save_dir+="_Xo_topk"
    elif total_config["use_Xo_topk_S2"]:
        filename += "_Xo_topk_S2"
        save_dir+=f"_Xo_topk_S2_{args.k_val_s2}"
    if total_config["onehot"]:
        filename += "_onehot"
        save_dir+="_onehot"
    if total_config["fix_template"]:
        filename += "_fix_temp"
    if total_config["use_fadding_weights"]:
        filename += "_with_fadding_weights"
    if total_config["use_scale_weights"]:
        filename += "_with_scale_weights"
    if total_config["fix_all_non_learnable_indices"]:
        filename += "_fix_all_NL_idx"
    if args.dlp_unmask:
        save_dir+= "_unmask"
    filename += f"_ar_gen_{total_config['ar_gen_equation']}_run_1.json"
    print("filename:", filename)

    main_save_dir = os.path.join(save_dir, "best_results_json")
    os.makedirs(main_save_dir, exist_ok = True)

    deco_text_dir = os.path.join(save_dir, "decoded_text")
    if args.save_txt:
        os.makedirs(deco_text_dir, exist_ok = True)

    return main_save_dir, deco_text_dir, filename



def main(args):

    ## Args
    batch_size = 1
    num_steps= args.num_steps
    save = True
    add_ppl_loss=False
    label_choice_method = args.label_choice_method

    total_config={
        # args from dab/configs/defaults/dlp.yaml in dab repository
        "logit_weight_val": args.logit_weight_val,
        "weight_val": args.weight_val, # ori code's config 1.0, # important
        "k_val":args.k_val, #250,
        "k_val_s2": args.k_val_s2,
        "proposal_temp":args.proposal_temp, #0.1,
        "onehot":args.onehot,
        "initialization":"from_ori_X", # in ori code "zero", # important
        "min_weight":1.0,
        "max_weight":1.0,
        "use_fadding_weights":False,
        "use_bolt_weights":False,
        "use_scale_weights":True,
        "ar_gen_equation": "ours", #"dlp", "dlp_wo_scale"
        "fix_template":True, #args.fix_template,
        "fix_all_non_learnable_indices":True, # args.fix_all_non_learnable_indices,
        "add_Xo_reg_in_loss":args.add_Xo_reg_in_loss,
        "xo_loss_weight":args.xo_loss_weight,
        "add_Xo_reg_in_bias":args.add_Xo_reg_in_bias,
        "Xo_bias_weight_val": args.Xo_bias_weight_val,
        "use_Xo_topk_ids": args.use_Xo_topk_ids,
        "use_Xo_topk_S2": args.use_Xo_topk_S2,
        "Xo_only_in_bias": False,
        "initialization_noise_rate":0.5,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "seed_dlp": args.seed_dlp,
        "save_txt": args.save_txt,
        "dlp_unmask": args.dlp_unmask
    }


    ## Data

    val_dataset = textual_highD_dataset(
        HF_name=args.HF_name,
        bias_type=args.bias_type,
        input_prompt_only=False,
        with_reverse_bias=True if args.reverse_bias else False,
        with_labels=True,
        split_X_and_Y=True,
        fix_ego_car= True if args.optim_part == "sv" else False,
        TEST_ONE_DATA=True if args.max_data in [0, 1] else False # for debug
        )

    # Retrieve llm ckpt name + safety-critical subset
    sampled_idx, crash_info, llm_ft_ckpts = retrieve_ckpt_specific_subset(args=args, no_config=False, no_config_data=None)
    val_dataset = torch.utils.data.Subset(val_dataset, sampled_idx)
    print(f"Launch optim on {len(val_dataset)} data")

    save_dir, deco_text_dir, filename = define_filename_and_folder(args, args.max_data, num_steps, label_choice_method, total_config, llm_ft_ckpts)

    ## Load models
    model, tokenizer, scorer, _ = load_base_and_adaptors(
        args=args, FT_type=args.FT_type, no_config=False
    )
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None

    ## Loss
    loss_fct = torch.nn.CrossEntropyLoss() # mean reduction by default, otherwise can use arg reduction='none'


    #--- Loop over dataset

    for sampled_data_it, (X_sys, X_vision, Y, label) in enumerate(val_dataset):

        start = time.time()
        data_it = sampled_idx[sampled_data_it]
        full_filename = f"Idx_{data_it}_{filename}"
        total_config['deco_text_fp'] = os.path.join(deco_text_dir, full_filename.replace(".json", ".txt"))

        # set X_T_o (init 1st decision token) and X_T_G (target 1st decision token)
        X_T_o, X_T_G, X_T_G_decoded = set_source_vs_target_token(
            args, crash_info, sampled_data_it, data_it, label, X_sys, X_vision
            )

        # Skip certain samples
        if skip_scenario(args, save_dir, full_filename, data_it, X_vision, X_T_G_decoded, tokenizer, dab_algo=True):
            continue

        if args.save_txt:
            with open(total_config['deco_text_fp'], 'a') as file:
                file.write(f"\n\nLabel switch goal: {X_T_o} => {X_T_G_decoded}\n\n")
                file.write("X_vision:\n" + X_vision + "\n")

        if total_config["fix_template"]:
            total_config["learnable_indices"] = get_fix_token_indices(args, X_vision, tokenizer, verbose=True) # idx are wrt X_vision
        else:
            total_config["learnable_indices"]=None

        encoded_x_sys = tokenizer([X_sys] * batch_size, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(total_config["device"]) # in ori code: "encoded_x_sys" is named "inputs"
        prompt_len = len(encoded_x_sys['input_ids'][0]) # ie, x_sys_len

        encoded_x_vision = tokenizer([X_vision] * batch_size, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(total_config["device"])
        x_vision_seq_len = len(encoded_x_vision['input_ids'][0])

        ori_bias_penalty = my_initialize_bias( # in dab repository, "bias_penalty" is named "cur_batch", shape [batch_size, x_vision_seq_len, vocab_size]
            model=model,
            total_config=total_config,
            batch_size=batch_size,
            ori_x_vision_ids=encoded_x_vision.input_ids,
            x_vision_seq_len=x_vision_seq_len
        )

        if total_config["use_Xo_topk_ids"] or total_config["use_Xo_topk_S2"]:
            with torch.no_grad():
                model.set_adapter("X_vision_gen")
                embedding_layer = model.get_input_embeddings()

                if total_config["learnable_indices"] is not None:
                    encoded_x_vision_input_ids = encoded_x_vision.input_ids[:,total_config["learnable_indices"]]
                else:
                    encoded_x_vision_input_ids = encoded_x_vision.input_ids
                # print("shape learnable xvision encoded tokens:", encoded_x_vision_input_ids.shape) # [bs=1, len(learnable_indices)]
                # print("deco learnable xvision encoded tokens:\n", tokenizer.batch_decode(encoded_x_vision_input_ids))
                X_vision_embeds = embedding_layer(encoded_x_vision_input_ids).detach()
                X_vision_embeds.requires_grad = False
                k = args.k_val_s2 if total_config["use_Xo_topk_S2"] else args.k_val
                X_vision_topk_nn_ids = torch.t(nn_project_fct(args, X_vision_embeds, embedding_layer=embedding_layer, K=k, print_hits=False)).unsqueeze(0).expand(batch_size,-1,-1) # shape [bs, len(learnable_indices), k_val]
                # print("X_vision_topk_nn_ids shape:", X_vision_topk_nn_ids.shape) # X_vision_topk_nn_ids shape: torch.Size([bs=1, len(learnable_indices), k])
                # for iterat in range(X_vision_topk_nn_ids.shape[1]):
                #     print("For:", tokenizer.decode(encoded_x_vision_input_ids[0,iterat]))
                #     print("nns are:", tokenizer.batch_decode(X_vision_topk_nn_ids[:,iterat,:]))

                del X_vision_embeds
                if total_config["use_Xo_topk_S2"]:
                    total_config["Xo_topk_for_S2"] = X_vision_topk_nn_ids.detach()


        if not total_config["add_Xo_reg_in_loss"] and not total_config["fix_template"]:
            del encoded_x_vision
            encoded_x_vision=None

        # prep storing var
        all_iteration_results=[]
        best_result = my_initialize_best_loss(batch_size)

        #----Main loop----

        for cur_step in tqdm(range(num_steps)):
            start_step = time.time()

            if cur_step == 0:
                bias_penalty = ori_bias_penalty.detach().clone()
            else:
                if total_config["Xo_only_in_bias"]:
                    # bias_penalty = ori_bias_penalty.detach().clone()
                    raise Exception("not implemented properly anymore")
                elif total_config["add_Xo_reg_in_bias"]:
                    bias_penalty = new_bias_penalty.detach().clone() #+ args.Xo_bias_weight_val * ori_bias_penalty.detach().clone()
                else:
                    bias_penalty = new_bias_penalty.detach().clone()


            # 1) Generating new X with biased AR gen + loss computation on the new seq (officially with B but using new X in practice (B=X))

            # output_ids: shape [batch_size, x_sys_len+x_vision_seq_len], T_tokens_ids: shape [batch_size, T] of gen Y tokens
            # onehot: values of onehot but backprop on soft values, shape [bs, x_vision_seq_len, vocab_size]
            rank_X_T_G, loss, sim_loss, ppl_loss, output_ids, T_tokens_ids, onehot, topk_ids = biased_ar_gen_and_loss(
                algo="DAB",
                HF_name=args.HF_name,
                model=model,
                tokenizer=tokenizer,
                loss_fct = loss_fct,
                add_ppl_loss=add_ppl_loss,
                encoded_x_vision=encoded_x_vision,
                bias_penalty=bias_penalty,
                Xo_bias_penalty=ori_bias_penalty,
                x_sys_input_ids=encoded_x_sys.input_ids,
                x_vision_seq_len=x_vision_seq_len,
                X_T_G=X_T_G,
                total_config=total_config,
                verbose=True, # if (cur_step>0 and cur_step<6) else False,
                return_rank=True
                )

            # 2) Sampling new bias seq B + computing logit bias penalty from it

            if total_config["use_Xo_topk_ids"]:
                topk_ids = X_vision_topk_nn_ids.detach().clone() # (bs, x_vision_seq, k) or (bs, len(learnable_indices), k)

            new_bias_penalty = bias_sampling_from_gradient( # in ori code, new_output_ids instead or output_ids
                model=model,
                tokenizer=tokenizer,
                loss=loss,
                output_ids=output_ids,
                onehot=onehot,
                topk_ids=topk_ids,
                prompt_len=prompt_len,
                total_config=total_config,
                encoded_x_vision=encoded_x_vision,
                verbose=True # if cur_step<5 else False
                )

            torch.cuda.empty_cache()

            # --- process and save iter results ---

            # get strings
            X_string = tokenizer.batch_decode(output_ids[:, prompt_len:]) # , skip_special_tokens=True
            Y_string = tokenizer.batch_decode(T_tokens_ids) # , skip_special_tokens=True

            # eval if Y_T_G found in Y_string
            inferred_classes=[get_class_from_string(elem) for elem in Y_string]
            constraint_satisfactions = [infer_label == X_T_G_decoded for infer_label in inferred_classes]

            # eval bertscore sim btw X_string and X_vision_o
            F1_list=semantic_similarity_bertscore(cands=X_string, refs=[X_vision]*batch_size, scorer=scorer)

            # updating best strings
            best_result = my_updating_best_loss(args, batch_size, cur_step, loss, ppl_loss, X_string, Y_string, inferred_classes, constraint_satisfactions, F1_list, best_result, Xo=X_sys+X_vision)

            # save info
            # template_fitness = [check_template_fitness(args, X_string[i], hard_control=True) for i in range(len(X_string))]

            iteration_result = {
                "X_string": X_string,
                "Y_string": Y_string,
                # "X_ids": output_ids[:, prompt_len:].detach().cpu().tolist(),
                # "Y_ids": T_tokens_ids.detach().cpu().tolist(),
                "loss": loss.item() if batch_size==1 else loss.detach().cpu().tolist(),
                "sim_score": F1_list,
                "ppl_loss": ppl_loss.item() if batch_size == 1 else ppl_loss.detach().cpu().tolist(),
                # "template_fitness": template_fitness,
                "step":cur_step,
                "XTG_rank":rank_X_T_G,
                "inferred_classes": inferred_classes,
                "constraint_satisfactions": constraint_satisfactions
                }
            all_iteration_results.append(iteration_result)

        # --- All optim iters finished on one data

        # complete the generation of best_y and compute final metrics
        final_result = finish_gen_and_eval_best_seq(
            args=args, model=model, tokenizer=tokenizer,
            X_sys=X_sys, X_vision=X_vision, Y=Y,
            best_x=best_result["best_x"][0], best_y=best_result["best_y"][0],
            learnable_indices=total_config["learnable_indices"] # wrt X_vision
            )
        best_result.update(final_result)

        one_data_result = {
            "X_T_G":X_T_G_decoded,
            "X_T_o":X_T_o,
            "X_vision":X_vision,
            "best_result":best_result,
            "iteration_results":all_iteration_results,
            "runtime_in_secs":round(time.time() - start, 2)
            }


        ### Freeing CUDA space
        del bias_penalty
        del new_bias_penalty
        del output_ids

        if save:
            with open(os.path.join(save_dir, full_filename), "w") as f:
                json.dump(one_data_result, f, indent=4)

    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
