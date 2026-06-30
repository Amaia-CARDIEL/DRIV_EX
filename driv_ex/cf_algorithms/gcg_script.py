# Copyright (c) 2026 Valeo. All rights reserved.
import sys
import math
import torch
from tqdm import tqdm
from driv_ex.utils.utils import *
from driv_ex.utils.shared_optim_utils import *
from driv_ex.utils.dab_utils import *
from driv_ex.utils.LLM_utils import *
from driv_ex.utils.driving_metrics import min_conditional_tok_prob
import torch.nn.functional as F
from driv_ex.utils.parsing_utils import *
from driv_ex.utils.shared_base_utils import *
from driv_ex.utils.brute_force_utils import *
from driv_ex.cf_algorithms.optim_config import get_args
from driv_ex.dataset import DATA_DIR
from driv_ex.dataset.textual_highD_dataset import textual_highD_dataset


def sample_control(learnable_indices, grad, topk=256, not_allowed_tokens=None): # temp=1, batch_size

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=-1).indices.squeeze(0).detach().cpu().tolist() # shape [nb_optim_tok, topk]

    # build my candidates_dict
    candidates_dict = {}
    for learn_pos_count, learn_pos_idx in enumerate(learnable_indices):
        candidates_dict[(learn_pos_idx,)] = [[cand_replace_tok] for cand_replace_tok in top_indices[learn_pos_count]]

    return candidates_dict


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))

    return cands



def main(args):

    s_script=time.time()

    ## Load frozen model and lora's adaptors
    set_seed(args.gcg_seed)
    no_config=False # if set to True, uses paper default config
    no_config_data=None
    single_tok=True
    model, tokenizer, scorer, _ = load_base_and_adaptors(
        args=args, FT_type=args.FT_type, no_config=no_config, return_no_config_data=True
        )

    eos_token_id = get_eos_token(args.HF_name)
    device=model.device
    max_bs = args.max_bs # 20 by default but can lead to oom
    # emb_dim = get_emb_dim(args)

    # avoid instable regions of Mistral and Qwen latent spaces
    valid_tok_ids = get_valid_vocab(args, tokenizer)

    # define loss function
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none') # mean reduction by default

    batch_size = 1
    total_config={"onehot": False, "use_Xo_topk_S2":False}

    ## Data
    val_dataset = textual_highD_dataset(
        HF_name=args.HF_name,
        bias_type=args.bias_type, # default to None
        input_prompt_only=False,
        with_reverse_bias=True if args.reverse_bias else False,
        with_labels=True,
        split_X_and_Y=True,
        fix_ego_car= True if args.optim_part == "sv" else False,
        TEST_ONE_DATA=True if args.max_data in [0, 1] else False # REMOVE LATER
        )

    # Retrieve llm ckpt name + safety-critical subset
    sampled_idx, crash_info, llm_ft_ckpts = retrieve_ckpt_specific_subset(args, no_config, no_config_data)
    val_dataset = torch.utils.data.Subset(val_dataset, sampled_idx)
    print(f"Launch optim on {len(val_dataset)} data")


    #--- Loop over dataset

    for sampled_data_it, (X_sys, X_vision, Y, label) in enumerate(val_dataset):

        real_data_it = sampled_idx[sampled_data_it]
        writer, save_filename, root_folder = define_writer(
            args, dataset_idx=real_data_it,
            label_choice_method=args.label_choice_method,
            llm_ft_ckpts=llm_ft_ckpts,
            algo="gcg"
            )

        # set X_T_o (init 1st decision token) and X_T_G (target 1st decision token)
        X_T_o, X_T_G, X_T_G_decoded = set_source_vs_target_token(
            args, crash_info, sampled_data_it, real_data_it, label, X_sys, X_vision
            )

        # Skip certain samples
        if skip_scenario(args, root_folder, save_filename, real_data_it, X_vision, X_T_G_decoded, tokenizer):
            continue

        text_tokens = tokenizer.tokenize(X_vision)
        init_token_id_list = tokenizer.convert_tokens_to_ids(text_tokens)
        learnable_indices = get_fix_token_indices(args, X_vision, tokenizer, verbose=True) # idx are wrt X_vision

        subvoc_idx = valid_tok_ids # None if llama3, valid tokens else
        total_config["learnable_indices"] = learnable_indices # idx are wrt X_vision
        encoded_x_sys = tokenizer([X_sys], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device) # in ori code: "encoded_x_sys" is named "inputs"

        if args.add_Xo_reg_in_loss:
            x_sys_len = len(encoded_x_sys['input_ids'][0]) # ie, prompt_len
            encoded_x_vision = tokenizer([X_vision] * batch_size, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)
            x_vision_seq_len = len(encoded_x_vision['input_ids'][0])

            with torch.no_grad():
                embedding_layer = model.get_input_embeddings()
                if total_config["learnable_indices"] is not None:
                    encoded_x_vision_input_ids = encoded_x_vision.input_ids[:,total_config["learnable_indices"]]
                else:
                    encoded_x_vision_input_ids = encoded_x_vision.input_ids
                print("shape learnable xvision encoded tokens:", encoded_x_vision_input_ids.shape) # [bs=1, len(learnable_indices)]
                print("deco learnable xvision encoded tokens:\n", tokenizer.batch_decode(encoded_x_vision_input_ids))
                X_vision_embeds = embedding_layer(encoded_x_vision_input_ids).detach()
                X_vision_embeds.requires_grad = False

        # Initialize soft embeddings (init_tok_ids/init_embeds_no_grad/optim_embeds/init_optim_ids have first dim bs=1 not K), image_features=None for LLM mode
        (
            init_tok_ids, init_embeds_no_grad, optim_tok_nn_ids, optim_embeds, optim_slice, modif_pos, seq_len, nb_optim_toks
            ) = init_all_embeds_vm(
            args=args, model=model, tokenizer=tokenizer, X_sys=X_sys, X_vision=X_vision
            )

        text_embedding = model.get_input_embeddings()

        # useful counters / bools
        _, best_prob_X_T_G, best_prob_X_T_G_for_all_K_nn, count = get_counters_and_booleans(args, nb_optim_toks)
        best_rank_X_T_G_wrt_prob=100000
        best_bertscore_wrt_prob=0


        ### ------------ LOOP ------------

        s_loop=time.time()
        for step in (range(args.num_steps)):
            print("\nStep", step+1, "/", args.num_steps)
            s_step=time.time()
            # model.zero_grad()

            optim_str = tokenizer.decode(optim_tok_nn_ids[0], skip_special_tokens=True)
            if args.sub_verbose:
                print(f"current optim tokens (step {step}):", optim_str)

            # pad text embeds with non modifiable tokens' embeds
            with torch.no_grad():
                tmp_onehot = torch.nn.functional.one_hot(
                                                optim_tok_nn_ids,
                                                num_classes=model.config.vocab_size, # text_embedding.shape[0]
                                            ).to(text_embedding.weight.dtype).to(model.device)
            tmp_onehot = tmp_onehot.clone().detach().requires_grad_(True) # tmp_onehot.requires_grad = True
            tmp_embeds = (tmp_onehot @ text_embedding.weight) # torch.Size([1, nb_optim_toks, emb_dim]) ?
            padded_embeds = insert_optim_embeds_into_full_encoding(tmp_embeds, init_embeds_no_grad, optim_slice) # shape [1, seq_len, emb_dim]

            # Generate T-1 tokens without grad + T-th with gradient
            curr_logits, T_tokens_ids = get_logits_at_step_T(
                HF_name=args.HF_name, model=model, tokenizer=tokenizer, padded_embeds=padded_embeds, total_config=total_config, check_requires_grad=True
                )

            loss = loss_fct(curr_logits, torch.tensor([X_T_G]).to(device)) # size 1

            if args.add_Xo_reg_in_loss:
                assert tmp_embeds.shape[-1] == X_vision_embeds.shape[-1] and tmp_embeds.shape[-2] == X_vision_embeds.shape[-2]
                sim_loss = 1 - (F.cosine_similarity(tmp_embeds, X_vision_embeds.squeeze(0), dim=-1).mean())
                loss = loss + args.xo_loss_weight * sim_loss

            # --Compute coordinate gradient
            grad, = torch.autograd.grad(loss, [tmp_onehot])  # optim_embeds_grad shape: torch.Size([bs=1, seqs_len=15, emb_dim])=> [1,nb_optim_toks,emb_dim] rather ?
            # loss.backward()
            # grad = tmp_onehot.grad.clone()
            grad = grad / grad.norm(dim=-1, keepdim=True)

            # Get prob, rank & loss => save to tensorboard
            Prob_X_T_G = (torch.exp(-loss.detach().clone())*100).item()
            rank_X_T_G = get_rank(curr_logits=curr_logits, X_T_G=X_T_G, device=device)
            current_loss = loss.item()
            if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                writing_main_metrics(writer, step, current_loss, Prob_X_T_G, rank_X_T_G)

            # --Sample a batch of new tokens based on the coordinate gradient.
            with torch.no_grad():
                candidates_dict = sample_control(
                    learnable_indices=learnable_indices,
                    grad=grad,
                    topk=args.gcg_topk,
                    not_allowed_tokens=None #  not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
                        )

                # --Compute loss on candidates and select argmin
                lowest_iter_loss, best_iter_optim_tok_ids, best_iter_rank, best_iter_p_xtg, best_iter_gen_ids_deco = 1000000, None, None, None, None
                for batch in tqdm(batched(combo_generator(init_token_id_list, candidates_dict, args.gcg_batch_size, single_tok, verbose=True if step==0 else False), max_bs)):
                    padded_token_combo, attention_mask = pad_sequences(batch, tokenizer.pad_token_id) # current_bs, full_x_vision_tok_len
                    cand_optim_toks = padded_token_combo[:, learnable_indices]
                    current_bs = padded_token_combo.shape[0]
                    padded_token_combo = padded_token_combo.to(model.device)
                    attention_mask = attention_mask.to(model.device)
                    full_input_ids = torch.cat((encoded_x_sys.input_ids.expand(current_bs,-1), padded_token_combo), dim=1)
                    full_attention_mask = torch.cat((encoded_x_sys.attention_mask.expand(current_bs,-1), attention_mask), dim=1)
                    assert full_input_ids.shape == full_attention_mask.shape

                    batch_loss, batch_rank_xtg, batch_prob_xtg, batch_gen_ids_deco, batch_final_pred_tok = eval_all_candidates(
                        model, tokenizer, full_input_ids, full_attention_mask, eos_token_id, loss_fct, X_T_G
                        )

                    if args.add_Xo_reg_in_loss:
                        # print("padded_token_combo.shape:", padded_token_combo.shape) # (current_bs, full_x_vision_tok_len=240)
                        # print("X_vision_embeds.shape:", X_vision_embeds.shape) # ([1, nb_optim_tok=30, emb_dim])
                        candidate_embeds = text_embedding(cand_optim_toks) # dim (current_bs, nb_optim_tok, emb_dim) ?
                        cos_sim = F.cosine_similarity(candidate_embeds, X_vision_embeds, dim=-1)  # (bs, nb_optim_tok)
                        batch_cos_sim = cos_sim.mean(dim=-1)             # (bs,)
                        batch_loss = batch_loss + batch_cos_sim # dim current_bs

                    best_batch_idx = torch.argmin(batch_loss).item()
                    best_batch_loss = batch_loss[best_batch_idx].item()
                    if best_batch_loss<lowest_iter_loss:
                        lowest_iter_loss = best_batch_loss
                        best_iter_optim_tok_ids = cand_optim_toks[best_batch_idx]
                        best_iter_rank = batch_rank_xtg[best_batch_idx]
                        best_iter_p_xtg = batch_prob_xtg[best_batch_idx]
                        best_iter_gen_ids_deco = batch_gen_ids_deco[best_batch_idx]

            # candidate with lowest loss is kept for next iteration
            assert best_iter_optim_tok_ids is not None
            optim_tok_nn_ids = best_iter_optim_tok_ids
            rank_X_T_G=best_iter_rank
            Prob_X_T_G=best_iter_p_xtg
            gen_ids_deco = best_iter_gen_ids_deco
            current_loss = lowest_iter_loss


            ### ------- Eval and update best values if current iter is better than historic best ------

            new_best_data = False
            if (rank_X_T_G==0 and best_rank_X_T_G_wrt_prob>0) or (rank_X_T_G>0 and best_rank_X_T_G_wrt_prob>0 and Prob_X_T_G > best_prob_X_T_G):
                new_best_data = True

            all_id_combi = init_tok_ids.detach().clone()
            all_id_combi[:, optim_slice == 1] = optim_tok_nn_ids.detach().clone().to(all_id_combi.device) # size [K=1, nb_optim_toks]
            all_decoded_texts = tokenizer.batch_decode(all_id_combi) # len total_nb_forwards
            all_decoded_texts = [elem.replace(X_sys, "") for elem in all_decoded_texts]
            F1_list=semantic_similarity_bertscore(cands=all_decoded_texts, refs=[X_vision]*len(all_decoded_texts), scorer=scorer)

            if (rank_X_T_G==0 and best_rank_X_T_G_wrt_prob==0 and F1_list[0]>best_bertscore_wrt_prob):
                new_best_data = True

            if new_best_data:

                full_prob_X_T_G = [Prob_X_T_G]
                all_y_decoded_texts = [gen_ids_deco] # tokenizer.batch_decode(T_tokens_ids)

                if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                    eval_data = f"\nEval at step: {step}, curr loss: {current_loss:.3f}, curr P(X_T_G|...): {round(Prob_X_T_G,4)} %, curr rank X_T_G: {rank_X_T_G}, curr CE:"
                    for one_prob_X_T_G, one_text, one_y, one_f1 in zip(full_prob_X_T_G, all_decoded_texts, all_y_decoded_texts, F1_list):
                        short_text = one_text #.replace('\n', ' ')
                        one_proba = round(one_prob_X_T_G.item(),4) if args.K>1 else round(one_prob_X_T_G,4)
                        eval_data += f"\nX_vision:\n'{short_text}'\n\nY:\n'{one_y}'\n    P(X_T_G|...)={one_proba}%\n    Bertsc(Xo,Xm)={one_f1}\n"
                    try:
                        with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                            file.write(eval_data + "\n")
                    except:
                        pass

                timing = time.time() - s_step
                best_prob_X_T_G = Prob_X_T_G
                best_loss_wrt_prob = current_loss
                best_text_wrt_prob = all_decoded_texts[0]
                best_y_wrt_prob = all_y_decoded_texts[0]
                best_bertscore_wrt_prob = F1_list[0]
                best_step_wrt_prob = step
                best_rank_X_T_G_wrt_prob = rank_X_T_G
                best_seq_ids = all_id_combi[0].unsqueeze(0).cpu().tolist()
                if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                    add_metric(x=step, y=best_prob_X_T_G, name="1) Best_P_XTG", writer=writer)


        # --- End of loop for current data ---

        # save best result wrt proba in json format
        if not args.dont_record_metrics:

            json_results = {}

            # Add GT data
            json_results["X_T_G_decoded"] = X_T_G_decoded
            json_results["X_T_o_decoded"] = X_T_o

            # Add init_data
            init_data_dico = {}
            full_seq_deco = tokenizer.batch_decode(init_tok_ids)[0]
            init_data_dico["seq_decoded"] = full_seq_deco
            json_results["init_data"] = init_data_dico

            # Add best results wrt prob
            result = finish_gen_and_eval_best_seq(
                args=args, model=model, tokenizer=tokenizer,
                X_sys=X_sys, X_vision=X_vision, Y=Y,
                best_x=best_text_wrt_prob, best_y=best_y_wrt_prob,
                learnable_indices=learnable_indices # wrt X_vision
                )

            Best_result_wrt_prob = {
                    # Y_modif and its metrics
                    "y_seq_decoded": result["best_y_full"], # best_y_wrt_prob,
                    "XTG_rank": best_rank_X_T_G_wrt_prob,
                    "Best_P_XTG_%": best_prob_X_T_G,
                    "loss": best_loss_wrt_prob,
                    "lc_traj_coherence": result["lc_traj_coherence"] if "lc_traj_coherence" in result else None,
                    "maneuver_eval_multi_sec": result["man_eval_multi_sec"]  if "man_eval_multi_sec" in result else None,
                    "rmse_lon": result["rmse_lon"] if "rmse_lon" in result else None,
                    "rmse_lat": result["rmse_lat"] if "rmse_lat" in result else None,
                    # X_modif and its metrics
                    "x_seq_decoded": best_text_wrt_prob,
                    "template_fitness":result["template_fitness"] if "template_fitness" in result else None,
                    "sim_score": best_bertscore_wrt_prob,
                    "min_cond_tok_prob":result["min_cond_tok_prob"],
                    # Other info on finding X_modif
                    "best_step": best_step_wrt_prob,
                    "best_timing_in_secs": timing,
                    "x_seq_ids": best_seq_ids
                }

            json_results["Best_result_wrt_prob"]=Best_result_wrt_prob

            # Add other info
            json_results["other_info"]={
                "total_optim_runtime_in_secs": round(time.time() - s_loop, 2)
            }

            with open(os.path.join(root_folder, "best_results_json", save_filename+'.json'), "w") as outfile:
                json.dump(json_results, outfile, indent=4)


        # save / print best result wrt proba in txt format
        final_res_string = f"\n\nResults for highest P(X_T_G|...):\nloss: {best_loss_wrt_prob:.3f}\nX_T_G's rank: {best_rank_X_T_G_wrt_prob}\nP(X_T_G|...)={round(best_prob_X_T_G,4)}%"
        short_text = best_text_wrt_prob.replace(X_sys, "") # .replace('\n', ' ')
        final_res_string += f"\nBest CE (1st proj) wrt P(X_T_G|...) found at step {best_step_wrt_prob} (in {round(timing,2)} secs):\n{short_text}"
        print(final_res_string)

        # runtime
        e=time.time()
        runtime_string = f"\nRuntime for current dataset optim: {round(e-s_loop, 2)} secs"
        print(runtime_string)
        if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
            try:
                with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                    file.write(runtime_string)
            except:
                pass

        if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
            add_metric(x=step, y=best_prob_X_T_G, name="1) Best_P_XTG", writer=writer)
            add_metric(x=step, y=best_prob_X_T_G_for_all_K_nn, name="1) Best_P_XTG_for_all_K", writer=writer)
            writer.close()

    print("\n\nFull script runtime:", round(time.time()-s_script, 2), "secs")

    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
