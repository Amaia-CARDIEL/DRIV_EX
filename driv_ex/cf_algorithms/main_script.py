# Copyright (c) 2026 Valeo. All rights reserved.
import math
import torch
from driv_ex.utils.utils import *
from torchvision.ops import box_iou
from tqdm import tqdm
from driv_ex.utils.shared_optim_utils import *
from driv_ex.utils.dab_utils import *
from driv_ex.utils.LLM_utils import *
from driv_ex.utils.driving_metrics import min_conditional_tok_prob
import torch.nn.functional as F
from driv_ex.utils.parsing_utils import *
from driv_ex.utils.shared_base_utils import *
from driv_ex.cf_algorithms.optim_config import get_args
from driv_ex.dataset import DATA_DIR
from driv_ex.dataset.textual_highD_dataset import textual_highD_dataset


def main(args):

    s_script=time.time()
    verbose = True

    ## Load frozen model and required LoRA's adaptors (for Driving LLM and optionally Fluency expert)
    set_seed(42)
    no_config=False # if set to True, uses paper default config
    no_config_data=None
    model, tokenizer, scorer, _ = load_base_and_adaptors(
        args=args, FT_type=args.FT_type, no_config=no_config, return_no_config_data=True
        )
    device=model.device
    emb_dim = get_emb_dim(args)

    # Avoid instable regions of Mistral and Qwen latent spaces
    valid_tok_ids = get_valid_vocab(args, tokenizer)

    # Define loss function
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none') # mean reduction by default

    # Total_config inherited from dab to add biased autoregressive step
    batch_size = 1
    if args.dab_reg:
        total_config = {
            "logit_weight_val":args.logit_weight_val,
            "weight_val":args.weight_val,
            "k_val":args.k_val,
            "use_fadding_weights":False,
            "use_bolt_weights":False,
            "use_scale_weights":True,
            "fix_template":True,
            "fix_all_non_learnable_indices":True,
            "add_Xo_reg_in_loss":False, # WARNING: always set to False here not to trigger it in the dab reg functions if used
            "xo_loss_weight":0, # same as above here
            "add_Xo_reg_in_bias":args.add_Xo_reg_in_bias,
            "Xo_bias_weight_val":args.Xo_bias_weight_val,
            "initialization":"from_ori_X",
            "learnable_indices":[], # filled later if needed
            "ar_gen_equation":"ours",
            "device":device,
            "onehot":args.onehot,
            "save_txt":False,
            "use_Xo_topk_S2":args.use_Xo_topk_S2,
        }
    else:
        total_config={"onehot":False, "use_Xo_topk_S2":False}

    # Load dataset
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

    # Retrieve llm ckpt name + slice dataset to safety-critical subset
    sampled_idx, crash_info, llm_ft_ckpts = retrieve_ckpt_specific_subset(args, no_config, no_config_data)
    val_dataset = torch.utils.data.Subset(val_dataset, sampled_idx)
    print(f"Launch optim on {len(val_dataset)} data")


    # --- Loop over dataset

    for sampled_data_it, (X_sys, X_vision, Y, label) in enumerate(val_dataset):

        real_data_it = sampled_idx[sampled_data_it]
        writer, save_filename, root_folder = define_writer(
            args, dataset_idx=real_data_it,
            label_choice_method=args.label_choice_method,
            llm_ft_ckpts=llm_ft_ckpts,
            algo="drivex" if args.dab_reg else "pez"
            )

        # Set X_T_o (initial 1st decision token) and X_T_G (target 1st decision token)
        X_T_o, X_T_G, X_T_G_decoded = set_source_vs_target_token(
            args, crash_info, sampled_data_it, real_data_it, label, X_sys, X_vision
            )

        # Skip certain samples
        if skip_scenario(args, root_folder, save_filename, real_data_it, X_vision, X_T_G_decoded, tokenizer):
            continue

        # Define which token indices can be changed by the algorithm
        learnable_indices = get_fix_token_indices(args, X_vision, tokenizer, verbose=True) # idx are wrt X_vision

        subvoc_idx = valid_tok_ids
        if args.dab_reg or args.add_Xo_reg_in_loss or args.proj_subvoc:
            # Tokenize X_sys and X_vision
            total_config["learnable_indices"] = learnable_indices # idx are wrt X_vision
            encoded_x_sys = tokenizer([X_sys], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device) # in ori code: "encoded_x_sys" is named "inputs"
            x_sys_len = len(encoded_x_sys['input_ids'][0]) # ie, prompt_len
            encoded_x_vision = tokenizer([X_vision] * batch_size, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)
            x_vision_seq_len = len(encoded_x_vision['input_ids'][0])

            if total_config["use_Xo_topk_S2"] or args.add_Xo_reg_in_loss or args.proj_subvoc:
                with torch.no_grad():
                    # Embed X_vision tokens
                    model.set_adapter("X_vision_gen")
                    embedding_layer = model.get_input_embeddings()
                    if total_config["learnable_indices"] is not None:
                        encoded_x_vision_input_ids = encoded_x_vision.input_ids[:,total_config["learnable_indices"]]
                    else:
                        encoded_x_vision_input_ids = encoded_x_vision.input_ids
                    print("shape learnable xvision encoded tokens:", encoded_x_vision_input_ids.shape) # [bs=1, len(learnable_indices)]
                    print("deco learnable xvision encoded tokens:\n", tokenizer.batch_decode(encoded_x_vision_input_ids))
                    X_vision_embeds = embedding_layer(encoded_x_vision_input_ids).detach()
                    X_vision_embeds.requires_grad = False

                    if args.proj_subvoc:
                        # Precompute eligible subvocabulary per learnable index, if proj_subvoc is used
                        subvoc_idx = torch.t(nn_project_fct(args, X_vision_embeds, embedding_layer=embedding_layer, K=args.subvoc_size, subvoc_idx=valid_tok_ids, print_hits=False)).unsqueeze(0).expand(batch_size,-1,-1)[0].detach().to(device) #.tolist() # tensor of size [len(learnable), subvoc_size]
                        assert subvoc_idx.shape[1] == args.subvoc_size
                        if args.sub_verbose:
                            print("details of suproj per learnable token it:")
                            for sub_it, sub in enumerate(subvoc_idx):
                                print("For:", tokenizer.decode(encoded_x_vision_input_ids[0,sub_it]))
                                print("nns ids are:", sub)
                                print("nns deco are:", tokenizer.decode(sub))

                    if total_config["use_Xo_topk_S2"]:
                        # Precompute eligible restricted list of candidates per learnable index, if use_Xo_topk_S2 is used
                        X_vision_topk_nn_ids = torch.t(nn_project_fct(args, X_vision_embeds, embedding_layer=embedding_layer, K=args.k_val, subvoc_idx=valid_tok_ids, print_hits=False)).unsqueeze(0).expand(batch_size,-1,-1) # shape [bs, len(learnable_indices), k_val]
                        if verbose:
                            for iterat in range(X_vision_topk_nn_ids.shape[1]):
                                print("For:", tokenizer.decode(encoded_x_vision_input_ids[0,iterat]))
                                print("nns are:", tokenizer.batch_decode(X_vision_topk_nn_ids[:,iterat,:]))
                            print("X_vision_topk_nn_ids shape:", X_vision_topk_nn_ids.shape) # X_vision_topk_nn_ids shape: torch.Size([bs=1, len(learnable_indices), k])
                        total_config["Xo_topk_for_S2"] = X_vision_topk_nn_ids.detach()

                    if not args.add_Xo_reg_in_loss:
                        del X_vision_embeds

            if args.add_Xo_reg_in_bias:
                # Precompute minimality bias wrt Xo
                ori_bias_penalty = my_initialize_bias( # in ori code, "bias_penalty" is named "cur_batch", shape [batch_size, x_vision_seq_len, vocab_size]
                    model=model,
                    total_config=total_config,
                    batch_size=batch_size,
                    ori_x_vision_ids=encoded_x_vision.input_ids,
                    x_vision_seq_len=x_vision_seq_len
                )
            else:
                ori_bias_penalty = None

        # Initialize soft embeddings with initial driving scene (init_tok_ids/init_embeds_no_grad/optim_embeds/init_optim_ids have first dim bs=1 not K)
        (
            init_tok_ids, init_embeds_no_grad, init_optim_ids, optim_embeds, optim_slice, modif_pos, seq_len, nb_optim_toks
            ) = init_all_embeds_vm(
            args=args, model=model, tokenizer=tokenizer, X_sys=X_sys, X_vision=X_vision
            )

        # Initialize optimizer and loss
        optimizer = torch.optim.AdamW([optim_embeds], lr=args.lr, weight_decay=args.weight_decay)

        # Define nb_forwards and useful counters / bools
        total_nb_forwards, best_prob_X_T_G, best_prob_X_T_G_for_all_K_nn, count = get_counters_and_booleans(args, nb_optim_toks)
        best_rank_X_T_G_wrt_prob=100000
        best_bertscore_wrt_prob=0


        ### ------------ OPTIM LOOP ------------

        s_loop=time.time()
        for step in (range(args.num_steps)):
            s_step=time.time()
            reinit_str = ""

            # Test if the result of vocab projection of soft embeddings changed
            with torch.no_grad():

                if step == 0:
                    # Proj on voc of soft optimized embeddings: projected_embeds of size [K, nb_optim_toks, emb_dim], optim_tok_nn_ids of size [K, nb_optim_toks], cos_sim_per_nn of size K
                    optim_tok_nn_ids = nn_project_fct(args=args, curr_embeds=optim_embeds, model=model, set_adapter=True, subvoc_idx=subvoc_idx) # here subvoc_idx not needed as it projects onto itself anyway so faster not to do subvoc proj (?)
                    last_tok_nn_ids = optim_tok_nn_ids.detach().clone()

                if step >= 1:

                    # Optional: "hard" reinit of optimized embeds to discrete token embeds values
                    if count["reinit_hard"] or (args.dab_reg and args.dab_reg_reinit_hard and step % args.reinit_mult ==0):
                        print("Reinit at step", step)
                        if verbose:
                            all_current_ids_for_best_K = init_tok_ids.detach().clone()
                            all_current_ids_for_best_K[:, optim_slice == 1] = best_optim_tok_ids_for_all_K # size [1, seq_len]
                            text_for_best_proba_K_proj = tokenizer.batch_decode(all_current_ids_for_best_K)[0]
                            reinit_str = f"\n   Reinit hard to: '{text_for_best_proba_K_proj}'"

                        # Reinit optim_embeds, its K proj and optimizer
                        optim_embeds = re_init_all_hard(args, best_optim_tok_ids_for_all_K, model=model) # no gradient info
                        optimizer = reinit_optimizer(args, optimizer, optim_embeds, keep_state = True)
                        count["reinit_hard_count"] += 1
                        grad_changes = True

                    # Proj on voc of soft optimized embeddings
                    optim_tok_nn_ids = nn_project_fct(args=args, curr_embeds=optim_embeds, model=model, set_adapter=True, subvoc_idx=subvoc_idx)
                    current_tok_nn_ids = optim_tok_nn_ids.detach().clone()
                    if args.sub_verbose:
                        print("after subproj:", tokenizer.batch_decode(optim_tok_nn_ids))

                    if not count["reinit_hard"] or args.dab_reg:

                        # Optional: if top 1 nearest proj changed, re-init all but one
                        if args.one_by_one in ["re_init_all_but_one"]:
                            modif_tok_ids_1st_proj = [i for i in range(last_tok_nn_ids.shape[1]) if last_tok_nn_ids[0][i]!=current_tok_nn_ids[0][i]]
                            count["need_reinit"] = True if len(modif_tok_ids_1st_proj)>0 else False

                            if count["need_reinit"]:
                                # Reinit optim_embeds, its K proj and optimizer
                                optim_embeds = re_init_all_but_one(args, modif_pos, modif_tok_ids_1st_proj, optim_embeds, last_tok_nn_ids, optim_embeds_grad, model=model)
                                optimizer = reinit_optimizer(args, optimizer, optim_embeds, keep_state = True)
                                optim_tok_nn_ids = nn_project_fct(args=args, curr_embeds=optim_embeds, model=model, set_adapter=True, subvoc_idx=subvoc_idx)
                                current_tok_nn_ids = optim_tok_nn_ids.detach().clone()

                        # Test if unordered lists of top K projection results changed
                        v_last, _ = torch.sort(last_tok_nn_ids, dim =0)
                        v_current, _ = torch.sort(current_tok_nn_ids, dim =0)
                        grad_changes = (torch.sum(v_last == v_current) / torch.numel(v_current)).item() < 1

                        # Optional: sanity check for "re_init_all_but_one":
                        if args.one_by_one in ["re_init_all_but_one"]:
                            if count["need_reinit"] and not grad_changes:
                                count["c_reinit_but_no_grad_change"] += 1
                            if not count["need_reinit"] and grad_changes:
                                count["c_no_reinit_but_grad_change"] += 1

                    last_tok_nn_ids = current_tok_nn_ids.detach().clone()


            # Recompute gradient **only if projection result changed**
            if step == 0 or grad_changes:

                if verbose:
                    print(f"\nStep {step} => forward", reinit_str)
                count["c_nb_forward"] += 1

                # Set last update's gradient to zero (otherwise we keep it)
                optimizer.zero_grad(set_to_none=True)

                # Pad optimized embeddings with non modifiable tokens' embeddings
                with torch.no_grad():
                    model.set_adapter("X_vision_gen")
                    text_embedding = model.get_input_embeddings()
                    tmp_embeds = text_embedding(optim_tok_nn_ids).detach() # torch.Size([K=1, nb_optim_toks, emb_dim]) like optim_embeds_grad
                    tmp_embeds.requires_grad = True
                padded_embeds = insert_optim_embeds_into_full_encoding(tmp_embeds, init_embeds_no_grad, optim_slice) # shape [1, seq_len, emb_dim]

                # Deprecated compute of perplexity loss
                if not args.dont_record_metrics and not args.dont_record_ppl_loss and subvoc_idx is None:
                    with torch.no_grad():
                        output_ids = nn_project_fct(args=args, curr_embeds=padded_embeds, model=model, set_adapter=True, subvoc_idx=subvoc_idx)
                        ppl_loss = model(inputs_embeds=padded_embeds, labels=output_ids).loss.detach().item()

                # Generate T-1 tokens without grad + T-th token with gradient
                curr_logits, T_tokens_ids = get_logits_at_step_T(
                    HF_name=args.HF_name, model=model, tokenizer=tokenizer, padded_embeds=padded_embeds, total_config=total_config, check_requires_grad=True
                    )

                # Compute core decision loss
                loss = loss_fct(curr_logits, torch.tensor([X_T_G]).to(device)) # size 1

                # Optional: add Xo reg in loss
                if args.add_Xo_reg_in_loss:
                    assert tmp_embeds.shape == X_vision_embeds.shape
                    sim_loss = 1 - (F.cosine_similarity(tmp_embeds, X_vision_embeds, dim=-1).mean())
                    loss = loss + args.xo_loss_weight * sim_loss

                # Compute gradient and prob of target token X_T_G
                optim_embeds_grad, = torch.autograd.grad(loss, [tmp_embeds])  # optim_embeds_grad shape: torch.Size([bs=1, seqs_len=15, emb_dim])=> [1,nb_optim_toks,emb_dim] rather ?
                K_prob_X_T_G = torch.exp(-loss.detach().clone())*100
                Prob_X_T_G = K_prob_X_T_G.item()

                # Optional / deprecated: apply filter onto gradient if required
                if args.one_by_one in ["biggest_norm", "seq_order", "seq_inv_order"]:
                    optim_embeds_grad = keep_grad_for_single_token(args=args, final_grad=optim_embeds_grad, modif_pos=modif_pos, update_step=count["update_step"])
                    if args.one_by_one in ["seq_order", "seq_inv_order"]:
                        count["update_step"]+=1

                # Set the computed gradient to optim_embeds tensor
                optim_embeds.grad = optim_embeds_grad.to(optim_embeds.dtype)

            # Optional: monitor gradient norms
            if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                try:
                    writing_grad_norm_metrics(writer, step, optim_embeds)
                except:
                    print("failure to write grad norm")

            # Perform optim step
            optimizer.step()

            # Optional: DAB reg
            if args.dab_reg:

                # Transform current optimized embeddings into bias penalty for auto-regressive biasing
                optim_tok_nn_ids_proj_for_reg = nn_project_fct(
                    args=args, curr_embeds=optim_embeds, model=model, set_adapter=True, subvoc_idx=subvoc_idx)

                bias_penalty, deco_before_ar = bias_pen_from_bias_seq(
                    optim_tok_nn_ids_proj_for_reg, # input shape should be [batch_size, nb_optim, vocab_size]
                    model,
                    tokenizer,
                    total_config,
                    encoded_x_vision,
                    verbose=True
                    )
                print(deco_before_ar)

                # Saving temp result, before autoregressive biasing, under txt format
                try:
                    with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                        file.write(deco_before_ar + "\n")
                except:
                    pass

                # Regerating the driving scene with biased AR gen + loss / rank / prob computation on the new seq
                ### all_id_combi: shape [batch_size, x_sys_len+x_vision_seq_len], T_tokens_ids: shape [batch_size, T] of gen Y tokens
                ### onehot: values of onehot but backprop on soft values, shape [bs, x_vision_seq_len, vocab_size]
                rank_X_T_G, loss, _, _, all_id_combi, T_tokens_ids, _, _ = biased_ar_gen_and_loss(
                    algo="PEZ",
                    HF_name=args.HF_name,
                    model=model,
                    tokenizer=tokenizer,
                    loss_fct = loss_fct,
                    add_ppl_loss=False,
                    encoded_x_vision=encoded_x_vision,
                    bias_penalty=bias_penalty,
                    Xo_bias_penalty=ori_bias_penalty,
                    x_sys_input_ids=encoded_x_sys.input_ids,
                    x_vision_seq_len=x_vision_seq_len,
                    X_T_G=X_T_G,
                    total_config=total_config,
                    verbose=True if (sampled_data_it>0 and sampled_data_it<6) else False,
                    return_rank=True
                    )
                K_prob_X_T_G = torch.exp(-loss.detach().clone())*100
                Prob_X_T_G = K_prob_X_T_G.item()
                all_decoded_texts = tokenizer.batch_decode(all_id_combi)

                # Saving current result under txt format
                print(f"\n\n[X_vision after AR reg]:\n{all_decoded_texts[0]}")
                try:
                    with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                        file.write(f"\n\nX_vision after AR reg:\n{all_decoded_texts[0]}" + "\n")
                except:
                    pass

                # Optional / deprecated
                if args.dab_reg_reinit_hard:
                    best_optim_tok_ids_for_all_K = all_id_combi[0, optim_slice == 1].unsqueeze(0)

            # If no DAB reg
            else:
                # Compute rank of target token
                rank_X_T_G = get_rank(curr_logits=curr_logits, X_T_G=X_T_G, device=device)

            # Save loss in tensorboard
            current_loss = loss.item()
            if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                writing_main_metrics(writer, step, current_loss, Prob_X_T_G, rank_X_T_G)


            ### ----- Eval and update new best seq if relevant ------

            # -- Test if current data is the best --
            new_best_data = False

            # When saved best data does not lead X_T_G to be generated greedily (best_rank_X_T_G_wrt_prob>0)
            # current data can become best if a) it enables target token to be of rank 0 or b) if it improves its proba
            if (rank_X_T_G==0 and best_rank_X_T_G_wrt_prob>0) or (rank_X_T_G>0 and best_rank_X_T_G_wrt_prob>0 and Prob_X_T_G > best_prob_X_T_G):
                new_best_data = True

            # Reconstruct full candidate sequences and compute bertscore similarity (using F1) with Xo (F1_list)
            if not args.dab_reg:
                all_id_combi = init_tok_ids.detach().clone()
                all_id_combi[:, optim_slice == 1] = optim_tok_nn_ids.detach().clone() # size [K=1, nb_optim_toks]
            all_decoded_texts = tokenizer.batch_decode(all_id_combi) # len total_nb_forwards
            all_decoded_texts = [elem.replace(X_sys, "") for elem in all_decoded_texts]
            F1_list=semantic_similarity_bertscore(cands=all_decoded_texts, refs=[X_vision]*len(all_decoded_texts), scorer=scorer)

            # when saved best data leads X_T_G to be generated greedily (best_rank_X_T_G_wrt_prob==0)
            # current data can become best if c) it also has rank==0 and it leads to a higher bertscore with Xo
            if (rank_X_T_G==0 and best_rank_X_T_G_wrt_prob==0 and F1_list[0]>best_bertscore_wrt_prob):
                new_best_data = True

            # -- Update best saved values if current data is the new best --
            if new_best_data:

                # Update values
                full_prob_X_T_G = [Prob_X_T_G]
                all_y_decoded_texts = tokenizer.batch_decode(T_tokens_ids)
                timing = time.time() - s_step
                best_prob_X_T_G = Prob_X_T_G
                best_loss_wrt_prob = current_loss
                best_text_wrt_prob = all_decoded_texts[0]
                best_y_wrt_prob = all_y_decoded_texts[0]
                best_bertscore_wrt_prob = F1_list[0]
                best_step_wrt_prob = step
                best_rank_X_T_G_wrt_prob = rank_X_T_G
                best_seq_ids = all_id_combi[0].unsqueeze(0).cpu().tolist()
                if not args.dont_record_metrics and not args.dont_record_ppl_loss and subvoc_idx is None:
                    best_ppl_loss_wrt_prob = ppl_loss

                # Optional: save updated best values under txt format
                if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                    eval_data = f"\nEval at step: {step}, curr loss: {current_loss:.3f}, curr P(X_T_G|...): {round(Prob_X_T_G,4)} %, curr rank X_T_G: {rank_X_T_G}, curr CE:"
                    for one_prob_X_T_G, one_text, one_y, one_f1 in zip(full_prob_X_T_G, all_decoded_texts, all_y_decoded_texts, F1_list):
                        short_text = one_text
                        one_proba = round(one_prob_X_T_G,4)
                        eval_data += f"\nX_vision:\n'{short_text}'\n\nY:\n'{one_y}'\n    P(X_T_G|...)={one_proba}%\n    Bertsc(Xo,Xm)={one_f1}\n"
                    try:
                        with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                            file.write(eval_data + "\n")
                    except:
                        pass

                # Optional: save via tensorboard
                if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
                    add_metric(x=step, y=best_prob_X_T_G, name="1) Best_P_XTG", writer=writer)

            ### ----- End "Eval and update" section ------


        # ------ End of optim loop for current data => saving best coutnerfactual for this sample ------

        # Optional / deprecated: sanity check for "re_init_all_but_one":
        if args.one_by_one in ["re_init_all_but_one", "re_init_hard"]:
            print("# re init but no grad change:", count["c_reinit_but_no_grad_change"])
            print("# no re init but grad change:", count["c_no_reinit_but_grad_change"])

        if args.modif_pos != [0]:
            best_optim_tok_nn_ids = [best_seq_ids[0][elem] for elem in args.modif_pos]

        # Save best counterfactual info in json format
        if not args.dont_record_metrics:

            json_results = {}

            # Add GT data
            json_results["X_T_G_decoded"] = X_T_G_decoded
            json_results["X_T_o_decoded"] = X_T_o

            # Add init_data
            init_data_dico = {}
            full_seq_deco = tokenizer.batch_decode(init_tok_ids)[0]
            init_data_dico["seq_ids"] = init_tok_ids.detach().cpu().tolist()
            init_data_dico["seq_decoded"] = full_seq_deco
            json_results["init_data"] = init_data_dico

            # Get further eval info on best retained counterfactual sequence
            result = finish_gen_and_eval_best_seq(
                args=args, model=model, tokenizer=tokenizer,
                X_sys=X_sys, X_vision=X_vision, Y=Y,
                best_x=best_text_wrt_prob, best_y=best_y_wrt_prob,
                learnable_indices=learnable_indices # wrt X_vision
                )

            # Add main info to json_results
            Best_result_wrt_prob = {
                    # Y_modif and its metrics
                    "y_seq_decoded": result["best_y_full"],
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

            if not args.dont_record_ppl_loss and subvoc_idx is None:
                Best_result_wrt_prob["ppl_loss"]=best_ppl_loss_wrt_prob
                Best_result_wrt_prob["ppl"]=math.exp(best_ppl_loss_wrt_prob)

            json_results["Best_result_wrt_prob"]=Best_result_wrt_prob

            # Add other info
            json_results["other_info"]={
                "c_nb_forward": count["c_nb_forward"],
                "total_optim_runtime_in_secs": round(time.time() - s_loop, 2)
            }

            # Save under json format
            with open(os.path.join(root_folder, "best_results_json", save_filename+'.json'), "w") as outfile:
                json.dump(json_results, outfile, indent=4)


        # Print best result
        best_optim_tok_nn_ids_to_print = best_optim_tok_nn_ids if args.modif_pos != [0] else best_seq_ids
        final_res_string = f"\n\nResults for highest P(X_T_G|...):\nloss: {best_loss_wrt_prob:.3f}\nX_T_G's rank: {best_rank_X_T_G_wrt_prob}\nP(X_T_G|...)={round(best_prob_X_T_G,4)}%"
        short_text = best_text_wrt_prob.replace(X_sys, "")
        final_res_string += f"\nBest CE (1st proj) wrt P(X_T_G|...) found at step {best_step_wrt_prob} (in {round(timing,2)} secs):\n{short_text}"
        print(final_res_string)
        print(f"\n{count['c_nb_forward']} forwards / {args.num_steps} steps")
        if args.reinit_hard_mode:
            print(f"\n{count['reinit_hard_count']} hard reinits / {args.num_steps} steps")

        # Optional: save info under txt format
        if not args.dont_record_metrics and not args.no_tensorboard_nor_deco:
            try:
                with open(os.path.join(root_folder, "decoded_text", save_filename+'.txt'), 'a') as file:
                    file.write(final_res_string + "\n")
            except:
                pass

        # Print / save runtime
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

    try:
        print("reinit_hard_count", count["reinit_hard_count"] )
    except:
        print("no inferences done with reinit")
    print("\n\nFull script runtime:", round(time.time()-s_script, 2), "secs")

    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
