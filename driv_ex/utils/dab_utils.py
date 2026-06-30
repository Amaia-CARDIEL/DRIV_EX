# Copyright (c) 2026 Valeo. All rights reserved.
"""DAB (Discrete Auto-regressive Biasing) algorithm internals: biased autoregressive
generation, gradient-based token distribution updates, loss computation, and
per-step best-result tracking."""
import os
import re
import time
import json
import random
from tqdm import tqdm
import torch
from driv_ex.utils.LLM_utils import load_HF_model_tok, load_base_and_adaptors
from driv_ex.utils.parsing_utils import get_class_from_string
from driv_ex.utils.shared_optim_utils import get_rank, finish_gen_and_eval_best_seq
from driv_ex.utils.shared_base_utils import *
from driv_ex.utils.driving_metrics import check_template_fitness, min_conditional_tok_prob
import torch.nn.functional as F




def soft_hard_trick(tokens, tokens_scores, model, requires_grad=True):

    # tokens_scores is a list of gen_len tensors of shape bs, vocab_size
    # tokens shape bs, gen_len
    batch_size = tokens.shape[0]
    gen_len = tokens.shape[1]

    sampled_next_tokens = torch.empty(batch_size, 0, model.config.vocab_size, device = model.device)

    for i in range(gen_len):

        one_token_score = tokens_scores[i]
        one_token_score.requires_grad_()

        next_tokens_soft = torch.nn.functional.softmax(one_token_score, dim=-1)

        next_tokens_onehot = ( # shape bs, vocab_size
            torch.nn.functional.one_hot(
                tokens[:,i], num_classes=model.config.vocab_size
            ).float().to(model.device)
        )

        if requires_grad:
            next_tokens_st_trick = ( # shape bs, vocab_size
                next_tokens_onehot
                - next_tokens_soft.detach()
                + next_tokens_soft
            )
        else:
            next_tokens_st_trick = ( # shape bs, vocab_size
                next_tokens_onehot
                - next_tokens_soft
                + next_tokens_soft
            ).detach()

        sampled_next_tokens = torch.cat( # final size will be  [bs, x_vision_seq_len, vocab_size]
            (sampled_next_tokens, next_tokens_st_trick.unsqueeze(1)), dim=1
        )

    embeds = torch.matmul(
        sampled_next_tokens.to(model.get_input_embeddings().weight.dtype),
        model.get_input_embeddings().weight
        )

    if not requires_grad:
        embeds.requires_grad = False

    return embeds



@torch.no_grad()
def forward_until_eos(model, padded_embeds, eos_token_id, tokenizer, max_new_tokens=600):
    """
    Iteratively generate tokens until all sequences reach eos_token_id.
    Returns:
      generated_ids: [bs, <=max_new_tokens]
      seq_lengths: tensor [bs] index of EOS token for each sequence
      past_key_values: KV cache at the end of generation
    """
    model.set_adapter("Y_gen")
    embedding_layer = model.get_input_embeddings()
    device = padded_embeds.device
    bs, seq_len, _ = padded_embeds.shape

    input_embeds = padded_embeds
    attention_mask = torch.ones((bs, seq_len), device=device, dtype=torch.long)

    generated_ids = torch.full((bs, 0), tokenizer.pad_token_id, device=device, dtype=torch.long)
    finished = torch.zeros(bs, dtype=torch.bool, device=device)
    seq_lengths = torch.full((bs,), -1, device=device, dtype=torch.long)

    past_key_values = None

    for step in range(max_new_tokens):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

        # mark EOS positions the first time they appear
        just_finished = (next_tokens == eos_token_id) & ~finished
        seq_lengths[just_finished] = seq_len + step
        finished |= just_finished

        if finished.all():
            break

        # embed next tokens for next iteration
        next_embeds = embedding_layer(next_tokens.unsqueeze(1))
        input_embeds = next_embeds  # only pass the new token next time
        attention_mask = torch.cat([attention_mask, torch.ones((bs, 1), device=device, dtype=torch.long)], dim=1)

    del attention_mask, past_key_values, finished
    torch.cuda.empty_cache()

    # # build full attention mask for X + gen Y (masking toks after eos_tok_ids)
    # max_len = max(seq_lengths).item() + 1
    # attention_mask = torch.zeros((bs, max_len), device=device, dtype=torch.long)
    # for i in range(bs):
    #     attention_mask[i, : seq_lengths[i] + 1] = 1

    return generated_ids, seq_lengths #, attention_mask #, past_key_values





def get_logits_at_step_T_for_K_gt_1(HF_name, model, tokenizer, padded_embeds, optim_slice, prefix_past_key_values=None, max_new_tokens=600):

    eos_token_id = get_eos_token(HF_name)
    model.set_adapter("Y_gen")
    embedding_layer = model.get_input_embeddings()
    device = padded_embeds.device
    bs = padded_embeds.size(0)

    # Step 1: run until given EOS (no grads), record positions of EOS
    generated_ids, seq_lengths = forward_until_eos(
        model, padded_embeds, eos_token_id, tokenizer, max_new_tokens
    )

    # Step 2: Forward pass for prefix part (no grad)
    start_idx = int(torch.where(torch.tensor(optim_slice) == 1)[0].min().item()) # first optimizable index

    if prefix_past_key_values is None:
        with torch.no_grad():
            prefix_out = model(
                inputs_embeds=padded_embeds[:, :start_idx, :].detach(),  # frozen prefix
                # attention_mask=torch.ones((bs,start_idx)), #attention_mask[:, :start_idx],
                use_cache=True,
            )
            prefix_past_key_values = prefix_out.past_key_values
    else:
        print("prefix_past_key_values:", prefix_past_key_values)

    # Step 3: Second forward for optimizable slice (with grad)
    optim_embeds = padded_embeds[:, start_idx:, :] # gradient flows here
    optim_embeds.requires_grad_(True)
    gen_embeds = embedding_layer(generated_ids)
    gen_embeds.requires_grad = False

    print("optim embeds shape:", optim_embeds.shape)
    print("gen_embeds shape:", gen_embeds.shape)

    # # take only the mask part that corresponds to these tokens
    # attn_mask_new = attention_mask[:, start_idx: ]
    # outputs = model(
    #     inputs_embeds=torch.cat((optim_embeds, gen_embeds), dim=1),
    #     attention_mask=attn_mask_new, # WARNING (slice needed here ?)
    #     past_key_values=past_key_values,
    #     use_cache=True,
    # )
    # next_token_logits = outputs.logits[:, -1, :]

    # take only the logits that corresponds to token after eos_tok_id
    outputs = model(
        inputs_embeds=torch.cat((optim_embeds, gen_embeds), dim=1),
        past_key_values=prefix_past_key_values,
        use_cache=True,
    )

    next_token_logits = outputs.logits[torch.arange(bs), (seq_lengths-start_idx)]

    del gen_embeds, outputs, seq_lengths #, attention_mask
    torch.cuda.empty_cache()


    return next_token_logits, generated_ids, prefix_past_key_values



def get_logits_at_step_T(HF_name, model, tokenizer, padded_embeds, total_config, check_requires_grad, use_st_trick=False):
    # if issues, see eval_llms_brouillon.ipynb
    if check_requires_grad:
        if total_config["onehot"]:
            padded_embeds.requires_grad == True
        else:
            assert padded_embeds.requires_grad == True
    model.set_adapter("Y_gen")
    embedding_layer = model.get_input_embeddings()

    max_new_tokens=600
    eos_token_id = get_eos_token(HF_name)

    # first forward without gradient to get token ids up until step T-1, until '"' (just before the class of the decision)
    '''Thought:
    - Notable features: [...]
    Final Answer:
    - Intention: "'''

    with torch.no_grad():
        proj_embed_input = {'inputs_embeds': padded_embeds.to(model.device),
                            'attention_mask': torch.ones((padded_embeds.shape[0], padded_embeds.shape[1])).to(model.device)}
        # input_length = padded_embeds.shape[1]
        model.temperature = None
        model.top_p=None
        model.top_k=None
        generated_ids = model.generate(
            **proj_embed_input,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id, #tokenizer.eos_token_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True
            )

        new_T_minus_1_tokens = generated_ids.sequences # here we dont need to add [:, input_length:] to focus on generated ids, because inputs are embeds and not input_ids
        if use_st_trick:
            new_T_minus_1_tokens_scores = generated_ids.logits # len gen_len of tensors of shape bs, vocab_size
            # print("new_T_minus_1_tokens.shape:", new_T_minus_1_tokens.shape) # bs, gen_len
            # print("len of scores:", len(new_T_minus_1_tokens_scores)) # gen_len
            # print("shape of one score:", new_T_minus_1_tokens_scores[-1].shape) # bs, vocab
            # print("requires_grad of one score:", new_T_minus_1_tokens_scores[-1].requires_grad) # False

            assert len(new_T_minus_1_tokens_scores) == new_T_minus_1_tokens.shape[1]
            T_minus_1_embeds = soft_hard_trick(new_T_minus_1_tokens, new_T_minus_1_tokens_scores, model, requires_grad=True)
        else:
            T_minus_1_embeds = embedding_layer(new_T_minus_1_tokens).detach()
            T_minus_1_embeds.requires_grad = False

    # last inference with gradient
    padded_embeds = torch.cat((padded_embeds, T_minus_1_embeds), dim=1)
    proj_embed_input = {'inputs_embeds': padded_embeds.to(model.device),
                        'attention_mask': torch.ones((padded_embeds.shape[0], padded_embeds.shape[1])).to(model.device)}

    output = model(
        **proj_embed_input
        ) # output.logits.shape = [current_bs, seq_len, voc_size]

    # llama3 class tokens
    # 15 : '0'
    # 16 : '1'
    # 17 : '2'

    # llama2 class tokens
    # 29900 : '0'
    # 29896 : '1'
    # 29906 : '2'

    # mistral 7B class tokens
    # 29502 : '0
    # 29508 : '1'
    # 29518 : '2

    with torch.no_grad():
        token_step_T = torch.argmax(output.logits[:, -1], dim=-1, keepdim=True)
        # print("T-1 tokens shape:", new_T_minus_1_tokens.shape)
        # print("T tokens shape:", token_step_T.shape)

        T_tokens_ids = torch.cat((new_T_minus_1_tokens, token_step_T), dim=-1)
        # print("after cat shape:", T_tokens_ids.shape) # hopefully shape [bs, T]
    return output.logits[:, -1], T_tokens_ids # premier element = torch.Size([current_bs, voc_size])



def biased_ar_gen_and_loss(
    algo,
    HF_name,
    model,
    tokenizer,
    loss_fct,
    add_ppl_loss,
    encoded_x_vision,
    bias_penalty,
    Xo_bias_penalty,
    x_sys_input_ids,
    x_vision_seq_len,
    X_T_G,
    total_config,
    verbose,
    return_rank=False
    ): # ori code "energy_fn_wrapper"

    # print("\nStep 1: biased_ar_gen_and_loss")
    bias_dim = model.get_input_embeddings().weight.shape[0]

    # this is to avoid biasing the AR gen of the x_sys part
    # x_sys_bias = torch.zeros(bias_penalty.size(0), x_sys_input_ids.shape[1], bias_dim).to(model.device)
    # bias_penalty_full = torch.concat([x_sys_bias, bias_penalty], dim=1)

    # 1) Gen modified X_vision

    # output_ids: shape [batch_size, full_seq=x_sys_len+x_vision_seq_len]
    # onehot_generates: values of onehot but backprop on soft values, shape [bs, x_vision_seq_len, vocab_size]
    # soft_generates: soft onehot (via softmax), shape [bs, x_vision_seq_len, vocab_size] => removed for now
    # logits: logits of generated tokens in x_vision # final shape [bs, x_vision_seq_len, vocab_size] => now i return topk_ids instead
    output_ids, onehot_generates, topk_ids = biased_ar_gen(
        model=model,
        tokenizer=tokenizer,
        input_ids=x_sys_input_ids,
        input_embeds=None,
        x_vision_seq_len=x_vision_seq_len,
        bias_penalty=bias_penalty,
        Xo_bias_penalty=Xo_bias_penalty,
        total_config=total_config,
        return_sampled_hard_and_soft_one_hot=True if algo == "DAB" else False
    )
    if verbose:
        deco_after_ar = tokenizer.batch_decode(output_ids[:, x_sys_input_ids.shape[1]:])[0]
        print("Seq after biased ar reg:\n", deco_after_ar) # , skip_special_tokens=True
        if total_config["save_txt"]:
            try:
                with open(total_config['deco_text_fp'], 'a') as file:
                    file.write("\n" + "Seq after biased ar reg:\n" + deco_after_ar + "\n")
            except:
                with open(total_config['deco_text_fp'], 'a') as file:
                    file.write("\n" + "Seq after biased ar reg:\n" + "[Failed to render generation]" + "\n")
                # pass
    bias_penalty.detach()
    del bias_penalty

    # 2) Knowing new X_vision, generate Y until step T to get loss on target token
    if return_rank:
        rank_X_T_G, loss, sim_loss, ppl_loss, T_tokens_ids, onehot_generates = gen_T_steps_and_get_loss(
            HF_name=HF_name,
            X_T_G=X_T_G,
            model=model,
            tokenizer=tokenizer,
            loss_fct=loss_fct,
            add_ppl_loss=add_ppl_loss,
            encoded_x_vision=encoded_x_vision,
            input_ids=x_sys_input_ids,
            output_ids=output_ids,
            onehot_generates=onehot_generates,
            total_config=total_config,
            return_rank=return_rank
        )
        return rank_X_T_G, loss, sim_loss, ppl_loss, output_ids, T_tokens_ids, onehot_generates, topk_ids

    else:
        loss, sim_loss, ppl_loss, T_tokens_ids, onehot_generates = gen_T_steps_and_get_loss(
            HF_name=HF_name,
            X_T_G=X_T_G,
            model=model,
            tokenizer=tokenizer,
            loss_fct=loss_fct,
            add_ppl_loss=add_ppl_loss,
            encoded_x_vision=encoded_x_vision,
            input_ids=x_sys_input_ids,
            output_ids=output_ids,
            onehot_generates=onehot_generates,
            total_config=total_config,
            return_rank=return_rank
        )

        return loss, sim_loss, ppl_loss, output_ids, T_tokens_ids, onehot_generates, topk_ids


def biased_ar_gen( # ori code soft_greedy_search_with_biases
        model,
        tokenizer,
        input_ids, # when calling function here, set to=x_sys.input_ids,
        input_embeds,  # None
        x_vision_seq_len,
        bias_penalty,
        Xo_bias_penalty,
        total_config, # total_config["learnable_indices"] is included
        return_sampled_hard_and_soft_one_hot, # added
        use_hidden_states_biases=False, # added
        trainable_weights=None,  # added
        reverse=False, # added
        return_soft_one_hot=False # added
    ):
    if bias_penalty is not None and Xo_bias_penalty is not None:
        assert bias_penalty.shape == Xo_bias_penalty.shape
    disc_weight=total_config["weight_val"]

    # print("   biased AR gen")
    model.set_adapter("X_vision_gen")
    embedding_layer = model.get_input_embeddings()

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1] # added, tbc
    seq_len = prompt_len + x_vision_seq_len
    init_len = prompt_len # or 0

    if return_sampled_hard_and_soft_one_hot:
        sampled_next_tokens = torch.empty(batch_size, 0, model.config.vocab_size, device = model.device) # final size will be [bs, x_vision_seq_len, vocab_size] by appending on dim 1
        logits_seq = torch.empty(batch_size, x_vision_seq_len, model.config.vocab_size, device = model.device).detach()  # final shape [bs, x_vision_seq_len, vocab_size]
        if return_soft_one_hot:
            soft_tokens = torch.empty(batch_size, x_vision_seq_len, model.config.vocab_size, device = model.device)

    past_key_values = None
    learnable_idx=0
    for it, cur_len in enumerate(range(prompt_len, seq_len)): # or for cur_len in range(x_vision_seq_len): # "it" in in the range of x_vision indices

        bias_idx = it if not reverse else seq_len - prompt_len - it
        if total_config["fix_all_non_learnable_indices"]:
            logit_mask=1 if it in total_config["learnable_indices"] else 0
        else:
            logit_mask=1 if it >= min(total_config["learnable_indices"]) else 0

        # if logit_mask == 1:

        if cur_len == prompt_len:
            model_inputs = {"input_ids": input_ids}
        else:
            model_inputs = {"input_ids": next_tokens[:, None]}

        if input_embeds is None:
            model_inputs["inputs_embeds"] = embedding_layer(model_inputs["input_ids"])
        else:
            if len(input_embeds.size()) == 2:
                model_inputs["inputs_embeds"] = input_embeds.unsqueeze(1)
            else:
                model_inputs["inputs_embeds"] = input_embeds
        model_inputs["input_ids"] = None

        # forward pass
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                past_key_values=past_key_values,
                use_cache=True, # to get past_key_values
                return_dict=True
                )
            past_key_values = outputs.past_key_values  # The KV cache
            next_tokens_scores = outputs.logits[:, -1, :].detach()  # shape bs, vocab_size

        if logit_mask == 1 or total_config["ar_gen_equation"]!="ours":

            if total_config["use_scale_weights"] and bias_penalty.mean().item() != 0:
                logit_norms = next_tokens_scores.detach().norm(dim=-1, p=2) # ori: outputs.logits[:, -1, :].detach().norm(dim=-1, p=2)
                bias_norms = bias_penalty[:, bias_idx, :].detach().norm(dim=-1, p=2)
                scaling_ratio = (logit_norms / (bias_norms + 1e-12)).unsqueeze(-1)
                if total_config['add_Xo_reg_in_bias']:
                    xo_bias_norms = Xo_bias_penalty[:, bias_idx, :].detach().norm(dim=-1, p=2)
                    xo_scaling_ratio = (logit_norms / (xo_bias_norms + 1e-12)).unsqueeze(-1)
            else:
                scaling_ratio, xo_scaling_ratio = 1.0, 1.0

            if total_config["use_fadding_weights"]:
                weight = 1 * (cur_len - init_len) / (seq_len - init_len)
            elif total_config["use_bolt_weights"]:
                weight = trainable_weights[cur_len]
            else:
                weight = 1

        if not use_hidden_states_biases:
            if total_config["ar_gen_equation"]=="dlp": #is_dlp:
                # bias_penalty is negative (bears the "minus sign")
                # ori code
                next_tokens_scores = (
                    next_tokens_scores + disc_weight * weight * scaling_ratio * bias_penalty[:, bias_idx, :]
                )
            elif total_config["ar_gen_equation"]=="ours":
                if logit_mask == 1:
                    next_tokens_scores = (
                        total_config["logit_weight_val"] * logit_mask * next_tokens_scores
                        + disc_weight * weight * scaling_ratio * bias_penalty[:, bias_idx, :]
                    )
                    if total_config["add_Xo_reg_in_bias"]:
                        next_tokens_scores += total_config["Xo_bias_weight_val"] * weight * xo_scaling_ratio * Xo_bias_penalty[:, bias_idx, :]
                elif logit_mask == 0:
                    # max_penalty = torch.max(torch.abs(bias_penalty[:, bias_idx, :]))
                    next_tokens_scores = (
                        # (max_penalty + bias_penalty[:, bias_idx, :])
                        bias_penalty[:, bias_idx, :]
                    )
            elif total_config["ar_gen_equation"]=="dlp_wo_scale":
                next_tokens_scores = next_tokens_scores + weight * bias_penalty[:, bias_idx, :] # ori: biases[bias_idx]
            else:
                raise Exception
        else:
            next_tokens_scores = next_tokens_scores + weight * model.lm_head(
                bias_penalty[bias_idx]
            )

        ## almost same as "soft_forward_without_decoding" (ori code) belows
        if total_config["use_Xo_topk_S2"]:
            if logit_mask == 1:
                topk_s2=total_config["Xo_topk_for_S2"][0,learnable_idx,:].to(next_tokens_scores.device) # size k_val_S2
                next_tokens_among_topk_s2 = torch.argmax(next_tokens_scores.detach()[:,topk_s2], dim=-1) # shape bs
                next_tokens = topk_s2[next_tokens_among_topk_s2]
                learnable_idx +=1
                # print("constrained step 2 decoding:", tokenizer.decode(next_tokens))
            else:
                next_tokens = torch.argmax(next_tokens_scores.detach(), dim=-1) # shape bs
        else: # classic deco
            next_tokens = torch.argmax(next_tokens_scores.detach(), dim=-1) # shape bs

        # NB: if i reactivate gradient flow here instead of right after LLM forward:
        # I will loose the fact that the gradient also reflects how the bias penalty/scaling_ratio impacts the logits
        # i deactivated gradient on bias_penalty so seems right

        if return_sampled_hard_and_soft_one_hot: # added
            next_tokens_scores.requires_grad_() # # shape bs, vocab_size
            inference = True
            if not inference:
                cur_sampled_next_token = torch.nn.functional.gumbel_softmax(
                    next_tokens_scores, tau=1, hard=True, dim=-1
                )
            else:
                if not total_config["onehot"] or return_soft_one_hot:
                    cur_sampled_next_token_soft = torch.nn.functional.softmax( # shape bs, vocab_size
                        next_tokens_scores, dim=-1
                    )
                cur_sampled_next_token_onehot = ( # shape bs, vocab_size
                    torch.nn.functional.one_hot(
                        next_tokens, num_classes=model.config.vocab_size
                    ).float().to(input_ids.device)
                )
                if total_config["onehot"]:
                    cur_sampled_next_token = ( # shape bs, vocab_size
                        cur_sampled_next_token_onehot
                    )
                else:
                    cur_sampled_next_token = ( # shape bs, vocab_size
                        cur_sampled_next_token_onehot
                        - cur_sampled_next_token_soft.detach()
                        + cur_sampled_next_token_soft
                    )
                if return_soft_one_hot:
                    soft_tokens = torch.cat( # final size will be  [bs, x_vision_seq_len, vocab_size]
                        (soft_tokens, cur_sampled_next_token_soft.unsqueeze(1)), dim=1
                    )
                # elif it not in total_config["learnable_indices"]:
                #     cur_sampled_next_token = ( # shape bs, vocab_size
                #         torch.nn.functional.one_hot(
                #             next_tokens, num_classes=model.config.vocab_size
                #         ).float().to(input_ids.device)
                #     ).detach()

            sampled_next_tokens = torch.cat( # final size will be  [bs, x_vision_seq_len, vocab_size]
                (sampled_next_tokens, cur_sampled_next_token.unsqueeze(1)), dim=1
            )
            # sampled_next_tokens[:, it, :] = cur_sampled_next_token #.detach() => breaks computational graph ?

            with torch.no_grad():
                # logits_seq = torch.cat((logits_seq, next_tokens_scores.unsqueeze(1)), dim=1) # final shape [bs, x_vision_seq_len, vocab_size]
                logits_seq[:, it, :] = next_tokens_scores.detach()

            # print("before matmul")
            # print("model.lm_head.weight.dtype:", model.lm_head.weight.dtype) # model.lm_head.weight.dtype: torch.bfloat16
            # print("model.get_input_embeddings().weight.dtype:", model.get_input_embeddings().weight.dtype) # model.get_input_embeddings().weight.dtype: torch.bfloat16
            # print("cur_sampled_next_token.dtype:", cur_sampled_next_token.dtype) # cur_sampled_next_token.dtype: torch.float32
            # with torch.cuda.amp.autocast():
            input_embeds = torch.matmul(
                cur_sampled_next_token.to(model.get_input_embeddings().weight.dtype),
                model.get_input_embeddings().weight
                )

        else:
            input_embeds = None

        # Append token to sequence
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1) # next_tokens[:, None] == next_tokens.unsqueeze(-1)
        del outputs, next_tokens_scores
        torch.cuda.empty_cache()

    # outputs for DAB
    if return_sampled_hard_and_soft_one_hot:
        # input_ids of shape [natch_size, full_seq=x_sys_len+x_vision_seq_len]
        # sampled_next_tokens: values of onehot but backprop on soft values, shape [bs, x_vision_seq_len, vocab_size]
        # soft_tokens: soft onehot (via softmax), shape [bs, x_vision_seq_len, vocab_size]
        # logits_seq: logits of generated tokens in x_vision # final shape [bs, x_vision_seq_len, vocab_size]
        with torch.no_grad():
            logits_seq.detach()
            if total_config["learnable_indices"] is not None:
                topk_ids = torch.topk(logits_seq, total_config['k_val'], dim=-1).indices[:, total_config["learnable_indices"], :]  # shape [bs, len(learnable_indices), k_val]
            else:
                topk_ids = torch.topk(logits_seq, total_config['k_val'], dim=-1).indices  # shape [bs, x_vision_seq_len, k_val]
            topk_ids.requires_grad = False

        del past_key_values, logits_seq
        torch.cuda.empty_cache()

        if return_soft_one_hot:
            return input_ids, sampled_next_tokens, soft_tokens, topk_ids # logits_seq # input_ids include the prompt + all generated token ids
        else:
            return input_ids, sampled_next_tokens, topk_ids #logits_seq
    # outputs for PEZ
    else:
        sampled_next_tokens, topk_ids = None, None
        return input_ids, sampled_next_tokens, topk_ids

# https://github.com/patrickpynadath1/dab/blob/main/models/model_with_biases.py#L169
# me: this both does the biased AR gen + the constraint loss eval
# it gets logits of the generation to be able to extract the top K to limit the discrete sampling of B
def gen_T_steps_and_get_loss( # ori code: my soft forward (includes biased_ar_gen + this code section)
        HF_name,
        X_T_G,
        model,
        tokenizer,
        loss_fct,
        add_ppl_loss,
        encoded_x_vision,
        input_ids,
        output_ids,
        onehot_generates,
        total_config,
        return_rank=False,
        sim_loss_only=False,
    ):

    prompt_len = input_ids.shape[1]
    # WARNING onehot_generates only contains one hot for gen tokens while ouput_ids contain all tok ids (input prompt and gen ones)
    # print("   gen T answer tokens + loss on step T (my_soft_forward)")

    # Get embeddings from one hot
    model.set_adapter("X_vision_gen")
    input_prompt_onehot = ( # shape [bs, input_prompt_len, vocab_size] ? while onehot_generates is [bs, gen_seq_len, vocab_size]
        torch.nn.functional.one_hot(
            input_ids, num_classes=model.config.vocab_size
        ).float().to(model.device)
    ).detach()
    input_prompt_onehot.requires_grad=False

    # for DAB
    if onehot_generates is not None:
        # print("onehot_generates.requires_grad:", onehot_generates.requires_grad)
        if total_config["onehot"]:
            onehot_generates.requires_grad=True
        full_onehot_generates = torch.cat((input_prompt_onehot, onehot_generates), dim=1)
        lm_embs = torch.matmul(
            full_onehot_generates.to(model.get_input_embeddings().weight.dtype),
            model.get_input_embeddings().weight
            )
    # for PEZ+DAB
    else:
        text_embedding = model.get_input_embeddings()
        lm_embs = text_embedding(output_ids).detach() # output_ids contains x_sys + x_vision
        lm_embs.requires_grad = False

    # Optionally compute ppl loss
    if add_ppl_loss:
        ppl_loss = model(inputs_embeds=lm_embs, labels=output_ids).loss
    else:
        with torch.no_grad():
            ppl_loss = model(inputs_embeds=lm_embs, labels=output_ids).loss.detach()

    if total_config["add_Xo_reg_in_loss"]: # and encoded_x_vision is not None: # ie, add_Xo_reg_in_loss is True
        with torch.no_grad():
            embedding_layer = model.get_input_embeddings()
            X_vision_embeds = embedding_layer(encoded_x_vision.input_ids).detach()
            X_vision_embeds.requires_grad = False
        assert lm_embs[:, prompt_len:, :].shape == X_vision_embeds.shape
        sim_loss = 1 - (F.cosine_similarity(lm_embs[:, prompt_len:, :], X_vision_embeds, dim=-1).mean())
    else:
        sim_loss = 0

    # Loss using Y_gen model forward
    model.set_adapter("Y_gen")
    logits_step_T, T_tokens_ids = get_logits_at_step_T(
        HF_name=HF_name, model=model, tokenizer=tokenizer, padded_embeds=lm_embs, total_config=total_config,
        check_requires_grad = True if onehot_generates is not None else False
        ) # torch.Size([current_bs, voc_size])
    token_target_loss = loss_fct(logits_step_T, torch.tensor([X_T_G]).to(model.device)) # size 1 ?

    if sim_loss_only: # only for test purposes
        loss = sim_loss
    else:
        if add_ppl_loss:
            loss = 1 * token_target_loss + 0.1 * ppl_loss
        else:
            loss = token_target_loss

        loss = loss + total_config["xo_loss_weight"] * sim_loss

    if return_rank:
        rank_X_T_G = get_rank(curr_logits=logits_step_T, X_T_G=X_T_G, device=model.device)
        return rank_X_T_G, loss, sim_loss, ppl_loss, T_tokens_ids, onehot_generates
    return loss, sim_loss, ppl_loss, T_tokens_ids, onehot_generates # TBC: logits is for X_vision while logits_step_T if only for Y_T

def calc_grad(loss, onehot, prompt_len): # total_config

    # gx = torch.autograd.grad(loss, onehot, allow_unused=True)[0].detach() #[:, prompt_len :, :]
    # gx = torch.autograd.grad(loss, onehot, allow_unused=True)[0].detach()[:, total_config["learnable_indices"], :]
    gx = torch.autograd.grad(loss, onehot, allow_unused=True)[0].detach() # (bs, x_vision_seq_len, vocab_size)

    return gx

# computes the distribution over all the tokens in the model vocabulary
def get_unfiltered_dist(gx, cur_token_ids, prompt_len, global_version, total_config, EPS=1e-10): #, cur_bias=None):
    # print(gx.shape)
    token_dist = torch.ones_like(gx).to(total_config["device"]) # (bs, x_vision_seq_len, vocab_size)
    if not total_config["dlp_unmask"]:
        token_dist[
            torch.arange(token_dist.size(0))[:, None, None],
            torch.arange(token_dist.size(1))[None, :, None],
            cur_token_ids[:, prompt_len :].unsqueeze(-1),
        ] = EPS

    if total_config["learnable_indices"] is not None:
        unfiltered_dist = gx[:, total_config["learnable_indices"], :] * token_dist[:, total_config["learnable_indices"], :] # (bs, len(learnable_indices), vocab_size)
    else:
        unfiltered_dist = gx * token_dist # (bs, x_vision_seq_len, vocab_size)

    if global_version:
        second_proposal_term = None
    else:
        if total_config["learnable_indices"] is not None:
            second_proposal_term = torch.pow(token_dist[:, total_config["learnable_indices"], :], 2)
        else:
            second_proposal_term = torch.pow(token_dist, 2)

    return -1 * unfiltered_dist, second_proposal_term

# selects the logits from unfiltered_dist for the tokens in the topk_ids
def apply_filter(unfiltered_dist, topk_ids):
    filtered_dist_logits = unfiltered_dist[
        torch.arange(unfiltered_dist.size(0))[:, None, None],
        torch.arange(unfiltered_dist.size(1))[None, :, None],
        topk_ids,
    ]
    return filtered_dist_logits

# given the sampled topk indices, converts them to tokens from the vocab
def topk_to_tokens(topk_ids, sampled_indices):
    actual_ids = topk_ids[
        torch.arange(topk_ids.size(0))[:, None],
        torch.arange(topk_ids.size(1))[None, :],
        sampled_indices,
    ]
    return actual_ids

# wrapper class for getting the dlp logits over the top k tokens
# takes care of filtering the logits
# onehot: values of onehot but backprop on soft values, shape [bs, x_vision_seq_len, vocab_size]
# logits: logits of generated tokens in x_vision # final shape [bs, x_vision_seq_len, vocab_size]
# cur_token_ids: shape [batch_size, full_seq=x_sys_len+x_vision_seq_len]

def get_dlp_dist(loss, onehot, cur_token_ids, prompt_len, total_config, global_version = True, alpha= 0.0001): # logits
    gx = calc_grad(loss, onehot, prompt_len)
    # print("sum all entries of onehot:", torch.sum(onehot).item())
    # print("nb of non nul entries in onehot:", torch.sum(onehot!=0))
    # print("sum all entries of gx:", torch.sum(gx).item())
    # print("nb of non nul entries in gx:", torch.sum(gx!=0))
    # logits = logits[:, prompt_len :, :]
    unfiltered_dist, token_dist_sq = get_unfiltered_dist(gx, cur_token_ids, prompt_len, global_version, total_config)
    # topk_ids = torch.topk(logits, total_config['k_val'], dim=-1).indices
    gx.detach()
    del gx
    if global_version:
        dist = unfiltered_dist / total_config["proposal_temp"]
    else:
        dist = (unfiltered_dist/2) - (token_dist_sq/(2*alpha)) # from icml 22 (a langevin like sampler for discrete distributions)
    return dist

def bias_sampling_from_gradient(model, tokenizer, loss, output_ids, onehot, topk_ids, prompt_len, total_config, encoded_x_vision, verbose): #, senti_losses): # ori code step_soft
    # print("\nStart bias_sampling_from_gradient")
    sampled_ids = sample_bias_seq(
        loss, output_ids, onehot, topk_ids, prompt_len, total_config #, senti_losses
    )
    if total_config["fix_template"]:
        new_sampled_ids = encoded_x_vision.input_ids.detach().clone()
        new_sampled_ids[:, total_config["learnable_indices"]] = sampled_ids
    else:
        new_sampled_ids = sampled_ids
    if verbose:
        deco_after_dlp = tokenizer.batch_decode(new_sampled_ids)[0]
        print("Sampled sent via DLP:\n", deco_after_dlp) # , skip_special_tokens=True
        if total_config["save_txt"]:
            try:
                with open(total_config['deco_text_fp'], 'a') as file:
                    file.write("\n" + "Sampled sent via DLP:\n" + deco_after_dlp + "\n")
            except:
                with open(total_config['deco_text_fp'], 'a') as file:
                    file.write("\n" + "Sampled sent via DLP:\n" + "[Failed to render generation]" + "\n")
                # pass
    bias = compute_bias_l2_pen(model, new_sampled_ids, total_config)
    return bias #, loss #, output_ids #, [senti_losses]


def bias_pen_from_bias_seq(sampled_ids, model, tokenizer, total_config, encoded_x_vision, verbose): #, senti_losses): # ori code step_soft
    # same as 'bias_sampling_from_gradient' but we already know the bias seq (sampled_ids)
    if total_config["fix_template"]:
        new_sampled_ids = encoded_x_vision.input_ids.detach().clone()
        new_sampled_ids[:, total_config["learnable_indices"]] = sampled_ids
    else:
        new_sampled_ids = sampled_ids
    # if verbose:
    deco_before_ar = f"\n\n[X_vision before AR reg, found by optim]:\n{tokenizer.batch_decode(new_sampled_ids)[0]}"
    bias = compute_bias_l2_pen(model, new_sampled_ids, total_config)
    return bias, deco_before_ar


# performs the actual sampling of the bias tokens for soft constraints
def sample_bias_seq(loss, output_ids, onehot, topk_ids, prompt_len, total_config): #, senti_losses): ori code: "compute_p_lm_soft"
    # print("   get dlp dist")
    unfiltered_dist = get_dlp_dist(loss, onehot, output_ids, prompt_len, total_config)
    onehot.detach()
    del onehot
    # print("   filter with top K")
    dist_logits = apply_filter(unfiltered_dist, topk_ids)
    # print("   Sample bias seq")
    proposal_dist = torch.distributions.Categorical(logits=dist_logits) # / total_config["proposal_temp"]) => added to get_dlp_dist
    torch.manual_seed(total_config["seed_dlp"])
    sampled_indices = proposal_dist.sample()
    sampled_tokens = topk_to_tokens(topk_ids, sampled_indices)
    unfiltered_dist.detach()
    del unfiltered_dist
    dist_logits.detach()
    del dist_logits
    return sampled_tokens # loss, output_ids, senti_losses.detach().cpu().numpy()



def compute_bias_l2_pen(model, sampled_ids, total_config): #, kw_token=None):
    # print("   compute l2 penalty based on sampled bias")
    model.set_adapter("X_vision_gen")
    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        # this is batch x seq_len x embed_dim
        cur_embeds =  embedding_layer(sampled_ids)  # ori code: sampler's self.embed_map(sampled_ids) = model.get_input_embeddings()

        # compute ||embed - sampled_embed||^2 using foil
        t1 = torch.einsum("ve -> v", [embedding_layer.weight**2])[None, None, :]
        t2 = torch.einsum("bse, ve -> bsv", [cur_embeds, embedding_layer.weight])
        t3 = torch.einsum("bse -> bs", [cur_embeds**2]).unsqueeze(-1)
        bias = -1 * total_config["weight_val"] * (t1 - 2 * t2 + t3)

    bias.requires_grad = False # added
    return bias


# sampler initialize batch method in https://github.com/patrickpynadath1/dab/blob/2eeeca6f0b91229904225ccb39c443e7d3553c43/samplers/dlp_embed.py#L193
def my_initialize_bias(  # ori code: "initialize_batch"
        model,
        total_config,
        batch_size,
        ori_x_vision_ids,
        x_vision_seq_len=None,
        seq_len=None,
        prompt_len=None
    ):

    initialization = total_config["initialization"]

    if x_vision_seq_len is None:
        x_vision_seq_len = seq_len - prompt_len

    embedding_layer = model.get_input_embeddings()
    logit_dim = model.get_input_embeddings().weight.size(0) # vocab_size
    embed_dim = model.get_input_embeddings().weight.size(1)
    last_dim = logit_dim

    if initialization == "zero":
        initial_bias = torch.zeros(
            batch_size, x_vision_seq_len, last_dim
        ).to(model.device)

    elif initialization == "from_ori_X":
        initial_bias = compute_bias_l2_pen(model=model, sampled_ids=ori_x_vision_ids, total_config=total_config) # [shape 1, x_vision_seq_len, vocab_size]
        initial_bias = initial_bias.expand(batch_size, -1, -1) # unlike repeat, this function does not copy the tensor’s data
        print("initial_bias shape:", initial_bias.shape)
        print(f"(batch_size={batch_size})")
    # elif initialization == "random_disc":
    #     sampled_ints = torch.randint(
    #         0, logit_dim, (batch_size, x_vision_seq_len)
    #     ).to(model.device)
    #     if last_dim == embed_dim:
    #         initial_bias = embedding_layer(sampled_ints)
    #     else:
    #         initial_bias = my_compute_bias_l2_pen(sampled_ints)
    # elif initialization == "random_cont":
    #     init_noise_rate = total_config["initialization_noise_rate"]
    #     initial_bias = init_noise_rate * torch.randn(
    #         batch_size, x_vision_seq_len, last_dim
    #     ).to(model.device)

    # weights = weight_val
    initial_bias = initial_bias.detach()
    initial_bias.requires_grad = False # ori code: True

    return initial_bias # shape [batch_size, x_vision_seq_len, vocab_size]


# ori function initialize_best_loss from https://github.com/patrickpynadath1/dab/blob/2eeeca6f0b91229904225ccb39c443e7d3553c43/utils/storing_metrics.py
def my_initialize_best_loss(batch_size):

    best_result = {}
    best_result["best_loss"] = [100000] * batch_size
    best_result["best_x"] = [""] * batch_size
    best_result["best_y"] = [""] * batch_size
    best_result["best_inferred_classes"] = [None] * batch_size
    best_result["best_constraint_satisfactions"] = [None] * batch_size
    best_result["best_sim_score"] = [0] * batch_size
    best_result["best_ppl_loss"] = [None] * batch_size
    best_result["best_template_fitness"] = [0]* batch_size
    best_result["best_steps"] = [None] * batch_size
    return best_result



# ori function in https://github.com/patrickpynadath1/dab/blob/2eeeca6f0b91229904225ccb39c443e7d3553c43/utils/storing_metrics.py
def my_updating_best_loss(args, batch_size, cur_step, loss, ppl_loss, X_string, Y_string, inferred_classes, constraint_satisfactions, F1_list, best_result, Xo):
    # print("updating best result")
    update_list = [False] * batch_size
    for i in range(batch_size):
        loss_i = loss.item() if batch_size == 1 else loss[i]

        if constraint_satisfactions[i]:
            if not best_result["best_constraint_satisfactions"][i]:
                update_list[i] = True
            else:
                if F1_list[i] > best_result["best_sim_score"][i]:
                    update_list[i] = True

        elif not constraint_satisfactions[i] and not best_result["best_constraint_satisfactions"][i]:
            if loss_i < best_result["best_loss"][i]:
                update_list[i] = True

    for i in range(len(update_list)):
        if update_list[i]:
            best_result["best_loss"][i] = loss_i
            best_result["best_x"][i] = X_string[i]
            best_result["best_y"][i] = Y_string[i]
            best_result["best_inferred_classes"][i] = inferred_classes[i]
            best_result["best_constraint_satisfactions"][i] = constraint_satisfactions[i]
            best_result["best_sim_score"][i] = F1_list[i]
            best_result["best_ppl_loss"][i] = ppl_loss.item() if batch_size == 1 else ppl_loss.detach().cpu().tolist()[i]
            best_result["best_template_fitness"][i] = check_template_fitness(args, X_string[i], hard_control=True)
            best_result["best_steps"][i] = cur_step
    # print("new best result:", best_result)
    return best_result


