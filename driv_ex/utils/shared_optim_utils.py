# Copyright (c) 2026 Valeo. All rights reserved.
"""Gradient-based counterfactual optimization primitives shared by DRIV-EX and GCG:
embedding initialization, nearest-neighbor projection onto the vocabulary, token
reinitialization strategies, rank/probability tracking, and gradient-slice extraction."""
import json
import os, sys
import random
import re
import time
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F

# sys.path.append(os.path.join(sys.path[0], "0_older_tests_and_data", "external_code"))
# from optim_utils import *
from driv_ex.utils.LLM_utils import gen_answer
from driv_ex.utils.driving_metrics import *
from driv_ex.dataset import DATA_DIR

from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)


def finish_gen_and_eval_best_seq(args, model, tokenizer, X_sys, X_vision, Y, best_x, best_y, learnable_indices=None):
    # input learnable_indices are with respect to X_vision and not X_sys !

    # gen the rest of Y
    model.set_adapter("Y_gen")
    best_y_end, full_scenario = gen_answer(
        HF_name=args.HF_name,
        model=model,
        tokenizer=tokenizer,
        prompt = X_sys + best_x + best_y,
        FT_type=None,
        gen_until_end=True,
        )

    # final metrics on best_Y
    best_y_full = best_y + best_y_end[0]

    try:
        coherence = is_lc_label_and_traj_coherent(best_y_full)
    except:
        coherence = "fail_to_extract"
    try:
        man_eval_multi_sec = evaluate_maneuver_multi_sec(full_scenario[0])
    except:
        man_eval_multi_sec = "fail_to_extract"

    try:
        GT_list = extract_trajectory(Y)
        pred_list = extract_trajectory(best_y_full)
        rmse_lon, rmse_lat = compute_trajectory_rmse(GT_list, pred_list)
    except:
        rmse_lon, rmse_lat = "fail_to_extract", "fail_to_extract"

    # final metrics on best_X
    # model.set_adapter("X_vision_gen") => done inside min_conditional_tok_prob
    x_sys_tokenized = tokenizer.encode(X_sys, add_special_tokens = False)
    if (x_sys_tokenized[0] != tokenizer.bos_token_id):
        print("WARNING: adding bos_token to X_sys for mctp evaluation")
        if tokenizer.bos_token_id is not None:
            X_sys = tokenizer.decode(tokenizer.bos_token_id) + X_sys
            x_sys_tokenized = tokenizer.encode(X_sys, add_special_tokens = False)
        # else:
        #     raise Exception
    # modify learnable_indices to be wrt X_sys+X_vision and not X_vision only
    # learnable_indices become learnable targets for mctp
    if learnable_indices is not None:
        learnable_indices = [idx+len(x_sys_tokenized)-1 for idx in learnable_indices] # -1 to fit targets in mctp

    try:
        mctp = min_conditional_tok_prob(
            model=model,
            tokenizer=tokenizer,
            best_seq=X_sys + best_x,
            learnable_targets=learnable_indices, # with respect to X_sys + X_vision with -1 shift to fit targets in mctp
            set_adapter=True,
            return_all_token_probs=True
            )
    except:
        mctp = "fail_to_extract"

    # template_fitness = check_template_fitness(args, best_x, hard_control=True)
    temp_fitness_hard = check_template_fitness(args=args, input_str=best_x, hard_control=True, return_details=False)
    temp_fitness_soft, valid_list, invalid_list = check_template_fitness(args=args, input_str=best_x, hard_control=False, return_details=True)
    temp_fitness_dico = {
            "hard_eval": temp_fitness_hard,
            "soft_eval": temp_fitness_soft
        }
    if len(invalid_list)>0:
        temp_fitness_dico["invalid_elements"]=invalid_list

    result = {
        "best_y_full":best_y_full,
        "lc_traj_coherence":coherence,
        "man_eval_multi_sec":man_eval_multi_sec,
        "rmse_lon":rmse_lon,
        "rmse_lat":rmse_lat,
        # "template_fitness":template_fitness,
        "template_fitness": temp_fitness_dico,
        "min_cond_tok_prob":mctp
    }
    return result


############

def get_ego_only_seq_len(args, tokenizer, X_vision):

    # if X_vision contains ego + sv, get the max token index of ego
    if args.optim_part == "ego":
        X_ego_end_template = "The information about its surrounding vehicles (within a range of 200 m) is listed as follows:\n"
        X_ego_end_id = X_vision.find(X_ego_end_template)
        X_ego_only = X_vision[:X_ego_end_id]
        x_ego_only_tokenization = tokenizer(X_ego_only,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=False,
                                       add_special_tokens=False)['input_ids']
        x_ego_only_seq_len = x_ego_only_tokenization.shape[1]
        print("x_ego_only_tokenization shape:", x_ego_only_tokenization.shape)
        # print("x_ego_only_seq_len:", x_ego_only_seq_len)
    else:
        x_ego_only_seq_len=None
    return x_ego_only_seq_len




def init_all_embeds_vm(args, model, tokenizer, X_sys, X_vision):

    if args.dataset == "lc_llm_highD":
        model.set_adapter("X_vision_gen")
    text_embedding = model.get_input_embeddings()

    with torch.no_grad():

        # # args.modif_pos is a list for example = [1, 3] listing indices between 0 and prompt_len excluded
        # modif_pos = list(range(1,T))

        X_sys_tokenization = tokenizer(X_sys,
                            padding=True,
                            truncation=False,
                            return_tensors="pt",
                            add_special_tokens=False)['input_ids']

        X_vision_tokenization = tokenizer(X_vision,
                            padding=True,
                            truncation=False,
                            return_tensors="pt",
                            add_special_tokens=False)['input_ids']

        init_tok_ids = torch.cat((X_sys_tokenization, X_vision_tokenization), dim=1)

        modif_pos = args.modif_pos
        if modif_pos == [0]:
            learnable_indices = get_fix_token_indices(args, X_vision, tokenizer)

            x_sys_tokenized_len = X_sys_tokenization.shape[1]
            modif_pos = [idx + x_sys_tokenized_len for idx in learnable_indices]
            print("X_sys_tokenization shape:", X_sys_tokenization.shape)
            print("X_vision_tokenization shape:", X_vision_tokenization.shape)
            print(f"\nwe want to optimize {len(modif_pos)} token(s):")
            for test_pos in modif_pos:
                print(f"Tok pos in seq={test_pos}, Tok val={init_tok_ids[0, test_pos]}: '{tokenizer.decode(token_ids=init_tok_ids[0, test_pos])}'")

        init_tok_ids = init_tok_ids.to(model.device)
        init_embeds_no_grad = text_embedding(init_tok_ids).detach()
        init_embeds_no_grad.requires_grad = False
        bs, seq_len, emb_dim = init_embeds_no_grad.shape
        print(f"\nInit of embeds with bs={bs}, xsys+xvision_len={seq_len}, emb_dim={emb_dim}")

        # init embeddings corresponding to positions of tokens that can be modified
        optim_slice = torch.tensor([1 if i in modif_pos else 0 for i in range(seq_len)])
        init_optim_ids = init_tok_ids[:, optim_slice == 1].detach().clone().to(model.device) # should be of shape bs, nb_optim_toks

        if args.init in ["input_toks", "semi_rand_voc"]:
            # init optim embeddings with proper input values in init_embeds_no_grad
            optim_embeds = init_embeds_no_grad[:, optim_slice == 1, :].detach().clone().to(model.device)
        elif args.init == "rand_voc":
            # randomly init optim embeddings
            torch.manual_seed(args.seed)
            optim_ids = torch.randint(len(tokenizer), (bs, len(modif_pos))).to(model.device)
            optim_embeds = text_embedding(optim_ids).detach()
        # elif args.init == "rand_full":
        #     # can do it for testing purposes only
        else:
            raise Exception(f"token init method {args.init} not implemented")

        optim_embeds.requires_grad = True
        print(f"optim_embeds of dim {optim_embeds.shape}")
        _, nb_optim_toks, _ = optim_embeds.shape

        # init_tok_ids of dim [bs=1, seq_len]
        # init_embeds_no_grad of dim [bs=1, seq_len, emb_dim]
        # optim_slice of len nb_optim_toks
        # optim_embeds of dim [bs=1, nb_optim_toks, emb_dim]
        # init_optim_ids of dim [bs=1, nb_optim_toks]

    return init_tok_ids, init_embeds_no_grad, init_optim_ids, optim_embeds, optim_slice, modif_pos, seq_len, nb_optim_toks







######## from before LCLLM

def get_greedy_id(logits):
    return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)




def get_proba_of_greedy_decoding(logits, reduce=None, in_percentage=True): # X_T_G
    # proba = torch.softmax(logits, dim=-1)[:, X_T_G]
    proba = torch.max(torch.softmax(logits, dim=-1), dim=-1).values
    if in_percentage:
        proba = proba*100
    if reduce is None:
        return proba
    else:
        return torch.prod(proba.squeeze(0))




def insert_optim_embeds_into_full_encoding(tmp_embeds, init_embeds_no_grad, optim_slice, current_bs=1):
    if current_bs == 1:
        padded_embeds = init_embeds_no_grad.detach().clone() # shape [1, seq_len, emb_dim]
    else:
        padded_embeds = init_embeds_no_grad.expand(current_bs,-1,-1).detach().clone() # shape [current_b, seq_len, emb_dim]
    padded_embeds.requires_grad = False
    padded_embeds[:, optim_slice == 1, :] = tmp_embeds
    return padded_embeds


def get_rank(curr_logits, X_T_G=None, X_T_G_list=None, device=None):

    # Getting rank of T_K_G from logits (softmax + rank extraction) when ranking by decreasing proba
    with torch.no_grad():

        gen_prob_distrib = torch.softmax(curr_logits[0].unsqueeze(0), dim=-1)  # curr_logits shape: [bs/K, voc_size]

        argsorted_proba = torch.argsort(gen_prob_distrib, dim=-1, descending = True)
        if X_T_G is not None:
            rank_X_T_G = (argsorted_proba == torch.tensor(X_T_G).unsqueeze(-1).to(device)).nonzero(as_tuple=False)[:,1].detach().cpu().item() #.tolist()
        elif X_T_G_list is not None:
            rank_X_T_G=[]
            for X_T_G in X_T_G_list:
                one_rank = (argsorted_proba == torch.tensor(X_T_G).unsqueeze(-1).to(device)).nonzero(as_tuple=False)[:,1].detach().cpu().item() #.tolist()
                rank_X_T_G.append(one_rank)

    return rank_X_T_G

def get_batch_rank(curr_logits, X_T_G, device=None):
    """
    curr_logits: tensor of shape [batch_size, vocab_size]
    X_T_G: int or tensor of shape [batch_size] with target token ids
    Returns: list of length batch_size with ranks (1-based)
    """
    with torch.no_grad():
        # Sort descending by probability
        probs = torch.softmax(curr_logits, dim=-1)                        # [B, V]
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)    # [B, V]

        # Handle single target for all batch or per-example targets
        if isinstance(X_T_G, int):
            X_T_G = torch.tensor([X_T_G] * curr_logits.size(0), device=curr_logits.device)
        else:
            X_T_G = X_T_G.to(curr_logits.device)

        # Compare each batch row with its target token id
        # (sorted_indices == X_T_G[:, None]) → [B, V] boolean mask
        ranks = (sorted_indices == X_T_G[:, None]).nonzero(as_tuple=False)
        # ranks[:, 1] contains the column (rank position) for each batch row
        ranks = ranks[:, 1] # + 1  # add +1 to make rank 1-based

    return ranks.detach().cpu().tolist()


def re_init_all_but_one(args, modif_pos, modif_tok_ids_1st_proj, optim_embeds, last_tok_nn_ids, optim_embeds_grad, text_embedding=None, model=None):
    if text_embedding is None:
        if args.dataset == "lc_llm_highD":
            model.set_adapter("X_vision_gen")
        text_embedding = model.get_input_embeddings()

    # last_tok_nn_ids of dim [K, nb_optim_toks]
    if len(modif_tok_ids_1st_proj)==1:
        kept_switch_id = modif_tok_ids_1st_proj[0]
    else:
        random.seed(args.seed)
        kept_switch_id = random.choice(modif_tok_ids_1st_proj)

    if args.verbose:
        wrt_full_seq = [modif_pos[i] for i in modif_tok_ids_1st_proj]
        print("all switch token idx list=", wrt_full_seq, "with kept id=", modif_pos[kept_switch_id])

    with torch.no_grad():

        # re-init embeds using last tok ids
        new_optim_embeds = text_embedding(last_tok_nn_ids[0].unsqueeze(0)).detach() # dim [1, nb_optim_toks, emb_dim]

        # modify to have reinit values everywhere except in kept_switch_id
        new_optim_embeds[:,kept_switch_id,:] = optim_embeds[:,kept_switch_id,:].detach().clone()
        new_optim_embeds.requires_grad = True
        new_optim_embeds.grad = optim_embeds_grad

    return new_optim_embeds #  dim [bs=1, nb_optim_toks, emb_dim]



def re_init_all_hard(args, best_optim_tok_ids_for_all_K, text_embedding=None, model=None):
    # best_optim_tok_ids_for_all_K of dim [1, nb_optim_toks]
    if text_embedding is None:
        if args.dataset == "lc_llm_highD":
            model.set_adapter("X_vision_gen")
        text_embedding = model.get_input_embeddings()
    with torch.no_grad():

        # re-init embeds using last tok ids
        new_optim_embeds = text_embedding(best_optim_tok_ids_for_all_K).detach() # dim [1, nb_optim_toks, emb_dim]
        new_optim_embeds.requires_grad = True

    return new_optim_embeds #  dim [bs=1, nb_optim_toks, emb_dim]



def get_fix_token_indices(args, text, tokenizer, fix_parts_list=None, fix_past_traj=True, verbose=False):
    """
    Tokenizes a text and identifies the indices of tokens that, when decoded,
    are fully included in any of the template_list.

    Args:
        text (str): The input text.
        fix_parts_list (list of str): A list of strings representing important parts of the text.
        model_name (str): The name of the Hugging Face tokenizer to use.

    Returns:
        list of int: A list of token indices corresponding to the important parts.
    """
    HF_name=args.HF_name
    text_tokens = tokenizer.tokenize(text)
    if "Llama-3" in HF_name:
        x_ego_only_seq_len = get_ego_only_seq_len(args, tokenizer, text) # with text=X_vision
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        # text_token_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).input_ids.squeeze(0) # in ori code: "encoded_x_sys" is named "inputs"

        if fix_parts_list is None:
            fix_parts_list = [
                "The target vehicle is driving on a",
                " two-lane highway, in the",
                " three-lane highway, in the",
                " four-lane highway, in the",
                " lane.\nThe information about the target vehicle is as follows:\n  - Velocity (km/h): vx",
                ", vy",
                ", ay",
                "\n  - Type:",
                "\n  - Acceleration: ax",
                ", with width of ",
                " m and length of ",
                "  -",
                " traveling at ",
                " km/h of X-axis, with a distance of ",
                ": a",
                ")]\n\nThe information about its surrounding vehicles (within a range of 200 m) is listed as follows:\n  -",
                " m.\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
                "<|start_header_id|>",
                "assistant",
                "<|end_header_id|>",
                "\n\n",
                "\n",
                ".",
                ","
            ]

        if fix_past_traj and args.optim_part !="sv":
            hist_match = re.search(r" m\n  - Historical position.*?(\[.*?\])\n\n", text)
            past_traj_str = hist_match.group(0)
        else:
            past_traj_str = " m\n  - Historical position of the last 2 seconds (One point every 0.4s): ["
        # print("past_traj_str:", past_traj_str)
        fix_parts_list = fix_parts_list + [past_traj_str]

        fix_indices = []
        for part in fix_parts_list:
            part_tokens = tokenizer.tokenize(part)
            part_token_ids = tokenizer.convert_tokens_to_ids(part_tokens)
            # part_token_ids = tokenizer(part, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).input_ids.squeeze(0)

            # Find occurrences of the important part token sequence in the text token sequence
            for i in range(len(text_token_ids) - len(part_token_ids) + 1):
                if text_token_ids[i:i+len(part_token_ids)] == part_token_ids:
                    fix_indices.extend(list(range(i, i + len(part_token_ids))))

        # if fix_ego_car:
        #     fix_indices=fix_indices+list(range(ego_car_end_idx))

        fix_indices = list(set(fix_indices))
        fix_indices.sort()

        if x_ego_only_seq_len is None:
            learnable_indices = [i for i in range(len(text_token_ids)) if i not in fix_indices]
        else:
            learnable_indices = [i for i in range(len(text_token_ids)) if i not in fix_indices and i<x_ego_only_seq_len]

    else:

        # -------------------------------
        # TOKENIZE WITH OFFSETS
        # -------------------------------
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        offsets = enc["offset_mapping"]

        # -------------------------------
        # CHAR → TOKEN SPAN MAPPING
        # -------------------------------
        def char_span_to_tokens(offsets, start, end):
            return [
                i for i, (s, e) in enumerate(offsets)
                if e > start and s < end
            ]

        # post filter for "."
        def token_is_dot(text, offset):
            """Return True if token corresponds only to a dot."""
            s, e = offset
            return text[s:e].strip() == "."

        # -------------------------------
        # REGEX DEFINITIONS (CORRECT)
        # -------------------------------

        FLOAT = r"[+-]?\d+(?:\.\d+)?"
        EQ_FLOAT = rf"(={FLOAT})"


        patterns = [
            # Ego lane
            # r"\b(left|middle|right) lane\b",

            # Ego kinematics (ONLY numbers captured)
            rf"vx{EQ_FLOAT}",
            rf"vy{EQ_FLOAT}",
            rf"ax{EQ_FLOAT}",
            rf"ay{EQ_FLOAT}",

            # Ego type
            r"- Type:\s*(car|truck)",

            # Ego dimensions
            rf"width of ({FLOAT}) m",
            rf"length of ({FLOAT}) m",

            # Surrounding vehicle positions
            r"- (Front side|Back side|Left side|Right side|Left front|Right front|Left rear|Right rear):",

            # Surrounding vehicle types
            r": a (car|truck)\b",

            # Surrounding vehicle speed & distance
            rf"traveling at ({FLOAT}) km/h",
            rf"distance of ({FLOAT}) m",
        ]


        # -------------------------------
        # COLLECT TOKEN INDICES
        # -------------------------------
        all_token_indices = set()

        for pat in patterns:
            for m in re.finditer(pat, text):
                if m.lastindex:
                    start, end = m.span(1)
                else:
                    start, end = m.span()

                # all_token_indices.update(
                #     char_span_to_tokens(offsets, start, end)
                # )
                for tok_idx in char_span_to_tokens(offsets, start, end):
                    if not token_is_dot(text, offsets[tok_idx]):
                        all_token_indices.add(tok_idx)

        learnable_indices = sorted(all_token_indices)

    if verbose:
        # print("total nb of tokens:", len(text_token_ids))
        # print("total nb of fix tokens:", len(fix_indices))
        print("total nb of learnable indices:", len(learnable_indices))
        all_toks = [text_tokens[i] for i in learnable_indices]
        print(all_toks)
    return learnable_indices #, fix_indices


def nn_project_fct(args, curr_embeds, model=None, embedding_layer=None, normalize_version=1, K=None, set_adapter=False, subvoc_idx=None, print_hits=False): # cos, with cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    if set_adapter and args.dataset == "lc_llm_highD":
        model.set_adapter("X_vision_gen")

    if embedding_layer is None:
        embedding_layer = model.get_input_embeddings()

    if K is None:
        K = args.K

    # proj on full vocab
    if subvoc_idx is None:

        with torch.no_grad():

            bs,nb_optim_toks,emb_dim = curr_embeds.shape

            # kNN search: 3 versions give same results and have more or less the same timing
            if normalize_version == 1:
                # s_n=time.time()
                curr_embeds = curr_embeds.reshape((-1,emb_dim)) / torch.norm(curr_embeds.reshape((-1,emb_dim)), dim=-1, keepdim=True) # queries, dim (1xseqlen), embed_dim
                embedding_matrix = embedding_layer.weight / torch.norm(embedding_layer.weight, dim=-1, keepdim=True)
                # print("start semantic search with K=",K)
                hits = semantic_search(curr_embeds, embedding_matrix,
                                        query_chunk_size=curr_embeds.shape[0],
                                        top_k=K,
                                        score_function=dot_score # default is cosine similarity
                                        )
                # print("TIMING norm for proj v1:", time.time() - s_n)

            elif normalize_version == 2:
                # s_n=time.time()
                curr_embeds = normalize_embeddings(curr_embeds.reshape((-1,emb_dim))) # queries, dim (1xseqlen), embed_dim
                embedding_matrix = normalize_embeddings(embedding_layer.weight) # dim voc_size, embed_dim
                hits = semantic_search(curr_embeds, embedding_matrix,
                                        query_chunk_size=curr_embeds.shape[0],
                                        top_k=K,
                                        score_function=dot_score # default is cosine similarity
                                        )
                # print("TIMING norm for proj v2:", time.time() - s_n)

            elif normalize_version == 3:
                # s_n=time.time()
                hits = semantic_search(curr_embeds.reshape((-1,emb_dim)), embedding_layer.weight,
                                        query_chunk_size=curr_embeds.reshape((-1,emb_dim)).shape[0],
                                        top_k=K,
                                        # score_function=dot_score # default is cosine similarity
                                        )
                # print("TIMING norm for proj v3:", time.time() - s_n)

            if print_hits:
                print("hits:", hits)

            # when only 1 optim token, hits is like
            # hits= [[{'corpus_id': 36592, 'score': 0.9921875}, ..., K times ]]
            # when more than 1 optim tokens, hits is like
            # hits= [ [{'corpus_id': 36592, 'score': 0.9921875}, ..., K times ], .., [nb of optim tokens times] ]

            if K>1:
                nn_indices = torch.empty((K,nb_optim_toks), device=curr_embeds.device, dtype = torch.int64)
                # cos_sim_per_nn = torch.empty((K,nb_optim_toks), device=curr_embeds.device, dtype = curr_embeds.dtype)
                for hit_it, hit in enumerate(hits): # iter among nb of optim tokens
                    nn_indices[:, hit_it] = torch.tensor([sub_hit["corpus_id"] for sub_hit in hit], device=curr_embeds.device) # list / tensor of K elems
                    # cos_sim_per_nn[:, hit_it] = torch.tensor([sub_hit["score"] for sub_hit in hit], device=curr_embeds.device) # list / tensor of K elems

            else:
                # with K=1, nn_indices is a tensor of a single dimension of size (nb_optim_toks)
                # # like tensor([23317]), if 2 optim tokens like tensor([36592, 36592]), etc
                nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device).reshape((1,nb_optim_toks))
                # cos_sim_per_nn = torch.tensor([hit[0]["score"] for hit in hits], device=curr_embeds.device).reshape((1,nb_optim_toks))

            # projected_embeds = embedding_layer(nn_indices).detach() # torch.Size([K, nb_optim_toks, emb_dim])
            # K_cos_sim = torch.mean(cos_sim_per_nn, dim=1) # len K


        # return projected_embeds, nn_indices, K_cos_sim

    # proj on subvoc
    elif subvoc_idx is not None:
        # here filter of valid idx for mistral/qwen
        if isinstance(subvoc_idx, list):
            nn_indices = project_to_valid_idx(
                curr_embeds=curr_embeds,                   # (1, T, D)
                embedding_layer=embedding_layer,
                valid_tok_ids=subvoc_idx,
                top_k=K
            )
        # here subproj on k nn for each nb_tok tokens
        else:
            if args.K==1:
                nn_indices = project_on_subvocab(
                    curr_embeds=curr_embeds, embedding_layer=embedding_layer, subvoc_idx=subvoc_idx
                    )
            else:
                raise Exception("not implemented")

    return nn_indices



@torch.no_grad()
def project_to_valid_idx(
    curr_embeds: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
    valid_tok_ids: list[int],
    top_k: int = 1,
):
    """
    Project embeddings onto a restricted vocabulary using cosine similarity.

    Args:
        curr_embeds: (B, T, D) or (N, D)
        embedding_layer: model.get_input_embeddings()
        valid_tok_ids: list of allowed vocab indices
        top_k: number of nearest tokens to return

    Returns:
        projected_ids: (B, T, top_k) or (N, top_k)
        projected_scores: cosine similarities
    """

    device = curr_embeds.device
    emb_dim = curr_embeds.size(-1)

    # ---- reshape queries ----
    flat_embeds = curr_embeds.view(-1, emb_dim)

    # ---- normalize queries ----
    flat_embeds = F.normalize(flat_embeds, dim=-1)

    # ---- get & normalize sub-vocab embeddings ----
    subvocab_ids = torch.tensor(valid_tok_ids, device=device)
    subvocab_embeds = embedding_layer.weight[subvocab_ids]
    subvocab_embeds = F.normalize(subvocab_embeds, dim=-1)

    # ---- cosine similarity ----
    # (N, D) @ (D, V) → (N, V)
    sims = flat_embeds @ subvocab_embeds.T

    # ---- top-k projection ----
    _, indices = sims.topk(top_k, dim=-1) # scores, indices

    # map back to original vocab ids
    projected_ids = subvocab_ids[indices]

    # ---- reshape back ----
    out_shape = curr_embeds.shape[:-1] + (top_k,) # bs=1, nb_toks, top_k=1
    projected_ids = projected_ids.view(out_shape)
    projected_ids = projected_ids.squeeze(0).t() # top_k, nb_toks
    # scores = scores.view(out_shape)
    del subvocab_embeds, flat_embeds, subvocab_ids, sims, indices
    return projected_ids # , scores


def project_on_subvocab(
    curr_embeds,           # (1, nb_toks, emb_dim)
    embedding_layer,       # nn.Embedding
    subvoc_idx             # tensor of size (nb_toks, K)
):
    print("Start project on subvocab")
    # if subvoc_idx List[List[int]] length nb_toks, each length K
    # subvoc_idx = torch.tensor(subvoc_idx, device=device) # (nb_toks, K)
    # print("subvoc_idx.shape:", subvoc_idx.shape)

    device = curr_embeds.device
    emb_dim = curr_embeds.size(-1)
    nb_toks = curr_embeds.size(1)

    # (nb_toks, emb_dim)
    curr_embeds_clone = curr_embeds.squeeze(0).detach().clone()

    # normalize (safety)
    curr_embeds_clone = curr_embeds_clone / curr_embeds_clone.norm(dim=-1, keepdim=True)
    # print("curr_embeds_clone.shape:", curr_embeds_clone.shape)

    # (nb_toks, K, emb_dim)
    subvoc_embeds = embedding_layer.weight[subvoc_idx]
    subvoc_embeds = subvoc_embeds / subvoc_embeds.norm(dim=-1, keepdim=True)
    # print("subvoc_embeds.shape:", subvoc_embeds.shape)

    # dot product: (nb_toks, K)
    scores = torch.einsum("te,tke->tk", curr_embeds_clone, subvoc_embeds)
    # print("scores.shape:", scores.shape)
    del curr_embeds_clone

    # best candidate per token
    best_k_idx = scores.argmax(dim=-1)           # (nb_toks,)
    # print("best_k_idx.shape:", best_k_idx.shape)

    best_vocab_idx = subvoc_idx[
        torch.arange(nb_toks, device=device),
        best_k_idx
    ]                                             # (nb_toks,)
    best_vocab_idx = best_vocab_idx.unsqueeze(0) # (1, nb_toks)


    return best_vocab_idx  #, best_scores


def keep_grad_for_single_token(args, final_grad, modif_pos, update_step, seq_len=False):
    # final_grad of dim  [bs=1, nb_optim_toks, emb_dim] hence needs to use seq_len=False
    # if it was of dim   [bs=1, seq_len, emb_dim] then use seq_len=True
    with torch.no_grad():

        # find index of best grad norm among modif_pos
        if args.one_by_one == "biggest_norm":
            norm = torch.norm(final_grad, dim=-1).squeeze(0) # size seq_len
            if seq_len:
                norm_sort_idx = torch.argsort(norm, descending=True) # size seq_len
                for elem in norm_sort_idx:
                    if elem in modif_pos:
                        kept_dim = elem.item()
                        break
            else:
                kept_dim = torch.argmax(norm).item()

        elif args.one_by_one in ["seq_order", "seq_inv_order"]:
            idx = update_step % len(modif_pos)
            if seq_len:
                if args.one_by_one == "seq_order":
                    kept_dim = modif_pos[idx]
                else:
                    kept_dim = modif_pos[len(modif_pos)-idx-1]
            else:
                if args.one_by_one == "seq_order":
                    kept_dim = idx
                else:
                    kept_dim = len(modif_pos)-idx-1

        if args.verbose:
            print("kept index for grad update (in full seq):", modif_pos[kept_dim])

        # keep gradient info only for the found index
        new_final_grad = torch.zeros(final_grad.shape, dtype=final_grad.dtype, device=final_grad.device)
        new_final_grad[:, kept_dim, :] = final_grad[:, kept_dim, :]
    return new_final_grad


def create_full_mat_of_voc_id(args, step, text_embedding, optim_tok_nn_ids, init_embeds_no_grad, init_tok_ids, optim_slice, model=None, set_adapter=False): #  optim_tok_nn_ids of size [K, nb_optim_toks]

    if set_adapter:
        if args.dataset == "lc_llm_highD":
            model.set_adapter("X_vision_gen")
        text_embedding = model.get_input_embeddings()

    K, nb_optim_toks = optim_tok_nn_ids.shape
    total_nb_forwards = nb_optim_toks * (K-1) + 1

    with torch.no_grad():

        # create a mat of all token ids
        full_mat_of_voc_id = optim_tok_nn_ids[0].unsqueeze(0).expand(total_nb_forwards,-1).detach().clone()
        # full_slice_id_mat = torch.zeros(full_mat_of_voc_id.shape, device=init_tok_ids.device) # cuda
        # full_slice_id_mat[0] = torch.ones(full_mat_of_voc_id.shape[1])
        for i in range(nb_optim_toks):
            first_row_id = i * (K-1) + 1
            full_mat_of_voc_id[first_row_id : first_row_id + (K-1), i] = optim_tok_nn_ids[1:, i]
            # full_slice_id_mat[first_row_id : first_row_id + (K-1), i] = torch.ones(K-1)

        # create the embedding matrix
        full_mat_of_proj_embeds = text_embedding(full_mat_of_voc_id).detach().cpu()
        # full_mat_of_proj_embeds.requires_grad = True

        all_id_combi = init_tok_ids.detach().clone().repeat(total_nb_forwards,1)
        all_id_combi[:, optim_slice == 1] = full_mat_of_voc_id # size [total_nb_forwards, seq_len]
        # else:
        #     all_id_combi = None

        return full_mat_of_proj_embeds, all_id_combi # full_slice_id_mat, full_mat_of_voc_id,

    # if return_emb_mat_with_padding:
    #     # create a full mat of proj embeds with padding of full sent (including unmodif tokens)
    #     tmp_embeds = text_embedding(full_mat_of_voc_id).detach()
    #     tmp_embeds.requires_grad = True
    #     padded_embeds = init_embeds_no_grad.repeat(total_nb_forwards,1,1).detach().clone() # shape [total_nb_forwards, seq_len, emb_dim]
    #     padded_embeds.requires_grad = False
    #     padded_embeds[:, optim_slice == 1, :] = tmp_embeds
    #     # full_mat_of_proj_embeds of dim  [total_nb_forwards, nb_optim_toks, emb_dim]
    #     # full_slice_id_mat and full_mat_of_voc_id are of size [total_nb_forwards, nb_optim_toks]
    #     # padded_embeds of size [total_nb_forwards, seq_len, emb_dim]
    #     return padded_embeds, full_slice_id_mat #, full_mat_of_voc_id,




def extract_G_slice_from_all_grads(args, G): # full_slice_id_mat

    _, nb_optim_toks, emb_dim = G.shape
    G_slice = torch.empty(args.K, nb_optim_toks, emb_dim, dtype=G.dtype, device=G.device)
    G_slice[0,:] = G[0,:]

    # G_slice[1:] = torch.transpose(G[1:][full_slice_id_mat[1:]==1].reshape((nb_optim_toks, args.K-1, emb_dim)), 0, 1) # old code, same res but slower
    for i in range(nb_optim_toks):
        G_slice[1:, i] = G[1 + i * (args.K-1): 1 + (i+1) * (args.K-1), i]

    return G_slice


def extract_prob_for_all_K_and_toks(args, nb_optim_toks, full_prob_X_T_G): # from size total_nb_forwards to [K, nb_optim_toks]
    K_prob_X_T_G=torch.empty(args.K, nb_optim_toks, dtype=full_prob_X_T_G.dtype).to(full_prob_X_T_G.device)
    K_prob_X_T_G[0,:] = full_prob_X_T_G[0].expand(nb_optim_toks)
    for i in range(nb_optim_toks):
        K_prob_X_T_G[1:, i] = full_prob_X_T_G[1 + i * (args.K-1): 1 + (i+1) * (args.K-1)]
    return K_prob_X_T_G

