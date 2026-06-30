# Copyright (c) 2026 Valeo. All rights reserved.
"""Combinatorial candidate generation and batch evaluation for the brute-force
counterfactual search: token-sequence building, product-space sampling, and
parallel candidate scoring against the driving LLM."""
import itertools
import math
import random
import torch
from driv_ex.utils.shared_optim_utils import get_batch_rank


def batched(iterable, batch_size):
    """Simple batching generator."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def pad_sequences(sequences, pad_token_id):
    """
    Pads a list of variable-length token sequences into a batch tensor.

    Args:
        sequences: list[list[int]]
        pad_token_id: int

    Returns:
        input_ids: torch.LongTensor [batch_size, max_len]
        attention_mask: torch.LongTensor [batch_size, max_len]
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    return input_ids, attention_mask


def build_seq(token_id_list, replace_spans, combo):
    """
    Build one tokenized sequence given a specific combination of replacements.

    Args:
        token_id_list: list[int] — base sequence
        replace_spans: TODO
        combo: tuple[list[int]] — one element from itertools.product(*candidate_sets)

    Returns:
        seq: list[int]
    """
    seq = []
    last_pos = 0

    for span, replacement in zip(replace_spans, combo):
        start, end = span[0], span[-1] + 1
        seq.extend(token_id_list[last_pos:start])
        seq.extend(replacement)
        last_pos = end

    seq.extend(token_id_list[last_pos:])
    return seq


def product_size(candidate_sets):
    return math.prod(len(s) for s in candidate_sets)


def nth_product(candidate_sets, n):
    sizes = [len(s) for s in candidate_sets]
    result = []
    for i, size in enumerate(sizes):
        block = 1
        for s in sizes[i+1:]:
            block *= s
        idx = (n // block) % size
        result.append(candidate_sets[i][idx])
    return tuple(result)


def sample_combos_fast(candidate_sets, total_combo_nb, N):
    random.seed(42)
    indices = random.sample(range(total_combo_nb), min(N, total_combo_nb))
    return [nth_product(candidate_sets, idx) for idx in indices]


def single_span_size(candidates_dict):
    """Total number of single-span replacement candidates."""
    return sum(len(replacements) for replacements in candidates_dict.values())


def build_seq_single_span(token_id_list, span, replacement):
    """
    Replace exactly one span in the token sequence.

    Args:
        token_id_list: list[int]
        span: tuple[int] — positions to replace
        replacement: list[int]

    Returns:
        list[int]
    """
    start, end = span[0], span[-1] + 1
    return (
        token_id_list[:start]
        + replacement
        + token_id_list[end:]
    )


def sample_single_span_candidates(token_id_list, candidates_dict, N_sampling, seed=42):
    random.seed(seed)

    all_choices = [
        (span, repl)
        for span, repls in candidates_dict.items()
        for repl in repls
    ]

    if not all_choices:
        return []

    sampled = random.sample(
        all_choices,
        min(N_sampling, len(all_choices))
    )

    return [
        build_seq_single_span(token_id_list, span, repl)
        for span, repl in sampled
    ]


def get_total_nb_of_possible_combinations(candidates_dict, single_tok):
    if not single_tok:
        replace_spans = sorted(candidates_dict.keys(), key=lambda x: x[0])
        candidate_sets = [candidates_dict[span] for span in replace_spans]
        total_combo_nb = product_size(candidate_sets)
    elif single_tok:
        total_combo_nb = single_span_size(candidates_dict)
        print(f"Total nb of possible combinations: {total_combo_nb}")
    return total_combo_nb


def combo_generator(token_id_list, candidates_dict, N_sampling, single_tok, verbose=True):
    if not single_tok:
        replace_spans = sorted(candidates_dict.keys(), key=lambda x: x[0])
        candidate_sets = [candidates_dict[span] for span in replace_spans]
        total_combo_nb = product_size(candidate_sets)
        if verbose:
            print(f"Total nb of possible combinations: {total_combo_nb}")
        # total_combo_nb = len(list(itertools.product(*candidate_sets)))
        # print(f"Total nb of possible combinations: {total_combo_nb}")

        if N_sampling is not None and N_sampling < total_combo_nb:
            sampled_combos = sample_combos_fast(candidate_sets, total_combo_nb=total_combo_nb, N=N_sampling)
            if verbose:
                print(f"Build generator for a subset of {len(sampled_combos)} combinations")
            for combo in sampled_combos:
                yield build_seq(token_id_list, replace_spans, combo)
        else:
            if verbose:
                print(f"Build generator for all combinations")
            for combo in itertools.product(*candidate_sets):
                yield build_seq(token_id_list, replace_spans, combo)
    elif single_tok:
        total_combo_nb = single_span_size(candidates_dict)
        if verbose:
            print(f"Total nb of possible combinations: {total_combo_nb}")

        sampled_combos = sample_single_span_candidates(token_id_list, candidates_dict, N_sampling, seed=42)
        if verbose:
            print(f"Build generator for a subset of {len(sampled_combos)} combinations")
        for seq in sampled_combos:
            yield seq


@torch.no_grad()
def eval_all_candidates(model, tokenizer, input_ids, attention_mask, eos_token_id, loss_fct, X_T_G, max_new_tokens=600):
    """
    Iteratively generate tokens until all sequences reach eos_token_id.
    Returns:
      generated_ids: [bs, <=max_new_tokens]
      seq_lengths: tensor [bs] index of EOS token for each sequence
      past_key_values: KV cache at the end of generation
    """
    model.set_adapter("Y_gen")
    # embedding_layer = model.get_input_embeddings()
    device = model.device
    bs, seq_len = input_ids.shape
    vocab_size = model.config.vocab_size # 128256 for llama3, 152064 for qwen, ? for mistral
    generated_ids = torch.full((bs, 0), tokenizer.pad_token_id, device=device, dtype=torch.long)
    finished = torch.zeros(bs, dtype=torch.bool, device=device)
    seq_lengths = torch.full((bs,), -1, device=device, dtype=torch.long)
    all_logits = torch.empty((bs,0,vocab_size), device=device)
    past_key_values = None

    for step in range(max_new_tokens):
        torch.manual_seed(0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        all_logits = torch.cat((all_logits, outputs.logits), dim=1)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

        # mark EOS positions the first time they appear
        just_finished = (next_tokens == eos_token_id) & ~finished
        seq_lengths[just_finished] = seq_len + step
        finished |= just_finished

        # embed next tokens for next iteration
        # next_embeds = embedding_layer(next_tokens.unsqueeze(1))
        # input_embeds = next_embeds  # only pass the new token next time
        input_ids = next_tokens.unsqueeze(1)
        attention_mask = torch.cat([attention_mask, torch.ones((bs, 1), device=device, dtype=torch.long)], dim=1)

        if finished.all():
            break

    # change attention mask to mask what happens after eos_token_id
    # last step with class prediction
    torch.manual_seed(0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

    all_logits = torch.cat((all_logits, outputs.logits), dim=1)

    batch_final_token_logits = all_logits[torch.arange(bs), (seq_lengths)] # (seq_lengths-start_idx)]
    batch_rank_X_T_G =  get_batch_rank(batch_final_token_logits, X_T_G, device=device)
    batch_final_pred_tokens = torch.argmax(batch_final_token_logits, dim=-1)

    batch_loss = loss_fct(batch_final_token_logits, torch.tensor([X_T_G]).expand(bs).to(device)) # size current_bs
    batch_prob_xtg = (torch.exp(-batch_loss)*100).detach().cpu().tolist()

    # gen_ids_deco = [s.rstrip("\n") for s in tokenizer.batch_decode(generated_ids)]
    gen_ids_deco = tokenizer.batch_decode(generated_ids)

    del attention_mask, past_key_values, finished, just_finished, input_ids
    del batch_final_token_logits, generated_ids, next_token_logits, next_tokens, all_logits, seq_lengths, outputs #, batch_loss,
    torch.cuda.empty_cache()

    return batch_loss, batch_rank_X_T_G, batch_prob_xtg, gen_ids_deco, batch_final_pred_tokens # generated_ids