# Copyright (c) 2026 Valeo. All rights reserved.
"""HuggingFace model/tokenizer loading, generation helpers, and LLM-specific
token/vocabulary mappings (EOS tokens, class-to-vocab-ID, vocabulary filtering,
adapter loading, and BERTScore semantic similarity scoring)."""
import json
import os
import random
import time
import torch
import sys
from driv_ex import REPO_DIR



def load_base_and_adaptors(args, FT_type, return_bertscore=True, return_sbert=False, no_config=False, load_X_vision_gen=True, return_no_config_data=False):
    from driv_ex.utils.shared_base_utils import resolve_ckpt_path  # local import to avoid circular dependency
    FT_type = None if FT_type == "classic_FT" else FT_type.replace("_FT", "")
    if no_config:
        quantization = "8bit"
        lr, bs = 0.0005, 8  # defaults matching released checkpoints
        if args.dataset == "lc_llm_highD":
            # choosing best ckpt for llama3 and mistral for classic_FT_on_base_data
            if "Llama-3" in args.HF_name:
                if FT_type is None: # classic_FT
                    ckpt_nb_y_gen=4400
                    ckpt_nb_x_vision=750
                elif FT_type in ["vehicle_bias", "digit_bias"]:
                    ckpt_nb_y_gen=2250
                    ckpt_nb_x_vision=750
                elif "vy" in FT_type or "free_front" in FT_type:
                    ckpt_nb_y_gen=2250
                    # ckpt_nb_x_vision=600 => see below
                else:
                    raise Exception(f"ckpt_nbs not yet identified for FT_type {FT_type}")
                grad_cumul=8

            elif "Qwen" in args.HF_name:
                if FT_type is None: # classic_FT
                    ckpt_nb_y_gen=8000
                    ckpt_nb_x_vision=800
                else:
                    raise Exception("no biased FT for qwen")
                grad_cumul=8

            elif "Mistral" in args.HF_name:
                if FT_type is None: # classic_FT
                    ckpt_nb_y_gen=1100
                    ckpt_nb_x_vision=500
                else:
                    raise Exception("no biased FT for mistral")
                grad_cumul=32
    else:
        quantization=args.eval_quantization
        lr, bs = args.FT_learning_rate, args.per_device_train_batch_size
        ckpt_nb_y_gen=args.ckpt_nb
        ckpt_nb_x_vision = args.ckpt_nb_x_vision
        grad_cumul = args.grad_cumul

    if args.dataset == "lc_llm_highD":
        if FT_type is not None:
            if "vy" not in FT_type and "free_front" not in FT_type:
                gen_X_FT_type = f"X_vision_{FT_type}_FT"
                gen_Y_FT_type = f"{FT_type}_FT"
            else:
                dico = {
                    ##
                    "vy_out": {
                        "gen_X_FT_type": "X_vision_FT", "ckpt_nb_x_vision":750,
                        "gen_Y_FT_type": "vy_out_FT"},
                    ##
                    "vy_in_out": {
                        "gen_X_FT_type": "X_vision_vy_in_out_FT", "ckpt_nb_x_vision":600,
                        "gen_Y_FT_type": "vy_in_out_FT" },
                    ##
                    "free_front": {
                        "gen_X_FT_type": "X_vision_FT", "ckpt_nb_x_vision":750,
                        "gen_Y_FT_type": "free_front_FT"},
                    ##
                    "vy_out_free_front": {
                        "gen_X_FT_type": "X_vision_FT", "ckpt_nb_x_vision":750,
                        "gen_Y_FT_type": "vy_out_free_front_FT"},
                    ##
                    "vy_in_out_free_front": {
                        "gen_X_FT_type": "X_vision_vy_in_out_FT", "ckpt_nb_x_vision":600,
                        "gen_Y_FT_type": "vy_in_out_free_front_FT"},
                    ##
                    "just_vy": {
                        "gen_X_FT_type": "X_vision_just_vy_FT", "ckpt_nb_x_vision":600,
                        "gen_Y_FT_type": "just_vy_FT"}
                }
                gen_X_FT_type = dico[FT_type]["gen_X_FT_type"]
                gen_Y_FT_type = dico[FT_type]["gen_Y_FT_type"]
                if no_config:
                    ckpt_nb_x_vision = dico[FT_type]["ckpt_nb_x_vision"]

        else:
            gen_X_FT_type = "X_vision_FT"
            gen_Y_FT_type = "classic_FT"

        print(f"load {args.HF_name}, gen_Y_FT_type={gen_Y_FT_type}, ckpt y_gen={ckpt_nb_y_gen}, gen_X_FT_type={gen_X_FT_type}, ckpt x_vision={ckpt_nb_x_vision}, quant={quantization}, grad_cumul={grad_cumul}")
    else:
        print(f"load {args.HF_name}, quant={quantization}, (off-the-shelf)")

    model, tokenizer, _, _ = load_HF_model_tok(
        HF_name=args.HF_name,
        eval_mode=True,
        freeze_params=True,
        FT_mode=False,
        timing=True,
        quantization=quantization,
        return_text_embedding=False,
        FT_LoRA_folder=None
        )
    model.eval()
    model.temperature = None
    model.top_p=None
    # 3 lines below used to be in algo script
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None

    print("device:", model.device)

    if args.dataset == "lc_llm_highD":
        if "Mistral" in args.HF_name or "Qwen" in args.HF_name:
            HF_name = args.HF_name.replace("/", "_")
        elif "Llama-2" in args.HF_name:
            HF_name = args.HF_name.split("/")[-1] if "/" in args.HF_name else args.HF_name
        elif "Llama-3" in args.HF_name:
            HF_name = args.HF_name
        long_subdir_name = f"{HF_name}_quant_{quantization.replace('bit', '')}_lr_{lr}_bs_{bs}_grad_cumul_{grad_cumul}"

        if load_X_vision_gen:
            X_vision_gen_FT_path = resolve_ckpt_path(gen_X_FT_type, args.HF_name, long_subdir_name, ckpt_nb_x_vision, grad_cumul, lr, bs)
            model.load_adapter(X_vision_gen_FT_path, adapter_name="X_vision_gen")

        Y_gen_FT_path = resolve_ckpt_path(gen_Y_FT_type, args.HF_name, long_subdir_name, ckpt_nb_y_gen, grad_cumul, lr, bs)
        model.load_adapter(Y_gen_FT_path, adapter_name="Y_gen")

        model.disable_adapters()
        model.enable_adapters()

    if "Llama-3" in args.HF_name:
        assert model.get_input_embeddings() is model.model.embed_tokens
    # True

    if return_bertscore:
        # -------------------------
        # BERTScore
        # we use "microsoft/deberta-xlarge-mnli" as advised on BERTScore repo (https://github.com/Tiiiger/bert_score/tree/master)
        # model_type = "roberta-large" if args.small_scorer else "microsoft/deberta-xlarge-mnli"
        from bert_score import BERTScorer
        import bert_score.utils
        local_path = str(REPO_DIR / "LLM_cache" / "deberta-xlarge-mnli")
        base_model_name = "microsoft/deberta-xlarge-mnli"
        model_type = local_path if os.path.isdir(local_path) else base_model_name
        print("Load BERTScore, instantiated with", base_model_name)

        # 1. To use a local model, first trick bert_score by adding your local path to its internal dictionary
        if model_type != base_model_name and base_model_name in bert_score.utils.model2layers:
            bert_score.utils.model2layers[model_type] = bert_score.utils.model2layers[base_model_name]

        scorer = BERTScorer( # uses ~ 3.5 GB
            model_type=model_type, #"microsoft/deberta-xlarge-mnli", #  default roberta-large
            lang="en", # language of the sentences
            rescale_with_baseline=True, # rescale bertscore with pre-computed baseline
            batch_size=1, # bert score processing batch size
            # device=None, #  If this argument is None, the model lives on cuda:0 if cuda is available.
            # idf=True # specify whether to use optional idf importance weighting
            )

        # 2. Fix the locked model_type
        scorer._model_type = base_model_name # ie "microsoft/deberta-xlarge-mnli"

        # 3. Fix the corrupted baseline_path explicitly!
        # We build the exact path to where bert_score keeps its built-in TSV files.
        correct_baseline_path = os.path.join(
            bert_score.__path__[0],
            "rescale_baseline",
            "en",
            "microsoft",
            "deberta-xlarge-mnli.tsv"
        )
        scorer.baseline_path = correct_baseline_path
        print("BERTScore Loading => Done")

    else:
        scorer=None

    if return_sbert:
        # SBERT
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # scorer = SentenceTransformer(SBERT_MODEL_NAME, device=device)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # MiniLM models are distilled versions of larger BERT models,
        # offering near-SOTA performance with much lower memory and faster inference.
        # SBERT_MODEL_NAME="all-MiniLM-L6-v2" # General-purpose:
        SBERT_MODEL_NAME="all-MiniLM-L12-v2" # Higher accuracy
        sbert_scorer = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    else:
        sbert_scorer = None

    if no_config and return_no_config_data:
        if args.dataset == "lc_llm_highD":
            no_config_data= {"ckpt_nb":ckpt_nb_y_gen,
                            "ckpt_nb_x_vision":ckpt_nb_x_vision}
        else:
            no_config_data=None
        return model, tokenizer, scorer, sbert_scorer, no_config_data
    return model, tokenizer, scorer, sbert_scorer



def get_valid_vocab(args, tokenizer):
    # avoid instable regions of Mistral and Qwen latent spaces
    if "Mistral" in args.HF_name or "Qwen" in args.HF_name:
        valid_tok_ids = [
            tok_id for tok, tok_id in tokenizer.get_vocab().items()
            if not tok.startswith("[") and tok!="<unk>" and tok.isprintable()
        ]
    else:
        valid_tok_ids=None
    return valid_tok_ids




def transform_class_into_voc_id(HF_name, label):
    if "Llama-3" in HF_name:
        if label == 0:
            voc_id=15
        elif label == 1:
            voc_id=16
        elif label == 2:
            voc_id=17

    elif "Llama-2" in HF_name:
        if label == 0:
            voc_id=29900
        elif label == 1:
            voc_id=29896
        elif label == 2:
            voc_id=29906

    elif "Mistral" in HF_name:
        if label == 0:
            voc_id=29502
        elif label == 1:
            voc_id=29508
        elif label == 2:
            voc_id=29518

    elif "Qwen" in HF_name:
        if label == 0:
            voc_id=15
        elif label == 1:
            voc_id=16
        elif label == 2:
            voc_id=17

    else:
        raise Exception
    return voc_id





def get_eos_token(HF_name):
    if "Llama-2" in HF_name or "Llama-3" in HF_name:
        # max_new_tokens=None
        if "Llama-2" in HF_name:
            eos_token_id=376 #  [3159, 2509, 29901, 376] = 'Int', 'ention', ':', '"'
        elif "Llama-3" in HF_name:
            eos_token_id=330 #  [1357, 3012, 25, 330] = ' Int', 'ention', ':', ' "'
    elif "Mistral" in HF_name:
        # max_new_tokens=600
        eos_token_id=1113 # [5434, 2916, 29515, 1113] : 'Int', 'ention', ':', '"'
    elif "Qwen" in HF_name:
        eos_token_id = 330 # [1333, 2939, 25, 330] :  'Int', 'ention', ':',  '"'
    return eos_token_id



def gen_answer(HF_name, model, tokenizer, prompt, FT_type, gen_until_end, strip_special_toks=True, verbose=False):

    max_new_tokens = None
    # For 0 shot and last ckpt (or best ckpt later), stop at eos token to save full results
    if gen_until_end:
        if FT_type == "0_shot":
            if "Llama-3" in HF_name:
                eos_tok_id = tokenizer.eos_token_id
            elif "Llama-2" in HF_name:
                # eos_tok_id = tokenizer.eos_token_id # too long so cut at trajectory
                eos_tok_id = 622 # 'ject' (3201='Tra', 706='ory')
                max_new_tokens=500
            elif "Mistral-7B-Instruct-v0.3" in HF_name or "Qwen2.5" in HF_name:
                eos_tok_id = tokenizer.eos_token_id
                max_new_tokens=600 # just a large nb to override max_length

        else: # elif FT_type in ["classic_FT", ...]:
            if "Llama-3" in HF_name:
                eos_tok_id = 7400 # ')]'
            elif "Llama-2" in HF_name:
                eos_tok_id = 4638 # ')]'
            elif "Mistral-7B-Instruct" in HF_name:
                eos_tok_id = 5521 # 5521=')]', 29507='"'
                max_new_tokens=600 # just a large nb to override max_length
            elif "Qwen2.5" in HF_name:
                eos_tok_id = 7252 # ')]'
                max_new_tokens=600
    # for intermediary FT output, stop at 'trajectory' just to eval the classif task
    else:
        if FT_type == "0_shot":
            raise Exception("not implem for 0shot + not gen_until_end")
        else:
            if "Llama-3" in HF_name:
                eos_tok_id = 24251 # 'jectory' (to stop at 'trajectory') - otherwise ' Tra' = 17747
            elif "Llama-2" in HF_name:
                eos_tok_id = 622 # 'ject' (3201='Tra', 706='ory')
            elif "Mistral-7B-Instruct-v0.3" in HF_name:
                eos_tok_id = 1432 # 'Tra' 9302, 'ject' 1432, 'ory' 1463
                max_new_tokens=600 # just a large nb to override max_length
            elif "Qwen2.5" in HF_name:
                eos_tok_id = 17298 # 'ĠTra' (also, 23363 : 'jectory')
                max_new_tokens=600

    with torch.no_grad():
        encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
        model_inputs = encoded_input.to(model.device) # .to("cuda")
        input_length= model_inputs['input_ids'].shape[1] # same as len(model_inputs['input_ids'][0])
        # print("len input ids:", input_length)

        # if "Llama" in HF_name or "llama" in HF_name:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k=None
        torch.manual_seed(42)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_tok_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        result = tokenizer.batch_decode(generated_ids[:, input_length:])
        full_scenario = tokenizer.batch_decode(generated_ids)

        if strip_special_toks and gen_until_end:
            if "Mixtral" in HF_name or "Llama-2" in HF_name or "Mistral-7B" in HF_name:
                result = [one_result.strip("</s>").strip()+'\"' for one_result in result]
                full_scenario = [one_scenario.strip("</s>").strip()+'\"' for one_scenario in full_scenario]
            elif "Llama-3" in HF_name or "llama-3" in HF_name:
                result = [one_result.strip("<|eot_id|>").strip()+'\"' for one_result in result]
                full_scenario = [one_scenario.strip("<|eot_id|>").strip()+'\"' for one_scenario in full_scenario]
            elif "Qwen2.5" in HF_name:
                result = [one_result.strip("<|im_end|>").strip()+'\"' for one_result in result]
                full_scenario = [one_scenario.strip("<|im_end|>").strip()+'\"' for one_scenario in full_scenario]
            elif "gemma" in HF_name:
                if "gemma-2-2b-it" in HF_name:
                    print("gemma : before strip:", result)
                result = [
                    one_result.strip("<eos>").strip().strip("\n").strip().strip("<end_of_turn>").strip().strip("\n").strip()+'\"'
                    for one_result in result]
                full_scenario = [
                    one_scenario.strip("<eos>").strip().strip("\n").strip().strip("<end_of_turn>").strip().strip("\n").strip()+'\"'
                    for one_scenario in full_scenario]
                if "gemma-2-2b-it" in HF_name:
                    print("gemma : after strip:", result)
            elif "gpt-neo" in HF_name:
                result = [one_result.strip()+'\"' for one_result in result]
                full_scenario = [one_scenario.strip()+'\"' for one_scenario in full_scenario]
            else:
                print("WARNING stripping on generation by this LLM not implemented")

        if verbose:
            strip_str = "(stripped)" if strip_special_toks else "(unstripped)"
            print(f"LLM generated {strip_str} result:", result)

    return result, full_scenario




def get_logits(model, padded_embeds, image_features=None, decoded_ids=None, which_logits=None, check_coord_tokens=True, set_adapter=False):
    if set_adapter:
        model.set_adapter("Y_gen")
    # LLM forward
    proj_embed_input = {'inputs_embeds': padded_embeds,
                        'attention_mask': torch.ones((padded_embeds.shape[0], padded_embeds.shape[1])).to(model.device)}
    output = model(**proj_embed_input) # output.logits.shape = [current_bs, seq_len, voc_size]
    gen_ids = None
    return output.logits[:, -1], gen_ids # torch.Size([current_bs, voc_size])





# Functions for LLM loading
def get_HF_model_id(HF_LLM_name):

    if "Mixtral_8x7" in HF_LLM_name or "Mixtral-8x7" in HF_LLM_name:
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        base_model_name = "Mixtral-8x7B-Instruct-v0.1"

    elif "Mixtral_8x22" in HF_LLM_name or "Mixtral-8x22" in HF_LLM_name:
        model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        base_model_name = "Mixtral-8x22B-Instruct-v0.1"

    elif "Llama-3-8B" in HF_LLM_name or "Llama3_8" in HF_LLM_name:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        base_model_name = "Meta-Llama-3-8B-Instruct"

    elif "Llama-3-70B" in HF_LLM_name or "Llama3_70" in HF_LLM_name:
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        base_model_name = "Meta-Llama-3-70B-Instruct"

    elif "gemma-2-9b-it" in HF_LLM_name:
        model_id = "google/gemma-2-9b-it"
        base_model_name = "gemma-2-9b-it"

    elif "gemma-2-2b-it" in HF_LLM_name:
        model_id = "google/gemma-2-2b-it"
        base_model_name = "gemma-2-2b-it"

    elif "gpt-neo-2.7B" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-2.7B"
        base_model_name = "gpt-neo-2.7B"

    elif "gpt-neo-1.3B" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-1.3B"
        base_model_name = "gpt-neo-1.3B"

    elif "gpt-neo-125m" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-125m"
        base_model_name = "gpt-neo-125m"

    else:
        model_id = HF_LLM_name
        base_model_name = HF_LLM_name.split("/")[-1] if "/" in HF_LLM_name else HF_LLM_name
        # raise Exception(f"not implemented for the LLM ({HF_LLM_name})")

    return model_id, base_model_name


def load_HF_model_tok(
    HF_name,
    main_folder=None,
    eval_mode=True,
    freeze_params=True,
    FT_mode=False,
    timing=True,
    quantization=None,
    return_text_embedding=False,
    FT_LoRA_folder=None,
    return_tokenizer_only = False,
    HF_token_path = None
    ): # ft_dataset=None,

    if timing:
        s = time.time()

    if HF_token_path is None:
        HF_token_path = REPO_DIR / "driv_ex" / "LLM_token.json"

    with open(HF_token_path) as f:
        HF_TOKEN = json.load(f)

    from huggingface_hub import login as hf_login
    hf_login(token=HF_TOKEN)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Find base model
    model_id, base_model_name = get_HF_model_id(HF_LLM_name=HF_name)

    # Define model / tokenizer's cache folders
    if main_folder is None:
        main_folder=REPO_DIR

    cache_dir = os.path.join(main_folder, "LLM_cache", f"cache_{base_model_name}")
    cache_dir_tok = os.path.join(main_folder, "LLM_cache", f"cache_{base_model_name}_tokenizer")
    print(f"(Down)loading base model {base_model_name} from/to {main_folder}/LLM_cache/")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir_tok, token=HF_TOKEN, force_download=False
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if return_tokenizer_only:
        return tokenizer

    # Load base model
    use_cache = True if FT_mode is False else False
    attn_implem = "eager" if "gemma" in HF_name else "flash_attention_2"

    if quantization is not None:

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            # quantization_config=None
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise Exception(f"Not implem for quantization={quantization}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            # load_in_8bit=True if quantization=="8bit" else False,
            quantization_config=quantization_config,
            use_cache=use_cache,
            attn_implementation=attn_implem,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16 # recently added
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            use_cache=use_cache,
            attn_implementation=attn_implem,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16
        )

    # add FT lora weights to base model by specifying folder where files like adapter_config.json are
    if FT_LoRA_folder is not None:
        print(f"Add to base model, LoRA weights from {FT_LoRA_folder}")
        # path_to_folder = os.path.join(main_folder, "my_FT_models")
        # assert os.path.exists(os.path.join(path_to_folder, HF_name))

        assert os.path.exists(FT_LoRA_folder)
        model = load_FT_model_via_base_model(
            base_model=model,
            complete_FT_path=FT_LoRA_folder,
            timing=timing,
        )

    if eval_mode:
        model.eval()

    if freeze_params:
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

    voc_size = len(tokenizer)
    device = model.device # "cuda" if torch.cuda.is_available() else "cpu"
    print("model device:", model.device)

    if timing:
        e = time.time()
        print("HF LLM / tokenizer loading time:", round(e - s, 1), "secs")

    if return_text_embedding:

        # Set embedding layer
        if HF_name == "gpt-neo-125m":
            text_embedding = {"wte": model.transformer.wte, "wpe": model.transformer.wpe}
            # emb_dim =
        elif HF_name == "Meta-Llama-3-8B-Instruct":
            text_embedding = model.model.embed_tokens
            emb_dim = 4096

        return model, text_embedding, tokenizer, voc_size, emb_dim, device

    else:

        return model, tokenizer, voc_size, device


def load_FT_model_via_base_model(base_model, complete_FT_path, timing):
    if timing:
        s = time.time()

    # new loading way (load base model + give checkpoint path)
    from peft import PeftModel

    print(f"add lora adapters to base model from: {complete_FT_path}")
    FT_model = PeftModel.from_pretrained(base_model, complete_FT_path)
    if timing:
        e = time.time()
        print("Time to load lora modules on base model:", round(e - s, 2), "secs")
    return FT_model



def semantic_similarity_bertscore(cands, refs, scorer):
    # print("compute bertscore")
    if isinstance(cands, str):
        cands = [cands]
    if isinstance(refs, str):
        refs = [refs]
    # outputs are tensors of dim 1 and len = len(sent_b)=len(sent_a) with float elems
    with torch.no_grad():
        P_ten, R_ten, F1_ten = scorer.score(cands=cands, refs=refs, batch_size=len(cands))

    F1_list = F1_ten.detach().cpu().tolist()
    del P_ten, R_ten, F1_ten
    return F1_list

