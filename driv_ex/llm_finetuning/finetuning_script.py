# Copyright (c) 2026 Valeo. All rights reserved.
import json
import os
import time
import sys
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, prepare_model_for_int8_training
from torch.utils.data import DataLoader
from transformers import TrainingArguments # TrainerCallback
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from driv_ex.llm_finetuning.finetuning_config import get_args
from driv_ex.utils.LLM_utils import load_HF_model_tok
from driv_ex.dataset.textual_highD_dataset import towards_llama3_template, towards_qwen_template



# ----------------------------------------------------------------------
# Function specifying the LLM modules to finetune with LoRA
# ----------------------------------------------------------------------


def define_modules_to_FT(my_args):
    # Set the modules to apply the adapter to.

    if my_args.modules_to_FT is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]

    elif my_args.modules_to_FT == "lm_head_only":
        target_modules = [
            "lm_head",
        ]

    elif my_args.modules_to_FT == "last_4_only":
        target_modules = [
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]

    elif my_args.modules_to_FT == "like_lcllm":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return target_modules


# ----------------------------------------------------------------------
# Functions to get the finetuning data in the right format
# ----------------------------------------------------------------------


def create_my_datasetdict(HF_name, data_fp, max_len=None, verbose=True, print_one_data=False):

    s=time.time()
    with open(data_fp) as f:
        prompt_dataset = json.load(f)

    # Create a list of texts with the right prompt format
    if "Llama-3" in HF_name:
        X = [towards_llama3_template(val["text"]) for val in prompt_dataset]

    elif "Mistral-7B-Instruct" in HF_name or "Llama-2-13b-chat-hf" in HF_name:
        # X = [val["text"].replace(' rig ',' truck ').replace(' rig,', ' truck,') for val in prompt_dataset]
        X = [val["text"] for val in prompt_dataset]

    elif "Qwen2.5" in HF_name:
        X = [towards_qwen_template(val["text"]) for val in prompt_dataset]

    else:
        raise Exception(f"Not implemented for LLM={HF_name}")

    X = X[:max_len] if max_len is not None else X
    if print_one_data:
        print("one data:\n", X[0])

    # Make a dict then a dataframe then a DatasetDict from it
    X_dict = {"text": X}
    df = pd.DataFrame(X_dict)
    my_dataset = Dataset.from_pandas(df)

    if verbose:
        e = time.time()
        print(f"Built fine-tuning dataset of len {len(my_dataset)} in {round(e-s, 1)} secs")

    return my_dataset



# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------


def main(my_args):


    #--- Print important args
    target_modules=define_modules_to_FT(my_args=my_args)
    ft_objective="Fluency Expert" if my_args.FT_on_inputs else "Driving LLM"
    print(f"\nFine-tuning {my_args.HF_name} to be a {ft_objective}")
    print(f"Params:  epoch={my_args.num_train_epochs}  | BS = {my_args.per_device_train_batch_size} | LR = {my_args.FT_learning_rate} | grad_cumul = {my_args.gradient_accumulation_steps}")
    print(f"LoRA params: r={my_args.lora_r} | alpha={my_args.lora_alpha}")
    print("Trainable modules:", target_modules)


    #--- Define finetuning datasets

    my_VAL_dataset = create_my_datasetdict(
        HF_name=my_args.HF_name,
        data_fp=my_args.validation_files,
        max_len=my_args.max_eval_samples,
    )

    my_TRAIN_dataset = create_my_datasetdict(
        HF_name=my_args.HF_name,
        data_fp=my_args.train_files,
        max_len=my_args.max_train_samples,
    )


    #--- Define filepaths where to save checkpoints / tensorboards data
    # (test first if output dir already exists
    # (it must exist if --resume_from_checkpoint, it mustn't exist if not --resume_from_checkpoint))

    model_save_name = f"{my_args.HF_name.replace('meta-llama/', '').replace('/', '_')}_quant_{my_args.load_in_bits}_lr_{my_args.FT_learning_rate}_bs_{my_args.per_device_train_batch_size}_grad_cumul_{my_args.gradient_accumulation_steps}"
    if my_args.lora_r != 64:
        model_save_name += f"_r_{my_args.lora_r}"
    if my_args.lora_alpha != 16:
        model_save_name += f"_a_{my_args.lora_alpha}"

    output_dir = os.path.join(my_args.output_dir, model_save_name)
    print("output_dir:", output_dir)

    if my_args.resume_from_checkpoint and not os.path.exists(output_dir):
        raise Exception("Cannot resume as output_dir does not exist at:", output_dir)
    if not my_args.resume_from_checkpoint and os.path.exists(output_dir):
        raise Exception("Output dir already exists, cannot overwrite it. Output dir:\n", output_dir)
    print("output_dir already exists ?", os.path.exists(output_dir))


    #--- Set training config

    peft_config = LoraConfig(
        lora_alpha=my_args.lora_alpha,  # The alpha parameter for Lora scaling
        lora_dropout=0.1,  # The dropout probability for Lora layers
        r=my_args.lora_r,  # Lora attention dimension (the “rank”).
        bias="none", # Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    main_step = 50
    args = TrainingArguments(
        disable_tqdm=False,
        output_dir=output_dir,  # where to save preds and checkpoints
        num_train_epochs=my_args.num_train_epochs,
        # max_steps = 500, # comment out this line if you want to train in epochs
        per_device_train_batch_size=my_args.per_device_train_batch_size,
        per_device_eval_batch_size=my_args.per_device_eval_batch_size,
        warmup_steps=my_args.warmup_steps,
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=main_step,
        save_strategy="steps",  # choices=["steps", "no", "epoch"]
        save_steps=main_step,  # needs to be multiple of eval_steps if load_best_model_at_end == True
        # if "steps" as not "no" this sets do_eval to True / with "steps".
        # Evaluation is done (and logged) every eval_steps. (Also "epoch")
        evaluation_strategy="steps",
        eval_steps=main_step,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=my_args.FT_learning_rate,
        bf16=True,
        # lr_scheduler_type='constant',
        # save_total_limit = 100,
        # load_best_model_at_end = True,
        metric_for_best_model="loss",
        greater_is_better=False,  # use False if use eval loss metric, True if use accuracy or similar
        logging_dir=os.path.join(
            my_args.output_dir,
            "trainer_logdir",
            model_save_name,
        ),  # (str, optional) – Tensorboard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME**.
        seed = my_args.seed,
        data_seed = my_args.seed,
        dataloader_num_workers=my_args.num_workers,  # (int, optional, defaults to 0)–
        gradient_accumulation_steps=my_args.gradient_accumulation_steps,
    )


    #--- Define LLM to fine-tune (and use a first callback eval if specified)

    quantization=None if my_args.load_in_bits is None else f"{my_args.load_in_bits}bit"

    model, tokenizer, voc_size, device = load_HF_model_tok(
        HF_name=my_args.HF_name,
        main_folder=my_args.main_folder, # path to llm cache
        eval_mode=False,
        freeze_params=False,
        FT_mode=True,
        timing=True,
        quantization=quantization, # None, "4bit", "8bit"
        return_text_embedding=False,
        FT_LoRA_folder=None,
        HF_token_path=None # resolves to  REPO_DIR / "driv_ex" / "LLM_token.json"
    )

    if my_args.load_in_bits==8:
        model = prepare_model_for_int8_training(model)
    elif my_args.load_in_bits==4:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # print details on trainable params

    print(model.print_trainable_parameters())
    def module_requires_grad(module):
        return any(p.requires_grad for p in module.parameters())
    print("lm_head trainable?", module_requires_grad(model.lm_head))
    emb = model.get_input_embeddings()
    print("embedding trainable?", emb.weight.requires_grad)


    #--- Setting up collator to fine-tune on correct completion only

    if not my_args.FT_on_inputs:
        if "Llama-2" in my_args.HF_name:
            response_template_string = "[/INST]"
            response_template_ids = tokenizer.encode(response_template_string, add_special_tokens=False) #[1:]
        elif "Llama-3" in my_args.HF_name:
            response_template_ids = [78191, 128007, 271] #  "assistant", "<|end_header_id|>" "\n\n" , (before ' [/INST]')
        elif "Mistral-7B-Instruct-v0.3" in my_args.HF_name:
            response_template_ids = [781, 29473, 4] # [781, 29473, 4] = '\n', '', '[/INST]'
        elif "Qwen2.5" in my_args.HF_name:
            response_template_ids = [151644, 77091, 198] # "<|im_start|>" "assistant" "\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)

        # sanity check on 1 sample that collator properly set
        # test inspired from: https://gist.github.com/Blaizzy/40de0f6b4340490e3920db9e182e6455#file-datacollatorforcompletiononlylm
        tokenized_sample = tokenizer(my_VAL_dataset[0]['text'], padding=True, truncation=True, return_special_tokens_mask=True)
        collator_output = collator.torch_call([tokenized_sample])
        if "Llama-3" in my_args.HF_name:
            all_masked_ids = torch.nonzero(collator_output['labels'].squeeze(0)[:-5]==-100, as_tuple=True)[0]
            deco = tokenizer.decode(tokenized_sample['input_ids'][max(all_masked_ids)+1:max(all_masked_ids)+2])
            print(f"deco: '{deco}'")
            assert 'Thought' in deco
        elif "Llama-2" in my_args.HF_name or "Mistral-7B-Instruct-v0.3" in my_args.HF_name or "Qwen2.5" in my_args.HF_name:
            all_masked_ids = torch.nonzero(collator_output['labels'].squeeze(0)[:-5]==-100, as_tuple=True)[0]
            assert 'Thought' in tokenizer.decode(tokenized_sample['input_ids'][max(all_masked_ids)+1:max(all_masked_ids)+5])

    else:
        # if train on X_vision only, given X_sys as input
        if "Llama-3" in my_args.HF_name:
            response_template_id = [882, 128007, 271]  # "user" "<|end_header_id|>" "\n\n"
        elif "Llama-2" in my_args.HF_name:
            response_template_id=[29966, 829, 14816, 29903, 6778, 13, 13] # for llama2: '<</SYS>>\n\n'
        elif 'Mistral-7B' in my_args.HF_name:
            response_template_id=[29557, 1468, 19509, 4828, 781, 781] # '<', '</', 'SYS', '>>', '\n', '\n'
        elif "Qwen2.5" in my_args.HF_name:
            response_template_id = [151644, 872, 198] # "<|im_start|>" "user" "\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_id, tokenizer=tokenizer)

        # sanity check
        tokenized_sample = tokenizer(my_VAL_dataset[0]['text'], padding=True, truncation=True, return_special_tokens_mask=True)
        collator_output = collator.torch_call([tokenized_sample])
        if "Llama-3" in my_args.HF_name or 'Mistral-7B' in my_args.HF_name or "Qwen2.5" in my_args.HF_name:
            non_masked_ids = torch.nonzero(collator_output['labels'].squeeze(0)!=-100, as_tuple=True)[0]
            assert 'The target vehicle' in tokenizer.decode([tokenized_sample['input_ids'][i] for i in non_masked_ids[:5]])
        else:
            all_masked_ids = torch.nonzero(collator_output['labels'].squeeze(0)==-100,  as_tuple=True)[0]
            assert 'The target vehicle' in tokenizer.decode(tokenized_sample['input_ids'][max(all_masked_ids)+1:max(all_masked_ids)+10])


    #--- Setting up trainer

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=my_args.block_size, # Max number of input tokens given to the LLM
        tokenizer=tokenizer,
        # formatting_func = create_prompt, # this will aplly the create_prompt mapping to all training and test dataset
        args=args,
        train_dataset=my_TRAIN_dataset,
        eval_dataset=my_VAL_dataset,
        data_collator=collator,  # to use DataCollatorForCompletionOnlyLM
        packing=False,  # to use DataCollatorForCompletionOnlyLM
    )


    #--- Train

    if my_args.resume_from_checkpoint:
        print(f"\nFT from existing checkpoint in {output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"\nFT from base model {my_args.HF_name}")
        trainer.train()

    return None


if __name__ == "__main__":

    my_args = get_args()
    main(my_args)