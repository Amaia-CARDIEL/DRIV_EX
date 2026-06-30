# Models

All our models are released under a research-only [DRIV-EX Model & Data License](LICENSE_MODEL_AND_DATA.txt).

## Downloading pretrained LoRA weights

The models' LoRA weights, used in our main results and based on Llama-3-8B-Instruct, Mistral-7B-Instruct and Qwen2.5-7B-Instruct, are released on our GitHub.

For all checkpoints, we used the following script to convert each set of weights to a single tar file:

```bash
tar czf checkpoint_folder_name.tar.gz checkpoint_folder_name
```

You can untar them with the simple command:

```bash
tar -xvzf filename.tar.gz
```

Once done, as mentioned in our [README.md](README.md), place the untared checkpoint folder in subfolders named as follows:
* for Driving LLMs: ``DRIV_EX/LCLLM_ckpts/classic_FT/{checkpoint_folder}``,
* for Fluency Experts: ``DRIV_EX/LCLLM_ckpts/X_vision_FT/{checkpoint_folder}``.
