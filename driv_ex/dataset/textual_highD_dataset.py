# Copyright (c) 2026 Valeo. All rights reserved.
import json
import os, sys
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from driv_ex.dataset import DATA_DIR


def towards_llama3_template(one_text):
    new_text = one_text.replace("<s>[INST] <<SYS>>Role: ", "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
    new_text = new_text.replace("<</SYS>>\n\n", "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
    new_text = new_text.replace(" [/INST]", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    new_text = new_text.replace(" </s>", "<|eot_id|>")
    return new_text


def towards_qwen_template(one_text):
    new_text = one_text.replace("<s>[INST] <<SYS>>Role: ", "<|im_start|>system\n")
    new_text = new_text.replace("<</SYS>>\n\n", "<|im_end|>\n<|im_start|>user\n")
    new_text = new_text.replace(" [/INST]", "<|im_end|>\n<|im_start|>assistant\n")
    new_text = new_text.replace(" </s>", "<|im_end|>")
    return new_text


def towards_non_llama3_template(one_text):
    new_text = one_text.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<s>[INST] <<SYS>>Role: ")
    new_text = new_text.replace("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "<</SYS>>\n\n", )
    new_text = new_text.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", " [/INST]", )
    new_text = new_text.replace("<|eot_id|>", " </s>")
    return new_text


class textual_highD_dataset(Dataset):
    def __init__(
        self,
        HF_name,
        bias_type=None,
        input_prompt_only=True,
        with_reverse_bias=False,
        with_labels=True,
        split_X_and_Y=False,
        fix_ego_car=True,
        split="val",
        TEST_ONE_DATA=False
    ):

        if split_X_and_Y and input_prompt_only:
            raise Exception("Cant use 'split_X_and_Y' and 'input_prompt_only' together")

        self.with_labels = with_labels
        self.split_X_and_Y = split_X_and_Y
        self.HF_name = HF_name
        self.fix_ego_car=fix_ego_car

        if not TEST_ONE_DATA:
            # Get filename
            if bias_type is not None:
                bias_str = f"_{bias_type}"
                if with_reverse_bias:
                    bias_str = bias_str.replace("_bias", "_reverse_bias")
            else:
                bias_str = ""

            input_str = "_input_only" if input_prompt_only else ""
            filename = f"llama_{split}_surrounding_thought{bias_str}{input_str}_with_labels.json"
            print(f"loading data from {filename}")

            # Get path
            full_fp = DATA_DIR / "highD_by_lcllm" / filename

            with open(full_fp) as f:
                self.dataset = json.load(f)

        elif TEST_ONE_DATA:
            self.dataset = [
                {
                'text': '<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert driving prediction model of an autonomous driving system, that can predict the future driving intention and future 4-second driving trajectory for a given target vehicle, avoiding collision with other vehicles and obstacles on the road.\nContext: \n- Coordinates: Y-axis is perpendicular, and X-axis is parallel to the direction target vehicle is facing. target vehicle\'s current position is (0,0). Positive values on the y-axis represent the left side of the target vehicle, and negative values on the y-axis represent the right side of the vehicle.\nOutput: \n- Thought:\n  - Notable features\n  - Potential behaviors\n- Final Answer:\n  - Intention:\n  - 0: Keep lane; 1: Left lane change; 2: Right lane change. The final answer should be one of the three modes.\n  - Trajectory (MOST IMPORTANT): 4 points, one every 1 second\n  - [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]\n<|start_header_id|>user<|end_header_id|>\n\nThe target vehicle is driving on a three-lane highway, in the right lane.\nThe information about the target vehicle is as follows:\n  - Velocity (km/h): vx=90.22, vy=3.02\n  - Acceleration: ax=1.15, ay=-1.15\n  - Type: truck, with width of 2.50 m and length of 20.21 m\n  - Historical position of the last 2 seconds (One point every 0.4s): [(-48.54,-1.28), (-39.72,-1.10), (-29.86,-0.87), (-19.93,-0.62), (-9.97,-0.34), (0.0,0.0)]\n\nThe information about its surrounding vehicles (within a range of 200 m) is listed as follows:\n  - Front side: a truck traveling at 82.62 km/h of X-axis, with a distance of 28 m.\n  - Left front: a car traveling at 116.93 km/h of X-axis, with a distance of 31 m.\n<|start_header_id|>assistant<|end_header_id|>\n\n\n Thought:\n  - Notable features: vy=3.02\n  - Notable features: ax=1.15\n  - Notable feature: Front side is block.\n  - Notable feature: Left front is free.\n  - Notable feature: right lane\n  - Potential behavior: Change to the left lane for overtaking.\nFinal Answer:\n  - Intention: "1: Left lane change"\n  - Trajectory: "[(25.22,0.86), (50.77,1.54), (76.60,1.98), (102.64,2.22)]"',
                 'label':1
                }
            ]
        print(f"dataset's length={len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        info_dict = self.dataset[idx] # dict with keys: 'text', 'label'
        text = info_dict['text']
        if "Llama-3" in self.HF_name:
            text = towards_llama3_template(text)
        elif "Qwen2.5" in self.HF_name:
            text = towards_qwen_template(text)

        if self.split_X_and_Y: # in that case text contains both X and Y
            if "Llama-3" in self.HF_name:
                response_template_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            elif "Llama-2" in self.HF_name or "Mistral-7B" in self.HF_name:
                response_template_string = "[/INST]"
            elif "Qwen2.5" in self.HF_name:
                response_template_string = "<|im_end|>\n<|im_start|>assistant\n"
            Y_start_id = text.find(response_template_string) + len(response_template_string)
            X = text[:Y_start_id]
            Y = text[Y_start_id:]

            if not self.fix_ego_car:
                if "Llama-3" in self.HF_name:
                    X_sys_end_template = "<|start_header_id|>user<|end_header_id|>\n\n"
                elif "Llama-2" in self.HF_name or "Mistral-7B" in self.HF_name:
                    X_sys_end_template = "<</SYS>>\n\n"
                elif "Qwen2.5" in self.HF_name:
                    X_sys_end_template = "<|im_end|>\n<|im_start|>user\n"
            else:
                X_sys_end_template = " its surrounding vehicles (within a range of 200 m) is listed as follows:\n"
            X_vision_start_id = X.find(X_sys_end_template) + len(X_sys_end_template)
            X_sys = X[:X_vision_start_id]
            X_vision = X[X_vision_start_id:]

        if self.with_labels:
            if self.split_X_and_Y:
                return (X_sys, X_vision, Y, info_dict['label'])
            else:
                return (text, info_dict['label'])
        else:
            if self.split_X_and_Y:
                return X_sys, X_vision, Y
            else:
                return text
