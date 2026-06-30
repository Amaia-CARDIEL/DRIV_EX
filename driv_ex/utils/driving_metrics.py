# Copyright (c) 2026 Valeo. All rights reserved.
"""Driving-domain evaluation: physics-based maneuver simulation, template fitness
validation, trajectory RMSE, label/trajectory coherence, and fluency
token-probability scoring."""
import re
import math
import ast
import numpy as np
import torch
from driv_ex.utils.parsing_utils import parse_vehicle_data

# Conversion factor for kilometers per hour to meters per second
KMH_TO_MPS = 1 / 3.6

LANE_WIDTH = 3.75  # meters

VEHICLE_DIMS = {
    "car": (1.94, 4.86),
    "truck": (2.50, 14.30)
}


def get_max_ego_dist(num_lanes):
    assert num_lanes in [2,3,4]
    return (num_lanes-1)*LANE_WIDTH




# ---------------- Crash metric ----------------


def evaluate_maneuver(scene_str, simulation_duration=4.0) -> str:
    """
    Evaluates the target vehicle's future maneuver based on the scene description.

    Args:
        scene_str: The string containing the full driving scene information.

    Returns:
        A string classification: "collision", "discomfort", or "good".
    """

    # try:
    #     scene_data = parse_vehicle_data(scene_str)
    # except ValueError as e:
    #     return f"Error parsing input: {e}"
    scene_data = parse_vehicle_data(scene_str)

    target = scene_data['ego_vehicle']
    surrounding = scene_data['surrounding_vehicles']

    # --- 1. Collision Check ---
    # Simulate the next 4 seconds at discrete time steps
    time_step = 0.2  # seconds
    # simulation_duration = 4.0 => now its an arg

    # Prepend the starting position to the future trajectory for easier interpolation
    full_trajectory = [(0.0, 0.0)] + target['trajectory']

    for t in [i * time_step for i in range(int(simulation_duration / time_step) + 1)]:
        # Interpolate target vehicle's position at time t
        time_index = int(t)
        if time_index >= len(full_trajectory) - 1:
            # Handle the very end of the simulation
            time_index = len(full_trajectory) - 2
            interp_ratio = 1.0
        else:
            interp_ratio = (t - time_index)

        # Linear interpolation between trajectory points
        prev_pt = full_trajectory[time_index]
        next_pt = full_trajectory[time_index + 1]
        target_x = prev_pt[0] + (next_pt[0] - prev_pt[0]) * interp_ratio
        target_y = prev_pt[1] + (next_pt[1] - prev_pt[1]) * interp_ratio

        # Check against each surrounding vehicle
        for sv in surrounding:
            # Predict surrounding vehicle's future position (constant velocity)
            sv_vx_mps = sv['speed'] * KMH_TO_MPS
            sv_x = sv['x'] + sv_vx_mps * t
            sv_y = sv['y'] # Assuming constant lateral position

            # Simple Axis-Aligned Bounding Box (AABB) collision check
            # This is a good approximation for driving scenarios
            dist_x = abs(target_x - sv_x)
            dist_y = abs(target_y - sv_y)

            sv_width, sv_length = VEHICLE_DIMS[sv['type']]
            min_safe_dist_x = (target['dimensions']['length'] + sv_length) / 2.0
            min_safe_dist_y = (target['dimensions']['width'] + sv_width) / 2.0

            if dist_x < min_safe_dist_x and dist_y < min_safe_dist_y:
                return "collision"

    # --- 2. Discomfort Check ---
    # Check if the vehicle's current movement contradicts its future intention.

    # Check 1: Recent historical trend vs. intention
    # Positive y_trend means moving left, negative means moving right.
    y_trend = target['history'][-1][1] - target['history'][0][1]

    # Discomfort if recently moving left but intends to change right
    if y_trend > 0.2 and target['intention']['id'] == 2: # 2: Right lane change
        return "discomfort"

    # Discomfort if recently moving right but intends to change left
    if y_trend < -0.2 and target['intention']['id'] == 1: # 1: Left lane change
        return "discomfort"

    # Check 2: Initial lateral velocity vs. intention
    initial_vy_mps = target['velocity']['vy'] * KMH_TO_MPS

    # Discomfort if initial velocity is leftward but intends right change
    if initial_vy_mps > 0.3 and target['intention']['id'] == 2:
        return "discomfort"

    # Discomfort if initial velocity is rightward but intends left change
    if initial_vy_mps < -0.3 and target['intention']['id'] == 1:
        return "discomfort"

    # --- 3. Good Maneuver ---
    # If no collision or discomfort is detected, the maneuver is good.
    return "good"


def evaluate_maneuver_multi_sec(full_scenario):
    result = {}
    for sec in [1.0, 2.0, 3.0, 4.0]:
        result[f"{str(int(sec))}_sec"] = evaluate_maneuver(
            full_scenario, simulation_duration=sec
            )
    return result

def is_lc_label_and_traj_coherent(text, ego_car_width = 1.94, lane_width = 3.75, return_lc_label = False):
    # example of type of text input
    # text= '''Final Answer:
    #   - Intention: "1: Left lane change"
    #   - Trajectory: "[(10.1,20.2), (30.3,40.4), (50.5,60.6), (70.7,80.8)]"
    # '''

    # focus on end of generation in case full scene / text given
    x_vision_pattern = "The target vehicle is driving"
    if x_vision_pattern in text:
        start_idx = text.find(x_vision_pattern)
        text = text[start_idx:]

    fa_pattern = "Final Answer:"
    if fa_pattern in text:
        start_idx = text.find(fa_pattern)
        text = text[start_idx:]

    # extract a list of 4 y coordinates
    pattern = r'\(\s*([-]?\d+\.?\d*)\s*,\s*([-]?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, text)
    y_coordinates = [float(match[1]) for match in matches]

    y_drift_lower_bound_for_LC = (lane_width-ego_car_width)/2 # default = 0.905 (m)

    # compute y_drift wrt last y coord value
    coherent = False
    if y_coordinates[-1] > y_drift_lower_bound_for_LC: # should be left LC
        if "1: Left lane change" in text:
            coherent = True

    elif y_coordinates[-1] < (-1)*y_drift_lower_bound_for_LC: # should be right LC
        if "2: Right lane change" in text:
            coherent = True

    else: # should be keep lane
        if "0: Keep lane" in text:
            coherent = True

    if return_lc_label:
        if "0: Keep lane" in text:
            label = 0
        elif "1: Left lane change" in text:
            label = 1
        elif "2: Right lane change" in text:
            label = 2
        else:
            label = None
        return coherent, label

    else:
        return coherent




def get_label_from_full_scenario(text):

    # focus on end of generation in case full scene / text given
    x_vision_pattern = "The target vehicle is driving"
    if x_vision_pattern in text:
        start_idx = text.find(x_vision_pattern)
        text = text[start_idx:]

    fa_pattern = "Final Answer:"
    if fa_pattern in text:
        start_idx = text.find(fa_pattern)
        text = text[start_idx:]

    if "0: Keep lane" in text:
        label = 0
    elif "1: Left lane change" in text:
        label = 1
    elif "2: Right lane change" in text:
        label = 2
    else:
        label=None
    return label



#----------------- check if template is respected


def check_template_fitness(args, input_str, hard_control=True, return_details=False):
    """
    Strictly check whether values in `input_str` match the template variable domains.
    Any corrupted, malformed, or out-of-range value is marked invalid.
    Returns the percentage of valid values.
    """
    # if "Llama-3" not in args.HF_name:
    #     raise Exception("for an llm other than llama3, check the tokenization'impact on patterns, candidate_str_values, etc")

    candidate_str_values = {
        # ego tokens
        "nb_lanes": [" two-lane", " three-lane", " four-lane"],
        "ego_lane_pos": [" middle", " right", " left"],
        "ego_type": [" car", " truck"],
        "ego_vx": (0.00, 161.46),
        "ego_vy": (-6.55, 8.21),
        "ego_ax": (-16.92, 9.11),
        "ego_ay": (-9.11, 16.92),
        "ego_width": (1.52, 3.44),
        "ego_length": (2.43, 23.24),
        # sv tokens
        "sv_positions": [
            'Front side', 'Back side', 'Left side', 'Left front', 'Left rear',
            'Right side', 'Right front', 'Right rear'
        ],
        "sv_vehicle_types": ['car', 'truck'],
        "sv_speed": (0.00, 218.74),
        "sv_distance": (2, 199),
    }

    valid = 0
    invalid = 0
    valid_list, invalid_list = [], []

    def check_numeric(var, value_str, hard_control=True):
        """Check if numeric string lies within valid range."""
        # test presence of = for vx,vy,ax,ay then remove it
        if var in ["ego_vx", "ego_vy", "ego_ax", "ego_ay"]:
            if value_str[0]!= "=":
                return False
            else:
                value_str = value_str[1:]

        # all ints
        if var == "sv_distance":
            if hard_control:
                for char in value_str:
                    if char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        return False
            else:
                for char in value_str:
                    if char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]:
                        return False
        # all floats with 2 decimals
        else:
            for it, char in enumerate(value_str):
                if it == 0 and char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]:
                    return False
                elif it>0 and char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]:
                    return False
            if hard_control:
                if len(value_str.split(".")[-1]) != 2 or len(value_str.split(".")[0]) ==0:
                    return False

        # ensure it's a clean number (rejects letters, symbols, multiple dots, etc.)
        try:
            if not re.fullmatch(r"-?\d+(\.\d+)?", value_str.strip()):
                return False
            val = float(value_str)
            if hard_control:
                lo, hi = candidate_str_values[var]
            else:
                lo, hi = -250, 250
            return lo <= val <= hi
        except Exception:
            return False

    # --- Extract and check ego-level variables ---
    # i replace (.*?) that is lazy (match as few as possible) and does not capture '\n' by ([\s\S]*) (greedy and captures everything)
    patterns = {
        "nb_lanes": r" on a([\s\S]*) highway",
        "ego_lane_pos": r" in the([\s\S]*) lane.",
        "ego_type": r"Type:([\s\S]*), with width",
        "ego_vx": r": vx([\s\S]*), vy",
        "ego_vy": r", vy([\s\S]*)\n  - Acceleration",
        "ego_ax": r": ax([\s\S]*), ay",
        "ego_ay": r", ay([\s\S]*)\n  - Type:",
        "ego_width": r" with width of ([\s\S]*) m and length",
        "ego_length": r" and length of ([\s\S]*) m\n  - Historical",
    }

    if args.optim_part != "sv":
        for var, pat in patterns.items():
            try:
                m = re.search(pat, input_str)
            except Exception as e:
                print(f"Regex error for variable '{var}' with pattern '{pat}': {e}")
                print(f"(issue related to input_str of type {type(input_str)})") #, printed below)\n", input_str)
                m=False
            if not m:
                print(f"WARNING: failure to match pattern '{pat}' in '{input_str}'")
                continue
            val = m.group(1) #.strip()
            # Categorical
            if isinstance(candidate_str_values[var], list):
                if val in candidate_str_values[var]:
                    valid += 1
                    valid_list.append((var, val))
                else:
                    invalid += 1
                    invalid_list.append((var, val))
            # Numeric
            else:
                if check_numeric(var, val, hard_control=hard_control):
                    valid += 1
                    valid_list.append((var, val))
                else:
                    invalid += 1
                    invalid_list.append((var, val))
        total_ego = valid+invalid
        if total_ego !=9:
            print(f"WARNING: valid + invalid != 9 when parsing '{input_str}'")
        assert total_ego == 9
    else:
        total_ego = 0

    # --- Extract and check surrounding vehicles ---
    sv_start_pattern = "The information about its surrounding vehicles"
    sv_start_id = input_str.find(sv_start_pattern)
    sv_str = input_str[sv_start_id:]

    if args.optim_part != "ego":
        # check if sv vehicles are described at all
        sv_exists = True
        if "- No surrounding vehicles within 200m." in sv_str:
            sv_exists=False
        elif "(within a range of 200 m) is listed as follows:\n [/INST]" in sv_str:
            sv_exists=False
        elif "(within a range of 200 m) is listed as follows:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" in sv_str:
            sv_exists=False
        elif "(within a range of 200 m) is listed as follows:\n<|start_header_id|>assistant<|end_header_id|>\n\n" in sv_str:
            sv_exists=False

        # if some are described, process them
        if sv_exists:
            # sv_pattern = r"  -([\s\S]*): a([\s\S]*) traveling at ([\s\S]*) km/h of X-axis, with a distance of ([\s\S]*) m."
            sv_pattern = r"^\s+-\s*(.*?): a (.*?) traveling at (.*?) km/h of X-axis, with a distance of (.*?) m\."
            sv_matches = re.findall(sv_pattern, sv_str, re.MULTILINE)

        if sv_matches:
            for pos, veh_type, speed, dist in sv_matches:
                # position
                if pos in candidate_str_values["sv_positions"]:
                    valid += 1
                    valid_list.append(("sv_pos", pos))
                else:
                    invalid += 1
                    invalid_list.append(("sv_pos", pos))
                # vehicle type
                if veh_type in candidate_str_values["sv_vehicle_types"]:
                    valid += 1
                    valid_list.append(("sv_type", veh_type))
                else:
                    invalid += 1
                    invalid_list.append(("sv_type", veh_type))
                # speed
                if check_numeric("sv_speed", speed, hard_control=hard_control):
                    valid += 1
                    valid_list.append(("sv_speed", speed))
                else:
                    invalid += 1
                    invalid_list.append(("sv_speed", speed))
                # distance
                if check_numeric("sv_distance", dist, hard_control=hard_control):
                    valid += 1
                    valid_list.append(("sv_dist", dist))
                else:
                    invalid += 1
                    invalid_list.append(("sv_dist", dist))

        nb_sv_vehicles = len(sv_matches) if sv_matches else 0
        assert (valid + invalid) == (total_ego + nb_sv_vehicles*4)

    # --- Compute stats ---

    total_all = valid + invalid
    final_score = round(100 * valid / total_all, 2) # if total > 0 else 100.0
    if not return_details:
        return final_score
    else:
        return final_score, valid_list, invalid_list

def compute_trajectory_rmse(GT_list, pred_list):
    """
    Computes RMSE for longitudinal (x) and lateral (y) coordinates
    between ground truth (GT_list) and predicted coordinates (pred_list).

    Args:
        GT_list: list of (x, y) tuples
        pred_list: list of (x, y) tuples

    Returns:
        rmse_lon (float), rmse_lat (float)
    """
    if len(GT_list) != len(pred_list):
        raise ValueError("GT_list and pred_list must have the same length.")

    n = len(GT_list)

    # Accumulate squared error
    sum_sq_x = 0.0
    sum_sq_y = 0.0

    for (gt_x, gt_y), (pred_x, pred_y) in zip(GT_list, pred_list):
        sum_sq_x += (gt_x - pred_x) ** 2
        sum_sq_y += (gt_y - pred_y) ** 2

    rmse_lon = math.sqrt(sum_sq_x / n)
    rmse_lat = math.sqrt(sum_sq_y / n)

    return rmse_lon, rmse_lat


def compute_dataset_trajectory_rmse(GT_lists, pred_lists):
    """
    Computes RMSE for longitudinal (x) and lateral (y) coordinates
    between ground truth (GT_list) and predicted coordinates (pred_list).

    Args:
        GT_lists: list of lists of (x, y) tuples
        pred_lists: list of lists of (x, y) tuples

    Returns:
        rmse_lon (float), rmse_lat (float)
    """
    if len(GT_lists) != len(pred_lists):
        print("Error with GT list's len != pred_lists")
        print("GT_lists:", GT_lists)
        print("pred_lists:", pred_lists)
        raise ValueError("GT_lists and pred_lists must have the same length.")

    n = len(GT_lists)*4

    # Accumulate squared error
    sum_sq_x = 0.0
    sum_sq_y = 0.0

    for GT_list, pred_list in zip(GT_lists, pred_lists):
        if len(GT_list) != len(pred_list):
            raise ValueError("elems of GT_lists and pred_lists must have the same length.")
        for (gt_x, gt_y), (pred_x, pred_y) in zip(GT_list, pred_list):
            sum_sq_x += (gt_x - pred_x) ** 2
            sum_sq_y += (gt_y - pred_y) ** 2

    rmse_lon = math.sqrt(sum_sq_x / n)
    rmse_lat = math.sqrt(sum_sq_y / n)

    return rmse_lon, rmse_lat


def extract_trajectory(s):
    # Extract the part inside the quotes using regex
    m = re.search(r' - Trajectory: "(\[.*\])"', s)

    if not m:
        raise ValueError("No trajectory list found in input string")

    list_str = m.group(1)

    # Convert textual list to a real Python list safely
    trajectory = ast.literal_eval(list_str)

    return trajectory

#-----



@torch.no_grad()
def min_conditional_tok_prob(model, tokenizer, best_seq, learnable_targets=None, set_adapter=True, return_all_token_probs=False):
    """
    Compute the Weakest Link Likelihood (WLL) metric for a batch of sequences.

    Args:
        model: Causal LM returning logits
        best_seq: either a string or a list of string(s):
        learnable_targets: index of learnable targets (idx with respect to X_sys + X_vision with -1 shift to fit targets as defined below)

    Returns:
        min_probs: list[float] — minimum conditional probability per sequence
        min_indices: list[int] — position of the weakest token in each sequence
        all_token_probs: list[list[float]] — full conditional probs per sequence
    """
    if set_adapter:
        model.set_adapter("X_vision_gen")
    model.eval()

    # prep input format
    if isinstance(best_seq, str):
        best_seq = [best_seq]

    # test if bos at the beginning of the str, if not require the tokenizer to add it
    add_bos=False
    if tokenizer.encode(best_seq[0], add_special_tokens = False)[0] != tokenizer.bos_token_id:
        add_bos = True
        print("warning: mctp added a bos token => given learnable_targets might be shifted by 1")
    input_ids = torch.tensor(tokenizer(best_seq, add_special_tokens=add_bos).input_ids).to(model.device) # LongTensor of shape [batch, seq_len]
    if input_ids.dim()==1:
        input_ids = input_ids.unsqueeze(0)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits  # [batch, seq_len, vocab]

    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)  # [batch, seq_len, vocab]

    # Targets are next tokens (shift left)
    targets = input_ids[:, 1:]            # [batch, seq_len-1]
    pred_probs = probs[:, :-1, :]         # [batch, seq_len-1, vocab]

    # Gather probabilities of the actual next token
    token_probs = pred_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # token_probs: [batch, seq_len - 1]

    min_probs = []
    min_tok_ids = []
    min_tok_decodings = []
    all_learn_token_probs = []
    all_avg_learn_token_probs = []
    # all_token_probs = []

    for b in range(token_probs.size(0)):
        one_prob_list = token_probs[b].tolist()
        one_target_list = targets[b].tolist()
        # if return_all_token_probs:
        #     all_token_probs.append([(tokenizer.decode(one_target_list[i]), one_prob_list[i]) for i in range(len(one_prob_list))])
        assert len(one_prob_list) == len(one_target_list)

        if learnable_targets is not None:
            one_prob_list = [one_prob_list[i] for i in learnable_targets]
            one_target_list = [one_target_list[i] for i in learnable_targets]

            if return_all_token_probs:
                all_learn_tok_to_prob = [(tokenizer.decode(one_target_list[i]), one_prob_list[i]) for i in range(len(one_prob_list))]
                all_learn_token_probs.append(all_learn_tok_to_prob)
                avg_learnable_token_probs = sum(one_prob_list)/len(one_prob_list) if len(one_prob_list)>0 else 0
                all_avg_learn_token_probs.append(avg_learnable_token_probs)

        # Find weakest link
        min_prob = min(one_prob_list)
        min_probs.append(min_prob)

        # find weakest tok_id and tok_deco
        min_idx = one_prob_list.index(min_prob)
        min_tok_id = one_target_list[min_idx]
        min_tok_deco = tokenizer.decode(min_tok_id)
        min_tok_ids.append(min_tok_id)
        min_tok_decodings.append(min_tok_deco)

    result ={
        "worst_prob": min_probs,
        "worst_tok_id": min_tok_ids,
        "worst_tok_deco": min_tok_decodings
    }

    if return_all_token_probs:
        result["avg_learnable_token_probs"] = all_avg_learn_token_probs
        result["all_learnable_token_probs"] = all_learn_token_probs
        # result["all_token_probs"] = all_token_probs

    return result
