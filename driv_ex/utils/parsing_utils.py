# Copyright (c) 2026 Valeo. All rights reserved.
"""Structured extraction from LLM-formatted driving scene descriptions: vehicle data
parsing (kinematics, type, surrounding vehicles), coordinate helpers, and
driving-intention label extraction from generated strings."""
import ast
# import contextlib
import re
import math



# --- Constants ---
KMH_TO_MPS = 1 / 3.6
LANE_WIDTH = 3.75  # meters

VEHICLE_DIMS = {
#     "Car": {"length": 4.86, "width": 1.94, "color": "#1f77b4"},
    "car": {"length": 4.86, "width": 1.94, "color": "#1f77b4"},
#     "Truck": {"length": 14.30, "width": 2.50, "color": "#ff7f0e"},
    "truck": {"length": 14.30, "width": 2.50, "color": "#ff7f0e"},
}


# def parse_scene(scene_str) -> Dict[str, Any]:
#     """
#     Parses the full scene string into a structured dictionary.

#     Raises:
#         ValueError: If a required pattern is not found in the string.
#     """
#     data = {}
#     try:
#         # --- Target Vehicle Parsing ---
#         vel_match = re.search(r"Velocity \(km/h\): vx=([\d.-]+), vy=([\d.-]+)", scene_str)
#         acc_match = re.search(r"Acceleration: ax=([\d.-]+), ay=([\d.-]+)", scene_str)
#         type_match = re.search(r"Type: (car|rig), with width of ([\d.]+) m and length of ([\d.]+) m", scene_str)
#         hist_match = re.search(r"Historical position.*?(\[.*?\])", scene_str, re.DOTALL)
#         intent_match = re.search(r"Intention: \"(\d): (.*?)\"", scene_str)
#         traj_match = re.search(r"Trajectory: \"(\[.*?\])\"", scene_str, re.DOTALL)

#         data['target'] = {
#             'velocity': {'vx': float(vel_match.group(1)), 'vy': float(vel_match.group(2))},
#             'acceleration': {'ax': float(acc_match.group(1)), 'ay': float(acc_match.group(2))},
#             'type': type_match.group(1),
#             'dimensions': {'width': float(type_match.group(2)), 'length': float(type_match.group(3))},
#             'history': ast.literal_eval(hist_match.group(1)),
#             'intention': {'id': int(intent_match.group(1)), 'text': intent_match.group(2)},
#             'trajectory': ast.literal_eval(traj_match.group(1))
#         }

#         # --- Surrounding Vehicles Parsing ---
#         data['surrounding'] = []
#         surr_pattern = r"- (.*?): a (car|rig) traveling at ([\d.]+) km/h.*?with a distance of ([\d.]+) m"
#         matches = re.findall(surr_pattern, scene_str)

#         for match in matches:
#             pos_str, v_type, v_speed, v_dist = match
#             pos_str = pos_str.strip()
#             angle = POSITION_TO_ANGLE[pos_str]
#             dist_m = float(v_dist)

#             data['surrounding'].append({
#                 'type': v_type,
#                 'velocity': {'vx': float(v_speed), 'vy': 0.0}, # Assuming vy=0
#                 'dimensions': {'width': VEHICLE_DIMS[v_type][0], 'length': VEHICLE_DIMS[v_type][1]},
#                 'position': {
#                     'x': dist_m * math.cos(angle),
#                     'y': dist_m * math.sin(angle)
#                 }
#             })

#     except (AttributeError, KeyError, SyntaxError) as e:
#         raise ValueError(f"Failed to parse scene string. Error: {e}")

#     return data


def parse_vehicle_data_to_count_bias_changes(text_data, tokenizer):

    result = {
        "sv_pos_1": [],
        "sv_pos_2": [],
        "sv_type": [],
        "sv_speed": [],
        "sv_speed_ints": [],
        "sv_speed_decimals": [],
        "sv_dist": []
        }

    # remove x_sys if provided
    x_vision_pattern = "The target vehicle is driving"
    if x_vision_pattern in text_data:
        start_idx = text_data.find(x_vision_pattern)
        text = text_data[start_idx:]
    else:
        text = text_data

    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()

        if "Velocity (km/h): vx=" in line:
            ego_vx = line.split("(km/h): vx")[-1].split(", vy")[0]
            # result["ego_vx"] = ego_vx
            result["ego_vx_sign"] = "=-" if ego_vx[1] == "-" else "="
            result["ego_vx_int"] = ego_vx.split(".")[0].strip("=").strip("-")
            result["ego_vx_decimal"] = ego_vx.split(".")[-1].strip("=").strip("-")
            ego_vy = line.split(", vy")[-1]
            # result["ego_vy"] = ego_vy
            result["ego_vy_sign"] = "=-" if ego_vy[1] == "-" else "="
            result["ego_vy_int"] = ego_vy.split(".")[0].strip("=").strip("-")
            result["ego_vy_decimal"] = ego_vy.split(".")[-1].strip("=").strip("-")

        elif "Acceleration: ax=" in line:
            ego_ax = line.split("Acceleration: ax")[-1].split(", ay")[0]
            # result["ego_ax"] = ego_ax
            result["ego_ax_sign"] = "=-" if ego_ax[1] == "-" else "="
            result["ego_ax_int"]= ego_ax.split(".")[0].strip("=").strip("-")
            result["ego_ax_decimal"]= ego_ax.split(".")[-1].strip("=").strip("-")

            ego_ay = line.split(", ay")[-1]
            # result["ego_ay"] = ego_ay
            result["ego_ay_sign"] = "=-" if ego_ay[1] == "-" else "="
            result["ego_ay_int"] = ego_ay.split(".")[0].strip("=").strip("-")
            result["ego_ay_decimal"] = ego_ay.split(".")[-1].strip("=").strip("-")

        elif "Type:" in line and "with width of" in line:
            ego_type = line.split("Type:")[-1].split(", with width of")[0].strip()
            result["ego_type"] = ego_type
            ego_width = line.split(", with width of ")[-1].split(" m and length of ")[0].strip()
            # result["ego_width"] = ego_width
            result["ego_width_int"] = ego_width.split(".")[0]
            result["ego_width_decimal"] = ego_width.split(".")[-1]

            ego_length = line.split(" m and length of ")[-1].split(" m")[0].strip()
            # result["ego_length"] = ego_length
            result["ego_length_int"] = ego_length.split(".")[0]
            result["ego_length_decimal"] = ego_length.split(".")[-1]

        elif "of X-axis, " in line:
            sv_pos = line.split(": a ")[0].replace("- ", "").strip()
            result["sv_pos_1"].append(sv_pos.split(" ")[0])
            result["sv_pos_2"].append(sv_pos.split(" ")[-1])

            sv_type = line.split(" traveling at ")[0].split(": a ")[-1].strip()
            result["sv_type"].append(sv_type)

            sv_speed = line.split("traveling at ")[-1].split(" km/h of X-axis,")[0].strip()
            assert "." in sv_speed
            result["sv_speed"].append(sv_speed)
            sv_speed_int = sv_speed.split(".")[0]
            result["sv_speed_ints"].append(sv_speed_int)
            sv_speed_decimals = sv_speed.split(".")[-1]
            result["sv_speed_decimals"].append(sv_speed_decimals)

            sv_dist = line.split("with a distance of ")[-1].split(" m")[0].strip()
            result["sv_dist"].append(sv_dist)

    # tokenize text
    text_tokens = tokenizer.tokenize(text)
    text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)

    return result, text_token_ids

# --- Data Parsing Function ---
def parse_vehicle_data(text_data: str) -> dict:
    """Parses the raw text, including future trajectory, and handles complex formats."""

    # remove x_sys if provided
    x_vision_pattern = "The target vehicle is driving"
    if x_vision_pattern in text_data:
        start_idx = text_data.find(x_vision_pattern)
        text = text_data[start_idx:]
    else:
        text = text_data

    data = {
        "road_info": {"num_lanes": 4, "ego_lane": "right lane"}, # Default
        "ego_vehicle": {"velocity": {}, "acceleration":{}, "dimensions": {}, "history": [], "intention":{}, "trajectory": []},
        "surrounding_vehicles": [],
    }

    # surrounding_pattern = re.compile(
    #     r"-\s*(.*?):\s*a ( car|car|truck) traveling at ([\d.]+) km/h .* with a distance of ([\d]+) m"
    # )

    surrounding_pattern = re.compile(
        r"-\s*(.*?):\s*a (car|truck) traveling at ([\d.]+) km/h .* with a distance of ([\d]+) m"
    )

    lines = text.strip().splitlines()
    if not lines: return data

    road_match = re.search(
        r"on a (two|three|four)-lane highway, in the (middle|right|left) lane",
        lines[0],
    )
    if road_match:
        lanes_word, lane_loc = road_match.groups()
        data["road_info"]["num_lanes"] = {"two": 2, "three": 3, "four": 4}[lanes_word]
        data["road_info"]["ego_lane"] = lane_loc + " lane"

    for line in lines:
        line = line.strip()

        if "Velocity" in line:
            match = re.search(r"vx=([\d\.-]+), vy=([\d\.-]+)", line)
            if match: data["ego_vehicle"]["velocity"] = {"vx": float(match.group(1)), "vy": float(match.group(2))}

        elif "Acceleration" in line:
            match = re.search(r"Acceleration: ax=([\d.-]+), ay=([\d.-]+)", line)
            if match: data["ego_vehicle"]["acceleration"] = {'ax': float(match.group(1)), 'ay': float(match.group(2))}

        elif "Type:" in line and "width" in line:
            match = re.search(r"Type: (car|truck), with width of ([\d.]+) m and length of ([\d.]+) m", line)
            # match = re.search(r"Type: ( car|car|truck), with width of ([\d.]+) m and length of ([\d.]+) m", line)

            if match:
                data["ego_vehicle"]["dimensions"] = {"width": float(match.group(2)), "length": float(match.group(3))}
                data["ego_vehicle"]["type"] = match.group(1).strip().lower()

        elif "Historical position" in line:
            match = re.search(r":\s*(\[.*\])", line)
            if match: data["ego_vehicle"]["history"] = ast.literal_eval(match.group(1).strip())

        elif "Intention" in line:
            match = re.search(r"Intention: \"(\d): (.*?)\"", line)
            if match: data["ego_vehicle"]["intention"] = {'id': int(match.group(1)), 'text': match.group(2)}

        elif "Trajectory:" in line:
            match = re.search(r'Trajectory:\s*"(\[.*\])"', line)
            if match: data["ego_vehicle"]["trajectory"] = ast.literal_eval(match.group(1))

        surr_match = surrounding_pattern.search(line)
        if surr_match:
            pos_str, v_type, speed, dist = surr_match.groups()
            processed_pos_key = pos_str.strip().lower().replace(" ", "_")
            sv={
                "position_key": processed_pos_key,
                "type": v_type.strip(),
                "speed": float(speed),
                "distance": int(dist)
            }
            sv_initial_x_abs, sv_initial_y_abs, sv_status = get_sv_coords(
                sv, ego_lane_str=data["road_info"]["ego_lane"]
                )
            sv["x"]=sv_initial_x_abs
            sv["y"]=sv_initial_y_abs
            sv["status"]=sv_status
            data["surrounding_vehicles"].append(sv)
    return data


# Parsing Output will be of form: {
    # 'road_info': {'num_lanes': 4, 'ego_lane': 'right lane'},
#  'ego_vehicle':
#   {
    # 'velocity': {'vx': 78.52, 'vy': 2.05},
#   'dimensions': {'width': 2.02, 'length': 4.65},
#   'history': [(-41.3, -0.79),
#    (-33.92, -0.71),
#    (-25.71, -0.58),
#    (-17.37, -0.41),
#    (-8.79, -0.22),
#    (0.0, 0.0)],
#   'trajectory': [(22.07, 0.59), (44.45, 1.08), (66.97, 1.42), (89.56, 1.61)],
#   'type': 'car'
# },

#  'surrounding_vehicles':
# [{'position_key': 'left_front',
#    'type': 'car',
#    'speed': 85.43,
#    'distance': 103,
#   'x': x_coord, 'y': y_coord, 'status': "in_bound"},
#   {'position_key': 'left_rear',
#    'type': 'car',
#    'speed': 91.84,
#    'distance': 60,
#   'x': x_coord, 'y': y_coord, 'status': "in_bound"}
# ]
# }


def get_sv_coords(sv, ego_lane_str, initial_ego_y=0.0):

    pos_key = sv["position_key"]
    sv_initial_y_abs = initial_ego_y
    sv_status = "in_bound"
    # check if certain vehicles are 'out of bound'
    if "left" in pos_key:
        sv_initial_y_abs += LANE_WIDTH
        if ego_lane_str == "left lane":
            sv_status = f'{pos_key} {sv["type"]} is out of the highway'
    elif "right" in pos_key:
        sv_initial_y_abs -= LANE_WIDTH
        if ego_lane_str == "right lane":
            sv_status = f'{pos_key} {sv["type"]} is out of the highway'
    if pos_key in ["left_side", "right_side"] and sv['distance']>25:
        sv_status = f'{pos_key} {sv["type"]} is too far from ego ({sv["distance"]}m)'

    # sv_initial_x_abs = sv['distance'] * (1 if "rear" not in pos_key and "back" not in pos_key else -1)
    # we use the fact that ego's x coord is 0 at t=0
    try:
        sv_initial_x_abs = math.sqrt(sv['distance']**2 - (sv_initial_y_abs-initial_ego_y)**2) * (1 if "rear" not in pos_key and "back" not in pos_key else -1)
    except:
        if sv['distance']<LANE_WIDTH and pos_key in ["right_side", "left_side"]:
            sv_initial_x_abs=0
            sv_initial_y_abs=sv['distance'] * (1 if pos_key=="left_side" else -1)+initial_ego_y

    return sv_initial_x_abs, sv_initial_y_abs, sv_status



def get_class_from_string(string):
    """Extract the driving-intention label (0/1/2) from the tail of a generated string."""
    if "Intention" in string[-15:] and "0" in string[-3:]:
        label = 0
    elif "Intention" in string[-15:] and "1" in string[-3:]:
        label = 1
    elif "Intention" in string[-15:] and "2" in string[-3:]:
        label = 2
    else:
        label = None
    return label


def get_ego_pos(scene_data, ref_wrt_ego=False):
    road_info = scene_data["road_info"]
    num_lanes = road_info["num_lanes"]
    ego_lane_str = road_info["ego_lane"]
    lane_centers_y = [(i - (num_lanes - 1) / 2.0) * LANE_WIDTH for i in range(num_lanes)]
    initial_ego_y = 0
    if not ref_wrt_ego:
        if ego_lane_str == "left lane": initial_ego_y = lane_centers_y[-1]
        elif ego_lane_str == "right lane": initial_ego_y = lane_centers_y[0]
        elif ego_lane_str == "middle lane": initial_ego_y = lane_centers_y[1]

    return num_lanes, ego_lane_str, initial_ego_y