# preprocessing.py

import re
import pandas as pd


def parse_height(height_str):
    if pd.isnull(height_str):
        return 177
    pattern = r"(?:(\d+)ft)?\s*(\d+)in"
    match = re.match(pattern, height_str)
    if match:
        feet = int(match.group(1)) if match.group(1) else 0
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches * 2.54
    else:
        return 177


def parse_reach(reach_str):
    if pd.isnull(reach_str):
        return 183
    pattern = r"(\d+)in"
    match = re.match(pattern, reach_str)
    if match:
        inches = int(match.group(1))
        return inches * 2.54
    else:
        return 183


def parse_strike(strike_str):
    if pd.isnull(strike_str):
        return [0, 0]
    pattern = r"(\d+)\s+of\s+(\d+)"
    match = re.match(pattern, strike_str)
    if match:
        landed = int(match.group(1))
        attempted = int(match.group(2))
        return [landed, attempted]
    else:
        return [0, 0]


def parse_control_time(time_str):
    if pd.isnull(time_str):
        return 0
    pattern = r"(\d+):(\d+)"
    match = re.match(pattern, time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds
    else:
        return 0


def parse_time(time_str):
    if pd.isnull(time_str):
        return 0
    pattern = r"(\d+):(\d+)"
    match = re.match(pattern, time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds
    else:
        return 0


def calculate_fight_time(round_num, time_str):
    if pd.isnull(round_num) or pd.isnull(time_str):
        return 0
    round_num = int(round_num)
    time_in_current_round = parse_time(time_str)
    total_fight_time = (round_num - 1) * 300 + time_in_current_round
    return total_fight_time


def process_landed_attempted(fight_data, column_name):
    landed_attempted = fight_data[column_name].apply(parse_strike)
    landed_col = column_name + "_Landed"
    attempted_col = column_name + "_Attempted"
    fight_data[landed_col] = [x[0] for x in landed_attempted]
    fight_data[attempted_col] = [x[1] for x in landed_attempted]


def is_finish(method_str):
    if pd.isnull(method_str):
        return False
    return "Decision" not in method_str
