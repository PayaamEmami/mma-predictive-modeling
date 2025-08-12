import re
import pandas as pd


def parse_height(height_str):
    if pd.isnull(height_str):
        return 177
    height_str = height_str.strip()
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
    reach_str = reach_str.strip()
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
    strike_str = strike_str.strip()
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
    time_str = time_str.strip()
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
    time_str = time_str.strip()
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


def compute_historical_stats(fight_data):
    """Compute per-fighter historical performance statistics in fight_data."""
    import numpy as np
    from collections import defaultdict

    # Initialize fighter statistics tracking
    fighter_stats = defaultdict(
        lambda: {
            "TotalFightTime": 0,
            "NumFights": 0,
            "Wins": 0,
            "Losses": 0,
            "Draws": 0,
            "NoContests": 0,
            "WinsByFinish": 0,
            "LastFightDate": None,
            "TotalControlTime": 0,
            "TotalSubmissionAttempts": 0,
            "TotalLegStrikes": 0,
            "TotalClinchStrikes": 0,
            "TotalStrikesLanded": 0,
            "TotalStrikesAttempted": 0,
            "TotalTakedownsLanded": 0,
            "TotalTakedownsAttempted": 0,
            "TotalReversals": 0,
        }
    )

    # Initialize historical performance columns
    for fighter_num in ["Fighter1", "Fighter2"]:
        fight_data[f"{fighter_num}_AvgFightTime"] = 0.0
        fight_data[f"{fighter_num}_TimeSinceLastFight"] = 0.0
        fight_data[f"{fighter_num}_FinishRate"] = 0.0
        fight_data[f"{fighter_num}_Wins"] = 0
        fight_data[f"{fighter_num}_Losses"] = 0
        fight_data[f"{fighter_num}_Draws"] = 0
        fight_data[f"{fighter_num}_NoContests"] = 0
        fight_data[f"{fighter_num}_AvgControlTime"] = 0.0
        fight_data[f"{fighter_num}_AvgSubmissionAttempts"] = 0.0
        fight_data[f"{fighter_num}_AvgLegStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgClinchStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgStrikesLanded"] = 0.0
        fight_data[f"{fighter_num}_AvgStrikesAttempted"] = 0.0
        fight_data[f"{fighter_num}_StrikeAccuracy"] = 0.0
        fight_data[f"{fighter_num}_AvgTakedownsLanded"] = 0.0
        fight_data[f"{fighter_num}_AvgTakedownsAttempted"] = 0.0
        fight_data[f"{fighter_num}_AvgReversals"] = 0.0

    # Sort fights chronologically
    fight_data = fight_data.sort_values("EventDate").reset_index(drop=True)

    # Walk through each fight and update stats
    for idx, row in fight_data.iterrows():
        fight_time = row["Fight_Time_sec"]
        event_date = row["EventDate"]

        # Pre-fight stats assignment
        for fighter_num in ["Fighter1", "Fighter2"]:
            fighter_id = row[f"{fighter_num}_ID"]
            stats = fighter_stats[fighter_id]
            stats_before = stats.copy()

            if stats_before["NumFights"] > 0:
                # Average fight time
                fight_data.at[idx, f"{fighter_num}_AvgFightTime"] = (
                    stats_before["TotalFightTime"] / stats_before["NumFights"]
                )
                # Time since last fight
                if stats_before["LastFightDate"] is not None:
                    days = (event_date - stats_before["LastFightDate"]).days
                    fight_data.at[idx, f"{fighter_num}_TimeSinceLastFight"] = days
                # Finish rate
                if stats_before["Wins"] > 0:
                    fight_data.at[idx, f"{fighter_num}_FinishRate"] = (
                        stats_before["WinsByFinish"] / stats_before["Wins"]
                    )
                # Average performance metrics
                fight_data.at[idx, f"{fighter_num}_AvgControlTime"] = (
                    stats_before["TotalControlTime"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgSubmissionAttempts"] = (
                    stats_before["TotalSubmissionAttempts"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgLegStrikes"] = (
                    stats_before["TotalLegStrikes"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgClinchStrikes"] = (
                    stats_before["TotalClinchStrikes"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgStrikesLanded"] = (
                    stats_before["TotalStrikesLanded"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgStrikesAttempted"] = (
                    stats_before["TotalStrikesAttempted"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_StrikeAccuracy"] = (
                    stats_before["TotalStrikesLanded"]
                    / stats_before["TotalStrikesAttempted"]
                    if stats_before["TotalStrikesAttempted"] > 0
                    else 0
                )
                fight_data.at[idx, f"{fighter_num}_AvgTakedownsLanded"] = (
                    stats_before["TotalTakedownsLanded"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgTakedownsAttempted"] = (
                    stats_before["TotalTakedownsAttempted"] / stats_before["NumFights"]
                )
                fight_data.at[idx, f"{fighter_num}_AvgReversals"] = (
                    stats_before["TotalReversals"] / stats_before["NumFights"]
                )
            # Assign win/loss/draw counts before fight
            fight_data.at[idx, f"{fighter_num}_Wins"] = stats_before["Wins"]
            fight_data.at[idx, f"{fighter_num}_Losses"] = stats_before["Losses"]
            fight_data.at[idx, f"{fighter_num}_Draws"] = stats_before["Draws"]
            fight_data.at[idx, f"{fighter_num}_NoContests"] = stats_before["NoContests"]

        # Update cumulative stats after fight
        for fighter_num in ["Fighter1", "Fighter2"]:
            fighter_id = row[f"{fighter_num}_ID"]
            stats = fighter_stats[fighter_id]
            stats["TotalFightTime"] += fight_time if not np.isnan(fight_time) else 0
            stats["NumFights"] += 1
            stats["LastFightDate"] = event_date
            stats["TotalControlTime"] += (
                row[f"{fighter_num}_Control_Time_sec"]
                if not np.isnan(row[f"{fighter_num}_Control_Time_sec"])
                else 0
            )
            sub = float(row[f"{fighter_num}_Submission_Attempts"])
            stats["TotalSubmissionAttempts"] += sub if not np.isnan(sub) else 0
            leg = row[f"{fighter_num}_Leg_Strikes_Landed"]
            stats["TotalLegStrikes"] += leg if not np.isnan(leg) else 0
            clinch = row[f"{fighter_num}_Clinch_Strikes_Landed"]
            stats["TotalClinchStrikes"] += clinch if not np.isnan(clinch) else 0
            sl = row[f"{fighter_num}_Significant_Strikes_Landed"]
            sa = row[f"{fighter_num}_Significant_Strikes_Attempted"]
            stats["TotalStrikesLanded"] += sl if not np.isnan(sl) else 0
            stats["TotalStrikesAttempted"] += sa if not np.isnan(sa) else 0
            td_l = row.get(f"{fighter_num}_Takedowns_Landed", 0)
            td_a = row.get(f"{fighter_num}_Takedowns_Attempted", 0)
            stats["TotalTakedownsLanded"] += td_l if not np.isnan(td_l) else 0
            stats["TotalTakedownsAttempted"] += td_a if not np.isnan(td_a) else 0
            rev = float(row[f"{fighter_num}_Reversals"])
            stats["TotalReversals"] += rev if not np.isnan(rev) else 0

        # Update outcome stats
        win = str(row["Winner"])
        method = str(row["Method"])
        s1 = fighter_stats[row["Fighter1_ID"]]
        s2 = fighter_stats[row["Fighter2_ID"]]
        if win == "1":
            s1["Wins"] += 1
            s2["Losses"] += 1
            if is_finish(method):
                s1["WinsByFinish"] += 1
        elif win == "2":
            s1["Losses"] += 1
            s2["Wins"] += 1
            if is_finish(method):
                s2["WinsByFinish"] += 1
        elif win == "D":
            s1["Draws"] += 1
            s2["Draws"] += 1
        elif win == "NC":
            s1["NoContests"] += 1
            s2["NoContests"] += 1
    return fight_data


def preprocess_features(features):
    """
    Preprocess features for inference.
    This function applies the same normalization/scaling that was used during training.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    import boto3
    import tempfile
    import os

    # If features is a list/array, convert to DataFrame
    if not isinstance(features, pd.DataFrame):
        # Assume it's a single sample - convert to DataFrame
        feature_names = [
            "Fighter1_Height_cm",
            "Fighter1_Reach_cm",
            "Fighter1_Age",
            "Fighter2_Height_cm",
            "Fighter2_Reach_cm",
            "Fighter2_Age",
            "Fighter1_AvgFightTime",
            "Fighter1_TimeSinceLastFight",
            "Fighter1_FinishRate",
            "Fighter1_Wins",
            "Fighter1_Losses",
            "Fighter1_Draws",
            "Fighter1_NoContests",
            "Fighter1_AvgControlTime",
            "Fighter1_AvgSubmissionAttempts",
            "Fighter1_AvgLegStrikes",
            "Fighter1_AvgClinchStrikes",
            "Fighter1_AvgStrikesLanded",
            "Fighter1_AvgStrikesAttempted",
            "Fighter1_StrikeAccuracy",
            "Fighter1_AvgTakedownsLanded",
            "Fighter1_AvgTakedownsAttempted",
            "Fighter1_AvgReversals",
            "Fighter2_AvgFightTime",
            "Fighter2_TimeSinceLastFight",
            "Fighter2_FinishRate",
            "Fighter2_Wins",
            "Fighter2_Losses",
            "Fighter2_Draws",
            "Fighter2_NoContests",
            "Fighter2_AvgControlTime",
            "Fighter2_AvgSubmissionAttempts",
            "Fighter2_AvgLegStrikes",
            "Fighter2_AvgClinchStrikes",
            "Fighter2_AvgStrikesLanded",
            "Fighter2_AvgStrikesAttempted",
            "Fighter2_StrikeAccuracy",
            "Fighter2_AvgTakedownsLanded",
            "Fighter2_AvgTakedownsAttempted",
            "Fighter2_AvgReversals",
        ]

        if len(features) != len(feature_names):
            # Pad or truncate to match expected size
            if len(features) < len(feature_names):
                features = list(features) + [0] * (len(feature_names) - len(features))
            else:
                features = features[: len(feature_names)]

        features = pd.DataFrame([features], columns=feature_names)

    # Handle missing values
    features = features.fillna(0)

    # Try to load scaler from S3, if it exists
    try:
        s3 = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET", "mpm-bucket-001")

        with tempfile.NamedTemporaryFile() as tmp_file:
            s3.download_file(bucket, "models/scaler.pkl", tmp_file.name)
            scaler = joblib.load(tmp_file.name)
            features_scaled = scaler.transform(features)
    except:
        # If no scaler exists, apply basic normalization
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

    return features_scaled
