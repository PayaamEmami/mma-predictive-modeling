import re
import pandas as pd
import numpy as np
import boto3
import tempfile
import os


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


def preprocess_features(
    upcoming_fights_data, historical_fight_data_path="fight_events.csv"
):
    """
    Preprocess features for inference by computing historical statistics.

    This function replicates the same preprocessing pipeline used in training:
    1. Loads historical fight data
    2. Processes fighter statistics and fight metrics
    3. Calculates historical performance metrics up to current date
    4. Extracts features for upcoming fights
    5. Applies the same feature engineering as training

    Args:
        upcoming_fights_data: DataFrame or dict with upcoming fight matchups
        historical_fight_data_path: Path to historical fight data

    Returns:
        numpy.array: Processed feature matrix ready for model prediction
    """

    # Load historical fight data from S3
    s3 = boto3.client("s3")
    bucket = os.environ.get("S3_BUCKET", "mpm-bucket-001")

    # Download and load historical fight data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
        try:
            s3.download_file(bucket, historical_fight_data_path, tmp_file.name)
            fight_data = pd.read_csv(
                tmp_file.name, quotechar='"', parse_dates=["EventDate"]
            )
        finally:
            os.unlink(tmp_file.name)

    print(f"Loaded {len(fight_data)} historical fights for feature computation")

    # Apply the same preprocessing as training pipeline
    # Remove non-contests and draws
    fight_data = fight_data.drop(columns=["EventName"], errors="ignore")
    fight_data = fight_data[~fight_data["Winner"].isin(["NC", "D"])]

    # Process physical attributes
    fight_data["Fighter1_Height_cm"] = fight_data["Fighter1_Height"].apply(parse_height)
    fight_data["Fighter2_Height_cm"] = fight_data["Fighter2_Height"].apply(parse_height)
    fight_data["Fighter1_Reach_cm"] = fight_data["Fighter1_Reach"].apply(parse_reach)
    fight_data["Fighter2_Reach_cm"] = fight_data["Fighter2_Reach"].apply(parse_reach)

    # Process fight metrics
    fight_data["Fighter1_Control_Time_sec"] = fight_data["Fighter1_Control_Time"].apply(
        parse_control_time
    )
    fight_data["Fighter2_Control_Time_sec"] = fight_data["Fighter2_Control_Time"].apply(
        parse_control_time
    )

    # Calculate fighter ages
    fight_data["Fighter1_Age"] = (
        fight_data["EventDate"]
        - pd.to_datetime(fight_data["Fighter1_DOB"], errors="coerce")
    ).dt.days / 365.25
    fight_data["Fighter1_Age"] = fight_data["Fighter1_Age"].fillna(30)
    fight_data["Fighter2_Age"] = (
        fight_data["EventDate"]
        - pd.to_datetime(fight_data["Fighter2_DOB"], errors="coerce")
    ).dt.days / 365.25
    fight_data["Fighter2_Age"] = fight_data["Fighter2_Age"].fillna(30)

    # Process strike statistics
    strike_columns = [
        "Fighter1_Significant_Strikes",
        "Fighter1_Head_Strikes",
        "Fighter1_Body_Strikes",
        "Fighter1_Leg_Strikes",
        "Fighter1_Distance_Strikes",
        "Fighter1_Clinch_Strikes",
        "Fighter1_Ground_Strikes",
        "Fighter1_Takedowns",
        "Fighter2_Significant_Strikes",
        "Fighter2_Head_Strikes",
        "Fighter2_Body_Strikes",
        "Fighter2_Leg_Strikes",
        "Fighter2_Distance_Strikes",
        "Fighter2_Clinch_Strikes",
        "Fighter2_Ground_Strikes",
        "Fighter2_Takedowns",
    ]

    for col in strike_columns:
        process_landed_attempted(fight_data, col)

    # Calculate fight duration
    fight_data["Fight_Time_sec"] = fight_data.apply(
        lambda row: calculate_fight_time(row["Round"], row["Time"]), axis=1
    )

    # Compute historical performance stats (this is the key step!)
    fight_data = compute_historical_stats(fight_data)

    # Clean the data
    fight_data = fight_data.dropna()

    # Now extract features for upcoming fights
    # Convert upcoming_fights_data to DataFrame if needed
    if isinstance(upcoming_fights_data, dict):
        upcoming_fights_data = pd.DataFrame([upcoming_fights_data])
    elif not isinstance(upcoming_fights_data, pd.DataFrame):
        # If it's a list of fights
        upcoming_fights_data = pd.DataFrame(upcoming_fights_data)

    # For each upcoming fight, get the latest historical stats for each fighter
    processed_features = []

    for _, upcoming_fight in upcoming_fights_data.iterrows():
        fighter1_name = upcoming_fight.get("Fighter1", upcoming_fight.get("fighter1"))
        fighter2_name = upcoming_fight.get("Fighter2", upcoming_fight.get("fighter2"))

        # Get latest stats for each fighter from historical data
        fighter1_stats = get_latest_fighter_stats(fight_data, fighter1_name)
        fighter2_stats = get_latest_fighter_stats(fight_data, fighter2_name)

        # Combine into feature vector matching training format
        feature_vector = [
            fighter1_stats.get("Fighter1_Height_cm", 177),
            fighter1_stats.get("Fighter1_Reach_cm", 183),
            fighter1_stats.get("Fighter1_Age", 30),
            fighter1_stats.get("Fighter1_AvgFightTime", 0),
            fighter1_stats.get("Fighter1_TimeSinceLastFight", 0),
            fighter1_stats.get("Fighter1_FinishRate", 0),
            fighter1_stats.get("Fighter1_Wins", 0),
            fighter1_stats.get("Fighter1_Losses", 0),
            fighter1_stats.get("Fighter1_AvgControlTime", 0),
            fighter1_stats.get("Fighter1_AvgSubmissionAttempts", 0),
            fighter1_stats.get("Fighter1_AvgLegStrikes", 0),
            fighter1_stats.get("Fighter1_AvgClinchStrikes", 0),
            fighter1_stats.get("Fighter1_AvgStrikesLanded", 0),
            fighter1_stats.get("Fighter1_AvgStrikesAttempted", 0),
            fighter1_stats.get("Fighter1_StrikeAccuracy", 0),
            fighter1_stats.get("Fighter1_AvgTakedownsLanded", 0),
            fighter1_stats.get("Fighter1_AvgTakedownsAttempted", 0),
            fighter1_stats.get("Fighter1_AvgReversals", 0),
            fighter2_stats.get("Fighter2_Height_cm", 177),
            fighter2_stats.get("Fighter2_Reach_cm", 183),
            fighter2_stats.get("Fighter2_Age", 30),
            fighter2_stats.get("Fighter2_AvgFightTime", 0),
            fighter2_stats.get("Fighter2_TimeSinceLastFight", 0),
            fighter2_stats.get("Fighter2_FinishRate", 0),
            fighter2_stats.get("Fighter2_Wins", 0),
            fighter2_stats.get("Fighter2_Losses", 0),
            fighter2_stats.get("Fighter2_AvgControlTime", 0),
            fighter2_stats.get("Fighter2_AvgSubmissionAttempts", 0),
            fighter2_stats.get("Fighter2_AvgLegStrikes", 0),
            fighter2_stats.get("Fighter2_AvgClinchStrikes", 0),
            fighter2_stats.get("Fighter2_AvgStrikesLanded", 0),
            fighter2_stats.get("Fighter2_AvgStrikesAttempted", 0),
            fighter2_stats.get("Fighter2_StrikeAccuracy", 0),
            fighter2_stats.get("Fighter2_AvgTakedownsLanded", 0),
            fighter2_stats.get("Fighter2_AvgTakedownsAttempted", 0),
            fighter2_stats.get("Fighter2_AvgReversals", 0),
        ]

        processed_features.append(feature_vector)

    # Convert to numpy array and apply basic normalization
    features_array = np.array(processed_features)

    # Apply the same normalization as training (simplified version)
    # In a production system, you'd load the saved scaler from training
    means = np.mean(features_array, axis=0)
    stds = np.std(features_array, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero

    features_scaled = (features_array - means) / stds

    return features_scaled


def get_latest_fighter_stats(fight_data, fighter_name):
    """
    Get the latest historical statistics for a specific fighter.
    This includes their performance up to and including their most recent fight.

    Args:
        fight_data: DataFrame with processed historical fight data
        fighter_name: Name of the fighter to get stats for

    Returns:
        dict: Latest statistics for the fighter including their most recent fight
    """
    # Find all fights involving this fighter
    fighter1_fights = fight_data[fight_data["Fighter1"] == fighter_name]
    fighter2_fights = fight_data[fight_data["Fighter2"] == fighter_name]

    # Get the most recent fight data
    latest_stats = {}
    most_recent_date = None

    # Check Fighter1 position fights
    if not fighter1_fights.empty:
        latest_f1 = fighter1_fights.loc[fighter1_fights["EventDate"].idxmax()]
        most_recent_date = latest_f1["EventDate"]

        # Get stats that were calculated BEFORE this fight (pre-fight stats)
        for col in latest_f1.index:
            if col.startswith("Fighter1_") and col.endswith(
                (
                    "_AvgFightTime",
                    "_TimeSinceLastFight",
                    "_FinishRate",
                    "_Wins",
                    "_Losses",
                    "_Draws",
                    "_NoContests",
                    "_AvgControlTime",
                    "_AvgSubmissionAttempts",
                    "_AvgLegStrikes",
                    "_AvgClinchStrikes",
                    "_AvgStrikesLanded",
                    "_AvgStrikesAttempted",
                    "_StrikeAccuracy",
                    "_AvgTakedownsLanded",
                    "_AvgTakedownsAttempted",
                    "_AvgReversals",
                    "_Height_cm",
                    "_Reach_cm",
                    "_Age",
                )
            ):
                latest_stats[col] = latest_f1[col]

    # Check Fighter2 position fights
    if not fighter2_fights.empty:
        latest_f2 = fighter2_fights.loc[fighter2_fights["EventDate"].idxmax()]

        # Use whichever fight is more recent
        if most_recent_date is None or latest_f2["EventDate"] > most_recent_date:
            most_recent_date = latest_f2["EventDate"]
            # Clear previous stats and use Fighter2 position stats
            latest_stats = {}

            for col in latest_f2.index:
                if col.startswith("Fighter2_") and col.endswith(
                    (
                        "_AvgFightTime",
                        "_TimeSinceLastFight",
                        "_FinishRate",
                        "_Wins",
                        "_Losses",
                        "_Draws",
                        "_NoContests",
                        "_AvgControlTime",
                        "_AvgSubmissionAttempts",
                        "_AvgLegStrikes",
                        "_AvgClinchStrikes",
                        "_AvgStrikesLanded",
                        "_AvgStrikesAttempted",
                        "_StrikeAccuracy",
                        "_AvgTakedownsLanded",
                        "_AvgTakedownsAttempted",
                        "_AvgReversals",
                        "_Height_cm",
                        "_Reach_cm",
                        "_Age",
                    )
                ):
                    # Convert Fighter2_ stats to Fighter1_ format for consistency
                    new_col = col.replace("Fighter2_", "Fighter1_")
                    latest_stats[new_col] = latest_f2[col]
        elif latest_f2["EventDate"] == most_recent_date:
            # Same fight date - merge stats, prioritizing Fighter2 position if available
            for col in latest_f2.index:
                if col.startswith("Fighter2_") and col.endswith(
                    (
                        "_AvgFightTime",
                        "_TimeSinceLastFight",
                        "_FinishRate",
                        "_Wins",
                        "_Losses",
                        "_Draws",
                        "_NoContests",
                        "_AvgControlTime",
                        "_AvgSubmissionAttempts",
                        "_AvgLegStrikes",
                        "_AvgClinchStrikes",
                        "_AvgStrikesLanded",
                        "_AvgStrikesAttempted",
                        "_StrikeAccuracy",
                        "_AvgTakedownsLanded",
                        "_AvgTakedownsAttempted",
                        "_AvgReversals",
                        "_Height_cm",
                        "_Reach_cm",
                        "_Age",
                    )
                ):
                    new_col = col.replace("Fighter2_", "Fighter1_")
                    latest_stats[new_col] = latest_f2[col]

    # Now we need to manually compute the updated stats that include their most recent fight
    # This simulates what the stats would be AFTER their most recent fight
    if most_recent_date is not None:
        # Find the most recent fight for this fighter
        recent_fight = None
        fighter_position = None

        if not fighter1_fights.empty:
            f1_recent = fighter1_fights.loc[fighter1_fights["EventDate"].idxmax()]
            if f1_recent["EventDate"] == most_recent_date:
                recent_fight = f1_recent
                fighter_position = 1

        if not fighter2_fights.empty:
            f2_recent = fighter2_fights.loc[fighter2_fights["EventDate"].idxmax()]
            if f2_recent["EventDate"] == most_recent_date:
                if (
                    recent_fight is None
                    or f2_recent["EventDate"] >= recent_fight["EventDate"]
                ):
                    recent_fight = f2_recent
                    fighter_position = 2

        if recent_fight is not None and fighter_position is not None:
            # Update stats to include the most recent fight
            prefix = f"Fighter{fighter_position}_"

            # Get pre-fight stats
            num_fights_before = (
                latest_stats.get("Fighter1_Wins", 0)
                + latest_stats.get("Fighter1_Losses", 0)
                + latest_stats.get("Fighter1_Draws", 0)
            )

            # Update win/loss record based on most recent fight outcome
            winner = str(recent_fight["Winner"])
            if (winner == "1" and fighter_position == 1) or (
                winner == "2" and fighter_position == 2
            ):
                # Fighter won
                latest_stats["Fighter1_Wins"] = latest_stats.get("Fighter1_Wins", 0) + 1
            elif (winner == "1" and fighter_position == 2) or (
                winner == "2" and fighter_position == 1
            ):
                # Fighter lost
                latest_stats["Fighter1_Losses"] = (
                    latest_stats.get("Fighter1_Losses", 0) + 1
                )
            elif winner == "D":
                # Draw
                latest_stats["Fighter1_Draws"] = (
                    latest_stats.get("Fighter1_Draws", 0) + 1
                )

            # Update averages to include the most recent fight
            if num_fights_before > 0:
                total_fights_after = num_fights_before + 1

                # Update average fight time
                old_total_time = (
                    latest_stats.get("Fighter1_AvgFightTime", 0) * num_fights_before
                )
                recent_fight_time = recent_fight.get("Fight_Time_sec", 0)
                if not pd.isna(recent_fight_time):
                    new_avg_fight_time = (
                        old_total_time + recent_fight_time
                    ) / total_fights_after
                    latest_stats["Fighter1_AvgFightTime"] = new_avg_fight_time

                # Update other averages similarly
                recent_control_time = recent_fight.get(f"{prefix}Control_Time_sec", 0)
                if not pd.isna(recent_control_time):
                    old_total_control = (
                        latest_stats.get("Fighter1_AvgControlTime", 0)
                        * num_fights_before
                    )
                    latest_stats["Fighter1_AvgControlTime"] = (
                        old_total_control + recent_control_time
                    ) / total_fights_after

                # Update strike statistics
                recent_strikes_landed = recent_fight.get(
                    f"{prefix}Significant_Strikes_Landed", 0
                )
                recent_strikes_attempted = recent_fight.get(
                    f"{prefix}Significant_Strikes_Attempted", 0
                )

                if not pd.isna(recent_strikes_landed):
                    old_total_landed = (
                        latest_stats.get("Fighter1_AvgStrikesLanded", 0)
                        * num_fights_before
                    )
                    latest_stats["Fighter1_AvgStrikesLanded"] = (
                        old_total_landed + recent_strikes_landed
                    ) / total_fights_after

                if not pd.isna(recent_strikes_attempted):
                    old_total_attempted = (
                        latest_stats.get("Fighter1_AvgStrikesAttempted", 0)
                        * num_fights_before
                    )
                    latest_stats["Fighter1_AvgStrikesAttempted"] = (
                        old_total_attempted + recent_strikes_attempted
                    ) / total_fights_after

                # Recalculate strike accuracy
                if latest_stats.get("Fighter1_AvgStrikesAttempted", 0) > 0:
                    latest_stats["Fighter1_StrikeAccuracy"] = (
                        latest_stats["Fighter1_AvgStrikesLanded"]
                        / latest_stats["Fighter1_AvgStrikesAttempted"]
                    )

    return latest_stats
