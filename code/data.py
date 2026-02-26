import os
import numpy as np
import pandas as pd
import boto3
import re
import tempfile
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DIFF_FEATURE_SUFFIXES = [
    "Height_cm",
    "Reach_cm",
    "Age",
    "AvgFightTime",
    "TimeSinceLastFight",
    "FinishRate",
    "Wins",
    "Losses",
    "WinRate",
    "TotalFights",
    "AvgControlTime",
    "AvgSubmissionAttempts",
    "AvgLegStrikes",
    "AvgClinchStrikes",
    "AvgGroundStrikes",
    "AvgHeadStrikes",
    "AvgStrikesLanded",
    "AvgStrikesAttempted",
    "StrikeAccuracy",
    "TakedownAccuracy",
    "AvgTakedownsLanded",
    "AvgTakedownsAttempted",
    "AvgReversals",
]


def compute_differential_features(df):
    """Compute Fighter1 - Fighter2 differential columns for all paired numerical features."""
    for suffix in DIFF_FEATURE_SUFFIXES:
        df[f"{suffix}_Diff"] = df[f"Fighter1_{suffix}"] - df[f"Fighter2_{suffix}"]
    return df


def build_preprocessor(X_train_df, numerical_columns, categorical_columns):
    """Fit preprocessing pipeline on training data only to prevent data leakage."""
    numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    preprocessor.fit(X_train_df)
    return preprocessor


def upload_results_to_s3(local_dir, bucket, s3_prefix):
    print("Uploading results to S3...")
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            s3.upload_file(local_path, bucket, s3_key)

    # Upload a blank json file as a completion signal to trigger events in AWS lambda
    done_key = os.path.join(s3_prefix, "done.json").replace("\\", "/")
    with open("done.json", "w") as f:
        f.write("")
    s3.upload_file("done.json", bucket, done_key)
    print(f"Results uploaded to s3://{bucket}/{s3_prefix}")


def load_fight_data(s3_bucket, s3_data_key, s3_results_prefix):
    """
    Load and preprocess fight event data for model training.

    This function:
    1. Loads fight event data directly from S3
    2. Processes fighter statistics and fight metrics
    3. Calculates historical performance metrics
    4. Computes differential features
    5. Returns unprocessed feature DataFrame for caller to handle preprocessing

    Returns:
        tuple: (X_df, y, label_encoder, event_dates)
            - X_df: Feature DataFrame (before StandardScaler/OneHotEncoder)
            - y: Encoded target labels
            - label_encoder: Fitted label encoder for target classes
            - event_dates: Series of EventDate values aligned with X_df
    """
    try:
        print("Loading fight event data from S3...")
        fight_data = pd.read_csv(
            f"s3://{s3_bucket}/{s3_data_key}",
            quotechar='"',
            parse_dates=["EventDate"],
            storage_options={"anon": False},
        )
        print(f"Records before dropping data: {len(fight_data)}")

        fight_data = fight_data.drop(columns=["EventName"])
        fight_data = fight_data[~fight_data["Winner"].isin(["NC", "D"])]

        # Process physical attributes
        fight_data["Fighter1_Height_cm"] = fight_data["Fighter1_Height"].apply(
            parse_height
        )
        fight_data["Fighter2_Height_cm"] = fight_data["Fighter2_Height"].apply(
            parse_height
        )
        fight_data["Fighter1_Reach_cm"] = fight_data["Fighter1_Reach"].apply(
            parse_reach
        )
        fight_data["Fighter2_Reach_cm"] = fight_data["Fighter2_Reach"].apply(
            parse_reach
        )

        # Process fight metrics
        fight_data["Fighter1_Control_Time_sec"] = fight_data[
            "Fighter1_Control_Time"
        ].apply(parse_control_time)
        fight_data["Fighter2_Control_Time_sec"] = fight_data[
            "Fighter2_Control_Time"
        ].apply(parse_control_time)

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

        # Compute historical performance stats
        fight_data = compute_historical_stats(fight_data)

        # Compute differential features (Fighter1 - Fighter2)
        fight_data = compute_differential_features(fight_data)

        # Final data cleaning
        fight_data = fight_data.dropna()
        print(f"Records after dropping data: {len(fight_data)}")

        # Analyze class distribution
        winner_counts = fight_data["Winner"].value_counts()
        print("Class distribution of 'Winner':")
        print(winner_counts)

        total_fights = len(fight_data)
        fighter1_win_percentage = (winner_counts.get("1", 0) / total_fights) * 100
        fighter2_win_percentage = (winner_counts.get("2", 0) / total_fights) * 100

        print(f"Fighter 1 wins: {fighter1_win_percentage:.2f}%")
        print(f"Fighter 2 wins: {fighter2_win_percentage:.2f}%")

        # Use differential features instead of absolute Fighter1/Fighter2 columns
        numerical_columns = [f"{suffix}_Diff" for suffix in DIFF_FEATURE_SUFFIXES]

        categorical_columns = ["Fighter1_Stance", "Fighter2_Stance"]
        relevant_columns = numerical_columns + categorical_columns

        # Prepare features and target
        X_df = fight_data[relevant_columns].reset_index(drop=True)
        event_dates = fight_data["EventDate"].reset_index(drop=True)
        y = fight_data["Winner"].reset_index(drop=True)

        # Encode target labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        print(f"Fight data: {len(fight_data)} records loaded.")
        print(
            f"Date range: {event_dates.min()} to {event_dates.max()}"
        )

        return X_df, y, le, event_dates

    except Exception as e:
        print(f"An error occurred while loading fight event data: {e}")
        raise


def get_latest_fighter_stats_by_url(fight_data, fighter_url):
    """
    Get the latest historical statistics for a specific fighter using their URL.
    This is more reliable than name matching since URLs are unique identifiers.

    Args:
        fight_data: DataFrame with processed historical fight data
        fighter_url: URL of the fighter profile

    Returns:
        dict: Latest statistics for the fighter, with proper Fighter1/Fighter2 prefixes
    """
    # Find all fights involving this fighter by URL (stored in Fighter1_ID/Fighter2_ID)
    fighter1_fights = fight_data[fight_data["Fighter1_ID"] == fighter_url]
    fighter2_fights = fight_data[fight_data["Fighter2_ID"] == fighter_url]

    # Get the most recent fight data
    latest_stats = {}
    most_recent_date = None

    # Check Fighter1 position fights
    if not fighter1_fights.empty:
        latest_f1 = fighter1_fights.loc[fighter1_fights["EventDate"].idxmax()]
        if most_recent_date is None or latest_f1["EventDate"] > most_recent_date:
            most_recent_date = latest_f1["EventDate"]

            # Get stats with Fighter1_ prefix
            for col in latest_f1.index:
                if col.startswith("Fighter1_") and (
                    col.endswith(
                        (
                            "_AvgFightTime",
                            "_TimeSinceLastFight",
                            "_FinishRate",
                            "_Wins",
                            "_Losses",
                            "_Draws",
                            "_NoContests",
                            "_WinRate",
                            "_TotalFights",
                            "_AvgControlTime",
                            "_AvgSubmissionAttempts",
                            "_AvgLegStrikes",
                            "_AvgClinchStrikes",
                            "_AvgGroundStrikes",
                            "_AvgHeadStrikes",
                            "_AvgStrikesLanded",
                            "_AvgStrikesAttempted",
                            "_StrikeAccuracy",
                            "_TakedownAccuracy",
                            "_AvgTakedownsLanded",
                            "_AvgTakedownsAttempted",
                            "_AvgReversals",
                            "_Height_cm",
                            "_Reach_cm",
                            "_Age",
                            "_Stance",
                        )
                    )
                ):
                    latest_stats[col] = latest_f1[col]

    # Check Fighter2 position fights
    if not fighter2_fights.empty:
        latest_f2 = fighter2_fights.loc[fighter2_fights["EventDate"].idxmax()]
        if most_recent_date is None or latest_f2["EventDate"] > most_recent_date:
            most_recent_date = latest_f2["EventDate"]

            # Get stats with Fighter2_ prefix (keep original prefixes)
            for col in latest_f2.index:
                if col.startswith("Fighter2_") and (
                    col.endswith(
                        (
                            "_AvgFightTime",
                            "_TimeSinceLastFight",
                            "_FinishRate",
                            "_Wins",
                            "_Losses",
                            "_Draws",
                            "_NoContests",
                            "_WinRate",
                            "_TotalFights",
                            "_AvgControlTime",
                            "_AvgSubmissionAttempts",
                            "_AvgLegStrikes",
                            "_AvgClinchStrikes",
                            "_AvgGroundStrikes",
                            "_AvgHeadStrikes",
                            "_AvgStrikesLanded",
                            "_AvgStrikesAttempted",
                            "_StrikeAccuracy",
                            "_TakedownAccuracy",
                            "_AvgTakedownsLanded",
                            "_AvgTakedownsAttempted",
                            "_AvgReversals",
                            "_Height_cm",
                            "_Reach_cm",
                            "_Age",
                            "_Stance",
                        )
                    )
                ):
                    latest_stats[col] = latest_f2[col]

    return latest_stats


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
            "TotalGroundStrikes": 0,
            "TotalHeadStrikes": 0,
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
        fight_data[f"{fighter_num}_WinRate"] = 0.0
        fight_data[f"{fighter_num}_TotalFights"] = 0
        fight_data[f"{fighter_num}_AvgControlTime"] = 0.0
        fight_data[f"{fighter_num}_AvgSubmissionAttempts"] = 0.0
        fight_data[f"{fighter_num}_AvgLegStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgClinchStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgGroundStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgHeadStrikes"] = 0.0
        fight_data[f"{fighter_num}_AvgStrikesLanded"] = 0.0
        fight_data[f"{fighter_num}_AvgStrikesAttempted"] = 0.0
        fight_data[f"{fighter_num}_StrikeAccuracy"] = 0.0
        fight_data[f"{fighter_num}_TakedownAccuracy"] = 0.0
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
                n = stats_before["NumFights"]
                fight_data.at[idx, f"{fighter_num}_AvgFightTime"] = (
                    stats_before["TotalFightTime"] / n
                )
                if stats_before["LastFightDate"] is not None:
                    days = (event_date - stats_before["LastFightDate"]).days
                    fight_data.at[idx, f"{fighter_num}_TimeSinceLastFight"] = days
                if stats_before["Wins"] > 0:
                    fight_data.at[idx, f"{fighter_num}_FinishRate"] = (
                        stats_before["WinsByFinish"] / stats_before["Wins"]
                    )
                total_decided = stats_before["Wins"] + stats_before["Losses"]
                if total_decided > 0:
                    fight_data.at[idx, f"{fighter_num}_WinRate"] = (
                        stats_before["Wins"] / total_decided
                    )
                fight_data.at[idx, f"{fighter_num}_AvgControlTime"] = (
                    stats_before["TotalControlTime"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgSubmissionAttempts"] = (
                    stats_before["TotalSubmissionAttempts"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgLegStrikes"] = (
                    stats_before["TotalLegStrikes"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgClinchStrikes"] = (
                    stats_before["TotalClinchStrikes"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgGroundStrikes"] = (
                    stats_before["TotalGroundStrikes"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgHeadStrikes"] = (
                    stats_before["TotalHeadStrikes"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgStrikesLanded"] = (
                    stats_before["TotalStrikesLanded"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgStrikesAttempted"] = (
                    stats_before["TotalStrikesAttempted"] / n
                )
                fight_data.at[idx, f"{fighter_num}_StrikeAccuracy"] = (
                    stats_before["TotalStrikesLanded"]
                    / stats_before["TotalStrikesAttempted"]
                    if stats_before["TotalStrikesAttempted"] > 0
                    else 0
                )
                fight_data.at[idx, f"{fighter_num}_TakedownAccuracy"] = (
                    stats_before["TotalTakedownsLanded"]
                    / stats_before["TotalTakedownsAttempted"]
                    if stats_before["TotalTakedownsAttempted"] > 0
                    else 0
                )
                fight_data.at[idx, f"{fighter_num}_AvgTakedownsLanded"] = (
                    stats_before["TotalTakedownsLanded"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgTakedownsAttempted"] = (
                    stats_before["TotalTakedownsAttempted"] / n
                )
                fight_data.at[idx, f"{fighter_num}_AvgReversals"] = (
                    stats_before["TotalReversals"] / n
                )
            # Assign win/loss/draw/experience counts before fight
            fight_data.at[idx, f"{fighter_num}_Wins"] = stats_before["Wins"]
            fight_data.at[idx, f"{fighter_num}_Losses"] = stats_before["Losses"]
            fight_data.at[idx, f"{fighter_num}_Draws"] = stats_before["Draws"]
            fight_data.at[idx, f"{fighter_num}_NoContests"] = stats_before["NoContests"]
            fight_data.at[idx, f"{fighter_num}_TotalFights"] = stats_before["NumFights"]

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
            ground = row[f"{fighter_num}_Ground_Strikes_Landed"]
            stats["TotalGroundStrikes"] += ground if not np.isnan(ground) else 0
            head = row[f"{fighter_num}_Head_Strikes_Landed"]
            stats["TotalHeadStrikes"] += head if not np.isnan(head) else 0
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
    upcoming_fights_data,
    historical_fight_data_path="data/fight_events.csv",
    s3_bucket=None,
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

    # Get bucket name from parameter or environment variable
    if s3_bucket is None:
        bucket = os.environ.get("S3_BUCKET")
        if bucket is None:
            raise ValueError(
                "S3 bucket must be provided either as parameter or S3_BUCKET environment variable"
            )
    else:
        bucket = s3_bucket

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
    # Handle different input formats for upcoming fights data
    if isinstance(upcoming_fights_data, dict):
        # If it's a JSON with event structure like {"EventName": "...", "Fights": [...]}
        if "Fights" in upcoming_fights_data:
            fights_list = upcoming_fights_data["Fights"]
        else:
            # Single fight as dict
            fights_list = [upcoming_fights_data]
    elif isinstance(upcoming_fights_data, list):
        # List of fights
        fights_list = upcoming_fights_data
    else:
        # DataFrame
        fights_list = upcoming_fights_data.to_dict("records")

    # For each upcoming fight, get the latest historical stats for each fighter
    processed_features = []

    for fight in fights_list:
        # Handle different field name formats
        fighter1_name = fight.get(
            "Fighter1Name", fight.get("Fighter1", fight.get("fighter1"))
        )
        fighter2_name = fight.get(
            "Fighter2Name", fight.get("Fighter2", fight.get("fighter2"))
        )
        fighter1_url = fight.get("Fighter1Url", fight.get("Fighter1_URL"))
        fighter2_url = fight.get("Fighter2Url", fight.get("Fighter2_URL"))

        print(f"Processing fight: {fighter1_name} vs {fighter2_name}")

        # Try URL-based matching first (more reliable), fallback to name matching
        fighter1_stats = {}
        fighter2_stats = {}

        # Get fighter stats using URL only (no name fallback)
        fighter1_stats = {}
        fighter2_stats = {}

        if fighter1_url:
            fighter1_stats = get_latest_fighter_stats_by_url(fight_data, fighter1_url)
            print(f"Found {len(fighter1_stats)} stats for {fighter1_name} via URL")
        else:
            print(f"No URL provided for Fighter1: {fighter1_name} - skipping stats")

        if fighter2_url:
            fighter2_stats = get_latest_fighter_stats_by_url(fight_data, fighter2_url)
            print(f"Found {len(fighter2_stats)} stats for {fighter2_name} via URL")
        else:
            print(f"No URL provided for Fighter2: {fighter2_name} - skipping stats")

        # Extract stance information (defaulting to Orthodox if not available)
        fighter1_stance = fighter1_stats.get("Fighter1_Stance", "Orthodox")
        fighter2_stance = fighter2_stats.get("Fighter2_Stance", "Orthodox")

        # Build feature dictionary with absolute values for both fighters
        fight_features = {}
        for suffix in DIFF_FEATURE_SUFFIXES:
            f1_key = f"Fighter1_{suffix}"
            f2_key = f"Fighter2_{suffix}"
            default = 0
            if suffix == "Height_cm":
                default = 177
            elif suffix == "Reach_cm":
                default = 183
            elif suffix == "Age":
                default = 30
            fight_features[f1_key] = fighter1_stats.get(f1_key, default)
            fight_features[f2_key] = fighter2_stats.get(f2_key, default)

        fight_features["Fighter1_Stance"] = fighter1_stance
        fight_features["Fighter2_Stance"] = fighter2_stance

        processed_features.append(fight_features)

    # Convert to DataFrame and compute differential features
    features_df = pd.DataFrame(processed_features)
    features_df = compute_differential_features(features_df)

    # Use differential features matching training format
    numerical_columns = [f"{suffix}_Diff" for suffix in DIFF_FEATURE_SUFFIXES]

    categorical_columns = ["Fighter1_Stance", "Fighter2_Stance"]

    # All possible stance categories to ensure consistent one-hot encoding
    stance_categories = ["Orthodox", "Southpaw", "Switch", "Open Stance"]

    # Create preprocessing pipelines exactly as in training
    numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    categories=[stance_categories, stance_categories],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    # Apply the same preprocessing as training
    features_processed = preprocessor.fit_transform(features_df)

    return features_processed
