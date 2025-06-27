import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import boto3

from config import DATA_PATH
from preprocessing import (
    parse_height,
    parse_reach,
    parse_control_time,
    calculate_fight_time,
    process_landed_attempted,
    compute_historical_stats,
)


def download_data_from_s3(bucket, key, dest):
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file(bucket, key, dest)


def upload_results_to_s3(local_dir, bucket, s3_prefix):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            s3.upload_file(local_path, bucket, s3_key)


def load_fight_data():
    """
    Load and preprocess fight event data for model training.

    This function:
    1. Loads raw fight event data
    2. Processes fighter statistics and fight metrics
    3. Calculates historical performance metrics
    4. Applies feature engineering and preprocessing
    5. Returns processed features and encoded labels

    Returns:
        tuple: (X_processed, y, label_encoder)
            - X_processed: Processed feature matrix
            - y: Encoded target labels
            - label_encoder: Fitted label encoder for target classes
    """
    try:
        print("Loading fight event data...")
        fight_data_path = os.path.join(DATA_PATH, "fight_events.csv")

        # Load and clean initial data
        fight_data = pd.read_csv(
            fight_data_path, quotechar='"', parse_dates=["EventDate"]
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

        # Define feature columns for preprocessing
        numerical_columns = [
            "Fighter1_Height_cm",
            "Fighter1_Reach_cm",
            "Fighter1_Age",
            "Fighter1_AvgFightTime",
            "Fighter1_TimeSinceLastFight",
            "Fighter1_FinishRate",
            "Fighter1_Wins",
            "Fighter1_Losses",
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
            "Fighter2_Height_cm",
            "Fighter2_Reach_cm",
            "Fighter2_Age",
            "Fighter2_AvgFightTime",
            "Fighter2_TimeSinceLastFight",
            "Fighter2_FinishRate",
            "Fighter2_Wins",
            "Fighter2_Losses",
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

        categorical_columns = ["Fighter1_Stance", "Fighter2_Stance"]
        relevant_columns = numerical_columns + categorical_columns

        # Prepare features and target
        X = fight_data[relevant_columns]
        y = fight_data["Winner"]

        # Encode target labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Create preprocessing pipelines
        numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)

        # Convert to dataframe with feature names
        feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

        # Combine features and target
        y_df = pd.Series(y, name="Winner")
        final_df = pd.concat([X_processed_df, y_df], axis=1)

        # Save processed data
        processed_data_path = os.path.join(DATA_PATH, "processed_fight_data.csv")
        final_df.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Processed data saved to {processed_data_path}")

        print(f"Fight data: {len(fight_data)} records loaded.")
        print(
            f"Date range: {fight_data['EventDate'].min()} to {fight_data['EventDate'].max()}"
        )

        return X_processed, y, le

    except Exception as e:
        print(f"An error occurred while loading fight event data: {e}")
        raise
