import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import DATA_PATH
from preprocessing import (
    parse_height,
    parse_reach,
    parse_control_time,
    calculate_fight_time,
    process_landed_attempted,
    is_finish,
)


def load_ufc_data():
    """
    Load and preprocess UFC fight data for model training.
    
    This function:
    1. Loads raw UFC fight data
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
        print("Loading UFC data...")
        fight_data_path = os.path.join(DATA_PATH, "ufc_events.csv")

        # Load and clean initial data
        fight_data = pd.read_csv(
            fight_data_path, quotechar='"', parse_dates=["EventDate"]
        )
        print(f"Records before dropping data: {len(fight_data)}")

        fight_data = fight_data.drop(columns=["EventName"])
        fight_data = fight_data[~fight_data["Winner"].isin(["NC", "D"])]
        print(f"Records after dropping data: {len(fight_data)}")

        # Process physical attributes
        fight_data["Fighter1_Height_cm"] = fight_data["Fighter1_Height"].apply(parse_height)
        fight_data["Fighter2_Height_cm"] = fight_data["Fighter2_Height"].apply(parse_height)
        fight_data["Fighter1_Reach_cm"] = fight_data["Fighter1_Reach"].apply(parse_reach)
        fight_data["Fighter2_Reach_cm"] = fight_data["Fighter2_Reach"].apply(parse_reach)

        # Process fight metrics
        fight_data["Fighter1_Control_Time_sec"] = fight_data["Fighter1_Control_Time"].apply(parse_control_time)
        fight_data["Fighter2_Control_Time_sec"] = fight_data["Fighter2_Control_Time"].apply(parse_control_time)

        # Calculate fighter ages
        fight_data["Fighter1_Age"] = (
            fight_data["EventDate"] - pd.to_datetime(fight_data["Fighter1_DOB"], errors="coerce")
        ).dt.days / 365.25
        fight_data["Fighter1_Age"] = fight_data["Fighter1_Age"].fillna(30)
        fight_data["Fighter2_Age"] = (
            fight_data["EventDate"] - pd.to_datetime(fight_data["Fighter2_DOB"], errors="coerce")
        ).dt.days / 365.25
        fight_data["Fighter2_Age"] = fight_data["Fighter2_Age"].fillna(30)

        # Process strike statistics
        strike_columns = [
            "Fighter1_Significant_Strikes", "Fighter1_Head_Strikes", "Fighter1_Body_Strikes",
            "Fighter1_Leg_Strikes", "Fighter1_Distance_Strikes", "Fighter1_Clinch_Strikes",
            "Fighter1_Ground_Strikes", "Fighter1_Takedowns", "Fighter2_Significant_Strikes",
            "Fighter2_Head_Strikes", "Fighter2_Body_Strikes", "Fighter2_Leg_Strikes",
            "Fighter2_Distance_Strikes", "Fighter2_Clinch_Strikes", "Fighter2_Ground_Strikes",
            "Fighter2_Takedowns",
        ]

        for col in strike_columns:
            process_landed_attempted(fight_data, col)

        # Calculate fight duration
        fight_data["Fight_Time_sec"] = fight_data.apply(
            lambda row: calculate_fight_time(row["Round"], row["Time"]), axis=1
        )

        # Initialize fighter statistics tracking
        fighter_stats = defaultdict(
            lambda: {
                "TotalFightTime": 0, "NumFights": 0, "Wins": 0, "Losses": 0,
                "Draws": 0, "NoContests": 0, "WinsByFinish": 0, "LastFightDate": None,
                "TotalControlTime": 0, "TotalSubmissionAttempts": 0, "TotalLegStrikes": 0,
                "TotalClinchStrikes": 0, "TotalStrikesLanded": 0, "TotalStrikesAttempted": 0,
                "TotalTakedownsLanded": 0, "TotalTakedownsAttempted": 0, "TotalReversals": 0,
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

        # Calculate historical statistics for each fight
        for idx, row in fight_data.iterrows():
            fight_time = row["Fight_Time_sec"]
            event_date = row["EventDate"]

            # Update fight data with pre-fight statistics
            for fighter_num in ["Fighter1", "Fighter2"]:
                fighter_id = row[f"{fighter_num}_ID"]
                stats = fighter_stats[fighter_id]
                stats_before = stats.copy()

                if stats_before["NumFights"] > 0:
                    # Calculate average metrics
                    fight_data.at[idx, f"{fighter_num}_AvgFightTime"] = (
                        stats_before["TotalFightTime"] / stats_before["NumFights"]
                    )
                    
                    # Calculate time since last fight
                    if stats_before["LastFightDate"] is not None:
                        days_since_last_fight = (event_date - stats_before["LastFightDate"]).days
                        fight_data.at[idx, f"{fighter_num}_TimeSinceLastFight"] = days_since_last_fight
                    else:
                        fight_data.at[idx, f"{fighter_num}_TimeSinceLastFight"] = 0

                    # Calculate finish rate
                    if stats_before["Wins"] > 0:
                        fight_data.at[idx, f"{fighter_num}_FinishRate"] = (
                            stats_before["WinsByFinish"] / stats_before["Wins"]
                        )
                    else:
                        fight_data.at[idx, f"{fighter_num}_FinishRate"] = 0

                    # Calculate average performance metrics
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
                        stats_before["TotalStrikesLanded"] / stats_before["TotalStrikesAttempted"]
                        if stats_before["TotalStrikesAttempted"] > 0 else 0
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
                else:
                    # Default values are already zero for fighters with no previous fights
                    pass

                # Update win/loss/draw counts
                fight_data.at[idx, f"{fighter_num}_Wins"] = stats_before["Wins"]
                fight_data.at[idx, f"{fighter_num}_Losses"] = stats_before["Losses"]
                fight_data.at[idx, f"{fighter_num}_Draws"] = stats_before["Draws"]
                fight_data.at[idx, f"{fighter_num}_NoContests"] = stats_before["NoContests"]

            # Update fighter statistics with current fight data
            for fighter_num in ["Fighter1", "Fighter2"]:
                fighter_id = row[f"{fighter_num}_ID"]
                stats = fighter_stats[fighter_id]

                # Update basic fight metrics
                stats["TotalFightTime"] += fight_time if not np.isnan(fight_time) else 0
                stats["NumFights"] += 1
                stats["LastFightDate"] = event_date

                # Update performance metrics
                control_time = row[f"{fighter_num}_Control_Time_sec"]
                stats["TotalControlTime"] += control_time if not np.isnan(control_time) else 0

                sub_attempts = float(row[f"{fighter_num}_Submission_Attempts"])
                stats["TotalSubmissionAttempts"] += sub_attempts if not np.isnan(sub_attempts) else 0

                leg_strikes = row[f"{fighter_num}_Leg_Strikes_Landed"]
                stats["TotalLegStrikes"] += leg_strikes if not np.isnan(leg_strikes) else 0

                clinch_strikes = row[f"{fighter_num}_Clinch_Strikes_Landed"]
                stats["TotalClinchStrikes"] += clinch_strikes if not np.isnan(clinch_strikes) else 0

                strikes_landed = row[f"{fighter_num}_Significant_Strikes_Landed"]
                strikes_attempted = row[f"{fighter_num}_Significant_Strikes_Attempted"]
                stats["TotalStrikesLanded"] += strikes_landed if not np.isnan(strikes_landed) else 0
                stats["TotalStrikesAttempted"] += strikes_attempted if not np.isnan(strikes_attempted) else 0

                takedowns_landed = row.get(f"{fighter_num}_Takedowns_Landed", 0)
                takedowns_attempted = row.get(f"{fighter_num}_Takedowns_Attempted", 0)
                stats["TotalTakedownsLanded"] += takedowns_landed if not np.isnan(takedowns_landed) else 0
                stats["TotalTakedownsAttempted"] += takedowns_attempted if not np.isnan(takedowns_attempted) else 0

                reversals = float(row[f"{fighter_num}_Reversals"])
                stats["TotalReversals"] += reversals if not np.isnan(reversals) else 0

            # Update fight outcome statistics
            winner = str(row["Winner"])
            method = str(row["Method"])
            stats1 = fighter_stats[row["Fighter1_ID"]]
            stats2 = fighter_stats[row["Fighter2_ID"]]
            fight_was_finish = is_finish(method)

            if winner == "1":
                stats1["Wins"] += 1
                stats2["Losses"] += 1
                if fight_was_finish:
                    stats1["WinsByFinish"] += 1
            elif winner == "2":
                stats1["Losses"] += 1
                stats2["Wins"] += 1
                if fight_was_finish:
                    stats2["WinsByFinish"] += 1
            elif winner == "D":
                stats1["Draws"] += 1
                stats2["Draws"] += 1
            elif winner == "NC":
                stats1["NoContests"] += 1
                stats2["NoContests"] += 1

        # Final data cleaning
        print(f"Records before final dropping: {len(fight_data)}")
        fight_data = fight_data.dropna()
        print(f"Records after final dropping: {len(fight_data)}")
        fight_data.fillna(0, inplace=True)

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
            "Fighter1_Height_cm", "Fighter1_Reach_cm", "Fighter1_Age",
            "Fighter1_AvgFightTime", "Fighter1_TimeSinceLastFight", "Fighter1_FinishRate",
            "Fighter1_Wins", "Fighter1_Losses", "Fighter1_AvgControlTime",
            "Fighter1_AvgSubmissionAttempts", "Fighter1_AvgLegStrikes",
            "Fighter1_AvgClinchStrikes", "Fighter1_AvgStrikesLanded",
            "Fighter1_AvgStrikesAttempted", "Fighter1_StrikeAccuracy",
            "Fighter1_AvgTakedownsLanded", "Fighter1_AvgTakedownsAttempted",
            "Fighter1_AvgReversals", "Fighter2_Height_cm", "Fighter2_Reach_cm",
            "Fighter2_Age", "Fighter2_AvgFightTime", "Fighter2_TimeSinceLastFight",
            "Fighter2_FinishRate", "Fighter2_Wins", "Fighter2_Losses",
            "Fighter2_AvgControlTime", "Fighter2_AvgSubmissionAttempts",
            "Fighter2_AvgLegStrikes", "Fighter2_AvgClinchStrikes",
            "Fighter2_AvgStrikesLanded", "Fighter2_AvgStrikesAttempted",
            "Fighter2_StrikeAccuracy", "Fighter2_AvgTakedownsLanded",
            "Fighter2_AvgTakedownsAttempted", "Fighter2_AvgReversals",
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
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
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
        processed_feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

        # Combine features and target
        y_df = pd.Series(y, name="Winner")
        final_df = pd.concat([X_processed_df, y_df], axis=1)

        # Save processed data
        processed_data_path = os.path.join(DATA_PATH, "processed_ufc_data.csv")
        final_df.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Processed data saved to {processed_data_path}")

        print(f"UFC Data: {len(fight_data)} records loaded.")
        print(f"Date range: {fight_data['EventDate'].min()} to {fight_data['EventDate'].max()}")

        return X_processed, y, le

    except Exception as e:
        print(f"An error occurred while loading UFC data: {e}")
        raise
