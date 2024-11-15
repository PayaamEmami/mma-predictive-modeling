# data.py
import pandas as pd
import os
from preprocessing import preprocess_fight_data, initialize_fighter_stats, update_fighter_statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import plot_confusion_matrix
from collections import defaultdict
import traceback

def load_ufc_data(data_path):
    try:
        print("Loading UFC data...")
        fight_data_path = os.path.join(data_path, 'ufc', 'ufc_events.csv')

        fight_data = pd.read_csv(fight_data_path, quotechar='"', parse_dates=['EventDate'])
        print(f"Records before dropping data: {len(fight_data)}")

        # preprocess fight data
        fight_data = preprocess_fight_data(fight_data)
        print(f"Records after initial preprocessing: {len(fight_data)}")

        # initialize fighter statistics
        fighter_stats = initialize_fighter_stats()

        # update fighter statistics based on fight outcomes
        fight_data = update_fighter_statistics(fight_data, fighter_stats)

        # final data cleaning
        print(f"Records before final dropping: {len(fight_data)}")
        fight_data = fight_data.dropna()
        print(f"Records after final dropping: {len(fight_data)}")
        fight_data.fillna(0, inplace=True)

        # check for class balance
        winner_counts = fight_data['Winner'].value_counts()
        print("Class distribution of 'Winner':")
        print(winner_counts)

        total_fights = len(fight_data)
        fighter1_win_percentage = (winner_counts.get('1', 0) / total_fights) * 100
        fighter2_win_percentage = (winner_counts.get('2', 0) / total_fights) * 100

        print(f"Fighter 1 wins: {fighter1_win_percentage:.2f}%")
        print(f"Fighter 2 wins: {fighter2_win_percentage:.2f}%")

        # define feature columns
        numerical_columns = [
            'Fighter1_Height_cm', 'Fighter1_Reach_cm',
            'Fighter1_Age', 'Fighter1_AvgFightTime', 'Fighter1_TimeSinceLastFight', 'Fighter1_FinishRate',
            'Fighter1_Wins', 'Fighter1_Losses',
            'Fighter1_AvgControlTime', 'Fighter1_AvgSubmissionAttempts', 'Fighter1_AvgLegStrikes',
            'Fighter1_AvgClinchStrikes', 'Fighter1_AvgStrikesLanded', 'Fighter1_AvgStrikesAttempted',
            'Fighter1_StrikeAccuracy', 'Fighter1_AvgTakedownsLanded', 'Fighter1_AvgTakedownsAttempted',
            'Fighter1_AvgReversals',
            'Fighter2_Height_cm', 'Fighter2_Reach_cm',
            'Fighter2_Age', 'Fighter2_AvgFightTime', 'Fighter2_TimeSinceLastFight', 'Fighter2_FinishRate',
            'Fighter2_Wins', 'Fighter2_Losses',
            'Fighter2_AvgControlTime', 'Fighter2_AvgSubmissionAttempts', 'Fighter2_AvgLegStrikes',
            'Fighter2_AvgClinchStrikes', 'Fighter2_AvgStrikesLanded', 'Fighter2_AvgStrikesAttempted',
            'Fighter2_StrikeAccuracy', 'Fighter2_AvgTakedownsLanded', 'Fighter2_AvgTakedownsAttempted',
            'Fighter2_AvgReversals'
        ]

        categorical_columns = ['Fighter1_Stance', 'Fighter2_Stance']

        relevant_columns = numerical_columns + categorical_columns

        # features and target
        X = fight_data[relevant_columns]
        y = fight_data['Winner']

        # label encoding for target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("Label Mapping:", label_mapping)

        # preprocessing pipelines
        numerical_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_columns),
                ('cat', categorical_pipeline, categorical_columns)
            ]
        )

        # fit and transform the data
        X_processed = preprocessor.fit_transform(X)

        # retrieve feature names after transformation
        processed_feature_names = preprocessor.get_feature_names_out()

        # convert x_processed to a dataframe
        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

        # include the target variable
        y_df = pd.Series(y_encoded, name='Winner')

        # combine features and target into a single dataframe
        final_df = pd.concat([X_processed_df, y_df], axis=1)

        return X_processed, y_encoded, le, preprocessor

    except Exception as e:
        print(f"An error occurred while loading UFC data: {e}")
        traceback.print_exc()
        raise
