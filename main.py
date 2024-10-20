# main.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from google.colab import drive
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_path = os.path.join('/content/drive/MyDrive/files/models/output', timestamp)
data_cache_path = os.path.join('/content/drive/MyDrive/files/models/data', 'cache')
os.makedirs(output_path, exist_ok=True)
os.makedirs(data_cache_path, exist_ok=True)
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load and preprocess ufc data
def load_ufc_data():
    try:
        ufc_data_path = '/content/drive/MyDrive/files/omscs/ML-CS7641/A1/data/ufc'
        fight_data_path = os.path.join(ufc_data_path, 'ufc_events.csv')

        fight_data = pd.read_csv(fight_data_path, quotechar='"', parse_dates=['EventDate'])

        print(f"Records before dropping data: {len(fight_data)}")
        fight_data = fight_data.drop(columns=['EventName'])
        fight_data = fight_data[~fight_data['Winner'].isin(['NC', 'D'])]
        print(f"Records after dropping data: {len(fight_data)}")

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

        def process_landed_attempted(column_name):
            landed_attempted = fight_data[column_name].apply(parse_strike)
            landed_col = column_name + '_Landed'
            attempted_col = column_name + '_Attempted'
            fight_data[landed_col] = [x[0] for x in landed_attempted]
            fight_data[attempted_col] = [x[1] for x in landed_attempted]

        def is_finish(method_str):
            if pd.isnull(method_str):
                return False
            return 'Decision' not in method_str

        # process height and reach
        fight_data['Fighter1_Height_cm'] = fight_data['Fighter1_Height'].apply(parse_height)
        fight_data['Fighter2_Height_cm'] = fight_data['Fighter2_Height'].apply(parse_height)

        fight_data['Fighter1_Reach_cm'] = fight_data['Fighter1_Reach'].apply(parse_reach)
        fight_data['Fighter2_Reach_cm'] = fight_data['Fighter2_Reach'].apply(parse_reach)

        # process control time
        fight_data['Fighter1_Control_Time_sec'] = fight_data['Fighter1_Control_Time'].apply(parse_control_time)
        fight_data['Fighter2_Control_Time_sec'] = fight_data['Fighter2_Control_Time'].apply(parse_control_time)

        # calculate age
        fight_data['Fighter1_Age'] = ((fight_data['EventDate'] - pd.to_datetime(fight_data['Fighter1_DOB'], errors='coerce')).dt.days / 365.25)
        fight_data['Fighter1_Age'] = fight_data['Fighter1_Age'].fillna(30)
        fight_data['Fighter2_Age'] = ((fight_data['EventDate'] - pd.to_datetime(fight_data['Fighter2_DOB'], errors='coerce')).dt.days / 365.25)
        fight_data['Fighter2_Age'] = fight_data['Fighter2_Age'].fillna(30)

        # process strikes
        strike_columns = [
            'Fighter1_Significant_Strikes', 'Fighter1_Head_Strikes', 'Fighter1_Body_Strikes',
            'Fighter1_Leg_Strikes', 'Fighter1_Distance_Strikes', 'Fighter1_Clinch_Strikes',
            'Fighter1_Ground_Strikes', 'Fighter1_Takedowns',
            'Fighter2_Significant_Strikes', 'Fighter2_Head_Strikes', 'Fighter2_Body_Strikes',
            'Fighter2_Leg_Strikes', 'Fighter2_Distance_Strikes', 'Fighter2_Clinch_Strikes',
            'Fighter2_Ground_Strikes', 'Fighter2_Takedowns'
        ]

        for col in strike_columns:
            process_landed_attempted(col)

        # calculate total fight time
        fight_data['Fight_Time_sec'] = fight_data.apply(
            lambda row: calculate_fight_time(row['Round'], row['Time']), axis=1
        )

        # initialize fighter statistics
        fighter_stats = defaultdict(lambda: {
            'TotalFightTime': 0,
            'NumFights': 0,
            'Wins': 0,
            'Losses': 0,
            'Draws': 0,
            'NoContests': 0,
            'WinsByFinish': 0,
            'LastFightDate': None,
            'TotalControlTime': 0,
            'TotalSubmissionAttempts': 0,
            'TotalLegStrikes': 0,
            'TotalClinchStrikes': 0,
            'TotalStrikesLanded': 0,
            'TotalStrikesAttempted': 0,
            'TotalTakedownsLanded': 0,
            'TotalTakedownsAttempted': 0,
            'TotalReversals': 0
        })

        # initialize additional columns
        for fighter_num in ['Fighter1', 'Fighter2']:
            fight_data[f'{fighter_num}_AvgFightTime'] = 0
            fight_data[f'{fighter_num}_TimeSinceLastFight'] = 0
            fight_data[f'{fighter_num}_FinishRate'] = 0
            fight_data[f'{fighter_num}_Wins'] = 0
            fight_data[f'{fighter_num}_Losses'] = 0
            fight_data[f'{fighter_num}_Draws'] = 0
            fight_data[f'{fighter_num}_NoContests'] = 0
            fight_data[f'{fighter_num}_AvgControlTime'] = 0
            fight_data[f'{fighter_num}_AvgSubmissionAttempts'] = 0
            fight_data[f'{fighter_num}_AvgLegStrikes'] = 0
            fight_data[f'{fighter_num}_AvgClinchStrikes'] = 0
            fight_data[f'{fighter_num}_AvgStrikesLanded'] = 0
            fight_data[f'{fighter_num}_AvgStrikesAttempted'] = 0
            fight_data[f'{fighter_num}_StrikeAccuracy'] = 0
            fight_data[f'{fighter_num}_AvgTakedownsLanded'] = 0
            fight_data[f'{fighter_num}_AvgTakedownsAttempted'] = 0
            fight_data[f'{fighter_num}_AvgReversals'] = 0

        # sort the fight_data by 'eventdate' to ensure chronological order
        fight_data = fight_data.sort_values('EventDate').reset_index(drop=True)

        # iterate through each fight to update fighter statistics
        for idx, row in fight_data.iterrows():
            fight_time = row['Fight_Time_sec']
            event_date = row['EventDate']

            # for each fighter, update fight_data with stats before the current fight
            for fighter_num, opponent_num in [('Fighter1', 'Fighter2'), ('Fighter2', 'Fighter1')]:
                fighter_id = row[f'{fighter_num}_ID']
                stats = fighter_stats[fighter_id]
                stats_before = stats.copy()

                # use the stats as of before this fight to fill in fight_data
                if stats_before['NumFights'] > 0:
                    fight_data.at[idx, f'{fighter_num}_AvgFightTime'] = stats_before['TotalFightTime'] / stats_before['NumFights']
                    if stats_before['LastFightDate'] is not None:
                        days_since_last_fight = (event_date - stats_before['LastFightDate']).days
                        fight_data.at[idx, f'{fighter_num}_TimeSinceLastFight'] = days_since_last_fight
                    else:
                        fight_data.at[idx, f'{fighter_num}_TimeSinceLastFight'] = 0

                    if stats_before['Wins'] > 0:
                        fight_data.at[idx, f'{fighter_num}_FinishRate'] = stats_before['WinsByFinish'] / stats_before['Wins']
                    else:
                        fight_data.at[idx, f'{fighter_num}_FinishRate'] = 0

                    fight_data.at[idx, f'{fighter_num}_AvgControlTime'] = (
                        stats_before['TotalControlTime'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgSubmissionAttempts'] = (
                        stats_before['TotalSubmissionAttempts'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgLegStrikes'] = (
                        stats_before['TotalLegStrikes'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgClinchStrikes'] = (
                        stats_before['TotalClinchStrikes'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesLanded'] = (
                        stats_before['TotalStrikesLanded'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesAttempted'] = (
                        stats_before['TotalStrikesAttempted'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_StrikeAccuracy'] = (
                        stats_before['TotalStrikesLanded'] / stats_before['TotalStrikesAttempted']
                        if stats_before['TotalStrikesAttempted'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsLanded'] = (
                        stats_before['TotalTakedownsLanded'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsAttempted'] = (
                        stats_before['TotalTakedownsAttempted'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgReversals'] = (
                        stats_before['TotalReversals'] / stats_before['NumFights'] if stats_before['NumFights'] > 0 else 0
                    )
                else:
                    # handle case where fighter has no previous fights
                    fight_data.at[idx, f'{fighter_num}_AvgFightTime'] = 0
                    fight_data.at[idx, f'{fighter_num}_TimeSinceLastFight'] = 0
                    fight_data.at[idx, f'{fighter_num}_FinishRate'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgControlTime'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgSubmissionAttempts'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgLegStrikes'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgClinchStrikes'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesLanded'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesAttempted'] = 0
                    fight_data.at[idx, f'{fighter_num}_StrikeAccuracy'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsLanded'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsAttempted'] = 0
                    fight_data.at[idx, f'{fighter_num}_AvgReversals'] = 0

                # update win/loss/draw counts before the current fight
                fight_data.at[idx, f'{fighter_num}_Wins'] = stats_before['Wins']
                fight_data.at[idx, f'{fighter_num}_Losses'] = stats_before['Losses']
                fight_data.at[idx, f'{fighter_num}_Draws'] = stats_before['Draws']
                fight_data.at[idx, f'{fighter_num}_NoContests'] = stats_before['NoContests']

            # after updating fight_data, update stats with the current fight's data
            for fighter_num in ['Fighter1', 'Fighter2']:
                fighter_id = row[f'{fighter_num}_ID']
                stats = fighter_stats[fighter_id]

                stats['TotalFightTime'] += fight_time if not np.isnan(fight_time) else 0
                stats['NumFights'] += 1
                stats['LastFightDate'] = event_date

                control_time = row[f'{fighter_num}_Control_Time_sec']
                stats['TotalControlTime'] += control_time if not np.isnan(control_time) else 0

                sub_attempts = float(row[f'{fighter_num}_Submission_Attempts'])
                stats['TotalSubmissionAttempts'] += sub_attempts if not np.isnan(sub_attempts) else 0

                leg_strikes = row[f'{fighter_num}_Leg_Strikes_Landed']
                stats['TotalLegStrikes'] += leg_strikes if not np.isnan(leg_strikes) else 0

                clinch_strikes = row[f'{fighter_num}_Clinch_Strikes_Landed']
                stats['TotalClinchStrikes'] += clinch_strikes if not np.isnan(clinch_strikes) else 0

                strikes_landed = row[f'{fighter_num}_Significant_Strikes_Landed']
                strikes_attempted = row[f'{fighter_num}_Significant_Strikes_Attempted']
                stats['TotalStrikesLanded'] += strikes_landed if not np.isnan(strikes_landed) else 0
                stats['TotalStrikesAttempted'] += strikes_attempted if not np.isnan(strikes_attempted) else 0

                # handle takedowns
                if f'{fighter_num}_Takedowns_Landed' in row:
                    takedowns_landed = row[f'{fighter_num}_Takedowns_Landed']
                    takedowns_attempted = row[f'{fighter_num}_Takedowns_Attempted']
                else:
                    takedowns_landed = row[f'{fighter_num}_Takedowns']
                    takedowns_attempted = row[f'{fighter_num}_Takedowns']

                stats['TotalTakedownsLanded'] += takedowns_landed if not np.isnan(takedowns_landed) else 0
                stats['TotalTakedownsAttempted'] += takedowns_attempted if not np.isnan(takedowns_attempted) else 0

                reversals = float(row[f'{fighter_num}_Reversals'])
                stats['TotalReversals'] += reversals if not np.isnan(reversals) else 0

            # update win/loss/draw statistics after updating stats with the current fight
            winner = str(row['Winner'])
            method = str(row['Method'])
            stats1 = fighter_stats[row['Fighter1_ID']]
            stats2 = fighter_stats[row['Fighter2_ID']]
            fight_was_finish = is_finish(method)
            if winner == '1':
                stats1['Wins'] += 1
                stats2['Losses'] += 1
                if fight_was_finish:
                    stats1['WinsByFinish'] += 1
            elif winner == '2':
                stats1['Losses'] += 1
                stats2['Wins'] += 1
                if fight_was_finish:
                    stats2['WinsByFinish'] += 1
            elif winner == 'D':
                stats1['Draws'] += 1
                stats2['Draws'] += 1
            elif winner == 'NC':
                stats1['NoContests'] += 1
                stats2['NoContests'] += 1

        processed_data_path = os.path.join(data_cache_path, 'fight_data_with_stats.csv')
        fight_data.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Fight data with statistics saved to {processed_data_path}")

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

        processed_data_path = os.path.join(data_cache_path, 'ufc_data.csv')
        fight_data.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Actual data saved to {processed_data_path}")

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

        processed_data_path = os.path.join(data_cache_path, 'relevant_ufc_data.csv')
        X.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Relevant data saved to {processed_data_path}")

        # label encoding for target
        le = LabelEncoder()
        y = le.fit_transform(y)
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
        y_df = pd.Series(y, name='Winner')

        # combine features and target into a single dataframe
        final_df = pd.concat([X_processed_df, y_df], axis=1)

        # save the processed data to csv
        processed_data_path = os.path.join(data_cache_path, 'processed_ufc_data.csv')
        final_df.to_csv(processed_data_path, index=False, quotechar='"')
        print(f"Processed data saved to {processed_data_path}")

        print(f"UFC Data: {len(fight_data)} records loaded.")
        print(f"Date range: {fight_data['EventDate'].min()} to {fight_data['EventDate'].max()}")

        return X_processed, y, preprocessor, fight_data, relevant_columns, le

    except Exception as e:
        print(f"An error occurred while loading UFC data: {e}")
        raise

if __name__ == "__main__":

    print("Loading UFC data...")
    X_ufc, y_ufc, preprocessor, fight_data, relevant_columns, le = load_ufc_data()
    if X_ufc is None:
        print("Failed to load UFC data.")
        exit(1)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_ufc, y_ufc, test_size=0.2, random_state=42, stratify=y_ufc
    )

    # initialize a dictionary to store model performances
    model_performances = {}

    # 1. random forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    model_performances['Random Forest'] = accuracy_rf

    # 2. gradient boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    model_performances['Gradient Boosting'] = accuracy_gb

    # 3. support vector machine
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    model_performances['Support Vector Machine'] = accuracy_svm

    # 4. logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    model_performances['Logistic Regression'] = accuracy_lr

    # 5. k-nearest neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    model_performances['K-Nearest Neighbors'] = accuracy_knn

    # 6. neural network (pytorch)
    class UFCNet(nn.Module):
        def __init__(self, input_size):
            super(UFCNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # convert data to pytorch tensors
    X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(device)
    y_train_tensor = torch.tensor(y_train.astype(np.longlong)).to(device)
    X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(device)
    y_test_tensor = torch.tensor(y_test.astype(np.longlong)).to(device)

    input_size = X_train.shape[1]
    model = UFCNet(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    # training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # evaluation
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy_nn = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Neural Network Accuracy: {accuracy_nn * 100:.2f}%')

    # 7. naive bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    model_performances['Naive Bayes'] = accuracy_nb

    # output model performances
    performance_df = pd.DataFrame(list(model_performances.items()), columns=['Model', 'Accuracy'])
    performance_df.to_csv(os.path.join(output_path, 'model_performances.csv'), index=False)
    plt.figure(figsize=(10, 6))
    plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.xlim([0, 1])  # Assuming accuracies are in the range of 0 to 1
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'model_accuracy_comparison.png'))
    plt.show()
    print(performance_df)

    # plotting confusion matrices
    models = {
        'Random Forest': (rf, y_pred_rf),
        'Gradient Boosting': (gb, y_pred_gb),
        'Support Vector Machine': (svm, y_pred_svm),
        'Logistic Regression': (lr, y_pred_lr),
        'K-Nearest Neighbors': (knn, y_pred_knn),
        'Neural Network': (model, predicted.numpy()),
        'Naive Bayes': (nb, y_pred_nb)
    }

    for model_name, (model_obj, y_pred) in models.items():
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix for {model_name}')
        plt.savefig(os.path.join(output_path, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
        plt.close()

    print(f"All tasks completed. Results and plots saved in {output_path}.")
