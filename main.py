# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pickle
import time
from google.colab import drive
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from itertools import product
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_path = os.path.join('/content/drive/MyDrive/files/models/output', timestamp)
data_cache_path = os.path.join('/content/drive/MyDrive/files/models/data', 'cache')
os.makedirs(output_path, exist_ok=True)
os.makedirs(data_cache_path, exist_ok=True)
drive.mount('/content/drive')

def load_ufc_data():
    try:
        ufc_data_path = '/content/drive/MyDrive/files/models/data/ufc'
        fight_data_path = os.path.join(ufc_data_path, 'ufc_events.csv')

        fight_data = pd.read_csv(fight_data_path, quotechar='"', parse_dates=['EventDate'])

        print(f"Records before dropping data: {len(fight_data)}")
        fight_data = fight_data.drop(columns=['EventName'])
        fight_data = fight_data[~fight_data['Winner'].isin(['NC', 'D'])]
        fight_data = fight_data.dropna()
        print(f"Records after dropping data: {len(fight_data)}")

        def parse_height(height_str):
            if pd.isnull(height_str):
                return np.nan
            pattern = r"(?:(\d+)ft)?\s*(\d+)in"
            match = re.match(pattern, height_str)
            if match:
                feet = int(match.group(1)) if match.group(1) else 0
                inches = int(match.group(2))
                total_inches = feet * 12 + inches
                return total_inches * 2.54
            else:
                return np.nan

        def parse_reach(reach_str):
            if pd.isnull(reach_str):
                return np.nan
            pattern = r"(\d+)in"
            match = re.match(pattern, reach_str)
            if match:
                inches = int(match.group(1))
                return inches * 2.54
            else:
                return np.nan

        def parse_strike(strike_str):
            if pd.isnull(strike_str):
                return [np.nan, np.nan]
            pattern = r"(\d+)\s+of\s+(\d+)"
            match = re.match(pattern, strike_str)
            if match:
                landed = int(match.group(1))
                attempted = int(match.group(2))
                return [landed, attempted]
            else:
                return [np.nan, np.nan]

        def parse_control_time(time_str):
            if pd.isnull(time_str):
                return np.nan
            pattern = r"(\d+):(\d+)"
            match = re.match(pattern, time_str)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                total_seconds = minutes * 60 + seconds
                return total_seconds
            else:
                return np.nan

        def parse_time(time_str):
            if pd.isnull(time_str):
                return np.nan
            pattern = r"(\d+):(\d+)"
            match = re.match(pattern, time_str)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                total_seconds = minutes * 60 + seconds
                return total_seconds
            else:
                return np.nan

        def calculate_fight_time(round_num, time_str):
            if pd.isnull(round_num) or pd.isnull(time_str):
                return np.nan
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

        fight_data['Fighter1_Height_cm'] = fight_data['Fighter1_Height'].apply(parse_height)
        fight_data['Fighter2_Height_cm'] = fight_data['Fighter2_Height'].apply(parse_height)

        fight_data['Fighter1_Reach_cm'] = fight_data['Fighter1_Reach'].apply(parse_reach)
        fight_data['Fighter2_Reach_cm'] = fight_data['Fighter2_Reach'].apply(parse_reach)

        fight_data['Fighter1_Control_Time_sec'] = fight_data['Fighter1_Control_Time'].apply(parse_control_time)
        fight_data['Fighter2_Control_Time_sec'] = fight_data['Fighter2_Control_Time'].apply(parse_control_time)

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

        fight_data['Fighter1_Age'] = (fight_data['EventDate'] - pd.to_datetime(fight_data['Fighter1_DOB'])).dt.days / 365.25
        fight_data['Fighter2_Age'] = (fight_data['EventDate'] - pd.to_datetime(fight_data['Fighter2_DOB'])).dt.days / 365.25

        fight_data['Fight_Time_sec'] = fight_data.apply(
            lambda row: calculate_fight_time(row['Round'], row['Time']), axis=1
        )

        fighter_stats = defaultdict(lambda: {
            'TotalFightTime': 0,
            'NumFights': 0,
            'Wins': 0,
            'Losses': 0,
            'Draws': 0,
            'NoContests': 0,
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

        for fighter_num in ['Fighter1', 'Fighter2']:
            fight_data[f'{fighter_num}_AvgFightTime'] = np.nan
            fight_data[f'{fighter_num}_TimeSinceLastFight'] = np.nan
            fight_data[f'{fighter_num}_FinishRate'] = np.nan
            fight_data[f'{fighter_num}_Wins'] = 0
            fight_data[f'{fighter_num}_Losses'] = 0
            fight_data[f'{fighter_num}_Draws'] = 0
            fight_data[f'{fighter_num}_NoContests'] = 0
            fight_data[f'{fighter_num}_AvgControlTime'] = np.nan
            fight_data[f'{fighter_num}_AvgSubmissionAttempts'] = np.nan
            fight_data[f'{fighter_num}_AvgLegStrikes'] = np.nan
            fight_data[f'{fighter_num}_AvgClinchStrikes'] = np.nan
            fight_data[f'{fighter_num}_AvgStrikesLanded'] = np.nan
            fight_data[f'{fighter_num}_AvgStrikesAttempted'] = np.nan
            fight_data[f'{fighter_num}_StrikeAccuracy'] = np.nan
            fight_data[f'{fighter_num}_AvgTakedownsLanded'] = np.nan
            fight_data[f'{fighter_num}_AvgTakedownsAttempted'] = np.nan
            fight_data[f'{fighter_num}_AvgReversals'] = np.nan

        for idx, row in fight_data.iterrows():
            fight_time = row['Fight_Time_sec']

            for fighter_num, opponent_num in [('Fighter1', 'Fighter2'), ('Fighter2', 'Fighter1')]:
                fighter_id = row[f'{fighter_num}_ID']
                stats = fighter_stats[fighter_id]

                stats['TotalFightTime'] += fight_time if not np.isnan(fight_time) else 0
                stats['NumFights'] += 1
                stats['LastFightDate'] = row['EventDate']

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

                takedowns_landed = row[f'{fighter_num}_Takedowns_Landed'] if f'{fighter_num}_Takedowns_Landed' in fight_data.columns else row[f'{fighter_num}_Takedowns']
                stats['TotalTakedownsLanded'] += takedowns_landed if not np.isnan(takedowns_landed) else 0
                stats['TotalTakedownsAttempted'] += takedowns_landed if not np.isnan(takedowns_landed) else 0

                reversals = float(row[f'{fighter_num}_Reversals'])
                stats['TotalReversals'] += reversals if not np.isnan(reversals) else 0

            winner = str(row['Winner'])
            stats1 = fighter_stats[row['Fighter1_ID']]
            stats2 = fighter_stats[row['Fighter2_ID']]
            if winner == '1':
                stats1['Wins'] += 1
                stats2['Losses'] += 1
            elif winner == '2':
                stats1['Losses'] += 1
                stats2['Wins'] += 1
            elif winner == 'D':
                stats1['Draws'] += 1
                stats2['Draws'] += 1
            elif winner == 'NC':
                stats1['NoContests'] += 1
                stats2['NoContests'] += 1

            for fighter_num, opponent_num in [('Fighter1', 'Fighter2'), ('Fighter2', 'Fighter1')]:
                fighter_id = row[f'{fighter_num}_ID']
                stats = fighter_stats[fighter_id]

                if stats['NumFights'] > 0:
                    fight_data.at[idx, f'{fighter_num}_AvgFightTime'] = stats['TotalFightTime'] / stats['NumFights']
                    if stats['LastFightDate'] is not None:
                        days_since_last_fight = (row['EventDate'] - stats['LastFightDate']).days
                        fight_data.at[idx, f'{fighter_num}_TimeSinceLastFight'] = days_since_last_fight
                    else:
                        fight_data.at[idx, f'{fighter_num}_TimeSinceLastFight'] = np.nan

                    fight_data.at[idx, f'{fighter_num}_FinishRate'] = (
                        (stats['Wins'] + stats['Losses'] - stats['NoContests'] - stats['Draws']) / stats['NumFights']
                        if stats['NumFights'] > 0 else np.nan
                    )

                    fight_data.at[idx, f'{fighter_num}_AvgControlTime'] = (
                        stats['TotalControlTime'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgSubmissionAttempts'] = (
                        stats['TotalSubmissionAttempts'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgLegStrikes'] = (
                        stats['TotalLegStrikes'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgClinchStrikes'] = (
                        stats['TotalClinchStrikes'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesLanded'] = (
                        stats['TotalStrikesLanded'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgStrikesAttempted'] = (
                        stats['TotalStrikesAttempted'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_StrikeAccuracy'] = (
                        stats['TotalStrikesLanded'] / stats['TotalStrikesAttempted']
                        if stats['TotalStrikesAttempted'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsLanded'] = (
                        stats['TotalTakedownsLanded'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgTakedownsAttempted'] = (
                        stats['TotalTakedownsAttempted'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgReversals'] = (
                        stats['TotalReversals'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )
                    fight_data.at[idx, f'{fighter_num}_AvgReversals'] = (
                        stats['TotalReversals'] / stats['NumFights'] if stats['NumFights'] > 0 else np.nan
                    )

                fight_data.at[idx, f'{fighter_num}_Wins'] = stats['Wins']
                fight_data.at[idx, f'{fighter_num}_Losses'] = stats['Losses']
                fight_data.at[idx, f'{fighter_num}_Draws'] = stats['Draws']
                fight_data.at[idx, f'{fighter_num}_NoContests'] = stats['NoContests']

        fight_data.fillna(0, inplace=True)

        numerical_columns = [
            'Fighter1_Height_cm', 'Fighter1_Reach_cm',
            'Fighter1_Age', 'Fighter1_AvgFightTime', 'Fighter1_TimeSinceLastFight', 'Fighter1_FinishRate',
            'Fighter1_Wins', 'Fighter1_Losses', 'Fighter1_Draws', 'Fighter1_NoContests',
            'Fighter1_AvgControlTime', 'Fighter1_AvgSubmissionAttempts', 'Fighter1_AvgLegStrikes',
            'Fighter1_AvgClinchStrikes', 'Fighter1_AvgStrikesLanded', 'Fighter1_AvgStrikesAttempted',
            'Fighter1_StrikeAccuracy', 'Fighter1_AvgTakedownsLanded', 'Fighter1_AvgTakedownsAttempted',
            'Fighter1_AvgReversals',
            'Fighter2_Height_cm', 'Fighter2_Reach_cm',
            'Fighter2_Age', 'Fighter2_AvgFightTime', 'Fighter2_TimeSinceLastFight', 'Fighter2_FinishRate',
            'Fighter2_Wins', 'Fighter2_Losses', 'Fighter2_Draws', 'Fighter2_NoContests',
            'Fighter2_AvgControlTime', 'Fighter2_AvgSubmissionAttempts', 'Fighter2_AvgLegStrikes',
            'Fighter2_AvgClinchStrikes', 'Fighter2_AvgStrikesLanded', 'Fighter2_AvgStrikesAttempted',
            'Fighter2_StrikeAccuracy', 'Fighter2_AvgTakedownsLanded', 'Fighter2_AvgTakedownsAttempted',
            'Fighter2_AvgReversals'
        ]

        categorical_columns = ['Fighter1_Stance', 'Fighter2_Stance', 'WeightClass']

        relevant_columns = numerical_columns + categorical_columns

        X = fight_data[relevant_columns]
        y = fight_data['Winner']

        le = LabelEncoder()
        y = le.fit_transform(y)

        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("Label Mapping:", label_mapping)

        numerical_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_columns),
                ('cat', categorical_pipeline, categorical_columns)
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        print(f"UFC Data: {len(fight_data)} records loaded.")
        print(f"Date range: {fight_data['EventDate'].min()} to {fight_data['EventDate'].max()}")

        return X_processed, y, preprocessor, fight_data, relevant_columns

    except Exception as e:
        print(f"An error occurred while loading UFC data: {e}")
        raise

def save_model(model, model_name, dataset_name, results_dir):
    model_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(model_name, dataset_name, results_dir):
    model_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def train_fully_connected_neural_network(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    with open(results_file, 'a+') as f:
        for hidden_layer_sizes in hyperparameters_range['hidden_layer_sizes']:
            for alpha in hyperparameters_range['alpha']:
                for solver in hyperparameters_range['solver']:
                    for max_iter in hyperparameters_range['max_iter']:
                        start_time = time.time()

                        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, solver=solver, max_iter=max_iter, random_state=21, probability=True)
                        model.fit(X_train, y_train)
                        score = evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir)

                        end_time = time.time()
                        training_time = end_time - start_time

                        f.write(f"Training time for hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}, solver={solver}, max_iter={max_iter}: {training_time:.2f} seconds. F1-Score: {score}.\n")
                        print(f"Training time for hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}, solver={solver}, max_iter={max_iter} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {score}.")

                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = {'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha, 'solver': solver, 'max_iter': max_iter}

        f.write(f"\nBest Neural Network params: {best_params} with F1-Score: {best_score}\n")
    save_model(best_model, model_name, dataset_name, results_dir)
    return best_model

def train_deep_neural_network(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    num_classes = np.unique(y_train).size
    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)

    with open(results_file, 'a+') as f:
        param_names = list(hyperparameters_range.keys())
        param_values = list(hyperparameters_range.values())

        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            start_time = time.time()

            model = Sequential()
            for layer_size in param_dict['hidden_layer_sizes']:
                model.add(Dense(layer_size, activation=param_dict['activation']))
                if 'dropout_rate' in param_dict and param_dict['dropout_rate'] > 0:
                    model.add(Dropout(param_dict['dropout_rate']))
            model.add(Dense(num_classes, activation='softmax'))

            optimizer_name = param_dict['optimizer']
            learning_rate = param_dict['learning_rate']
            if optimizer_name == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = SGD(learning_rate=learning_rate)
            else:
                optimizer = optimizer_name

            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            callbacks = []
            if param_dict.get('early_stopping', False):
                callbacks.append(EarlyStopping(patience=5, restore_best_weights=True))

            history = model.fit(X_train, y_train_categorical,
                                validation_data=(X_test, y_test_categorical),
                                epochs=param_dict['epochs'],
                                batch_size=param_dict['batch_size'],
                                callbacks=callbacks,
                                verbose=0)

            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            f1 = f1_score(y_test, y_pred, average='weighted')

            end_time = time.time()
            training_time = end_time - start_time

            f.write(f"Training time with parameters {param_dict}: {training_time:.2f} seconds. F1-Score: {f1}.\n")
            print(f"Training time with parameters {param_dict} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {f1}.")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = param_dict

        f.write(f"\nBest DNN params: {best_params} with F1-Score: {best_score}\n")

    model_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.h5")
    best_model.save(model_path)
    print(f"Model saved to {model_path}")
    return best_model


def train_support_vector_machine(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    with open(results_file, 'a+') as f:
        for kernel in hyperparameters_range['kernel']:
            for C in hyperparameters_range['C']:
                for gamma in hyperparameters_range['gamma']:
                    start_time = time.time()

                    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
                    model.fit(X_train, y_train)
                    score = evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir)

                    end_time = time.time()
                    training_time = end_time - start_time

                    f.write(f"Training time for kernel={kernel}, C={C}, gamma={gamma}: {training_time:.2f} seconds. F1-Score: {score}.\n")
                    print(f"Training time for kernel={kernel}, C={C}, gamma={gamma} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {score}.")

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

        f.write(f"\nBest SVM params: {best_params} with F1-Score: {best_score}\n")
    save_model(best_model, model_name, dataset_name, results_dir)
    return best_model

def train_k_nearest_neighbor(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    param_names = hyperparameters_range.keys()
    param_values = hyperparameters_range.values()

    with open(results_file, 'a+') as f:
        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))
            start_time = time.time()

            model = KNeighborsClassifier(**param_dict, probability=True)
            model.fit(X_train, y_train)
            score = evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir)

            end_time = time.time()
            training_time = end_time - start_time

            f.write(f"Training time with parameters {param_dict}: {training_time:.2f} seconds. F1-Score: {score}.\n")
            print(f"Training time with parameters {param_dict} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {score}.")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = param_dict

        f.write(f"\nBest KNN params: {best_params} with F1-Score: {best_score}\n")
    save_model(best_model, model_name, dataset_name, results_dir)
    return best_model

def train_decision_tree(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    with open(results_file, 'a+') as f:
        for max_depth in hyperparameters_range['max_depth']:
            for min_samples_split in hyperparameters_range['min_samples_split']:
                for min_samples_leaf in hyperparameters_range['min_samples_leaf']:
                    start_time = time.time()

                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=21, probability=True)
                    model.fit(X_train, y_train)
                    score = evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir)

                    end_time = time.time()
                    training_time = end_time - start_time

                    f.write(f"Training time for max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}: {training_time:.2f} seconds. F1-Score: {score}.\n")
                    print(f"Training time for max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {score}.")

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

        f.write(f"\nBest Decision Tree params: {best_params} with F1-Score: {best_score}\n")
    save_model(best_model, model_name, dataset_name, results_dir)
    return best_model

def train_boosting(X_train, y_train, X_test, y_test, hyperparameters_range, model_name, dataset_name, results_dir):
    best_model = None
    best_score = 0
    best_params = {}
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    with open(results_file, 'a+') as f:
        for n_estimators in hyperparameters_range['n_estimators']:
            for learning_rate in hyperparameters_range['learning_rate']:
                start_time = time.time()

                estimator = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, max_leaf_nodes=10)
                model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, probability=True)
                model.fit(X_train, y_train)
                score = evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir)

                end_time = time.time()
                training_time = end_time - start_time

                f.write(f"Training time for n_estimators={n_estimators}, learning_rate={learning_rate}: {training_time:.2f} seconds. F1-Score: {score}.\n")
                print(f"Training time for n_estimators={n_estimators}, learning_rate={learning_rate} on {dataset_name}: {training_time:.2f} seconds. F1-Score: {score}.")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

        f.write(f"\nBest AdaBoost params: {best_params} with F1-Score: {best_score}\n")
    save_model(best_model, model_name, dataset_name, results_dir)
    return best_model

def evaluate_model(model, X_test, y_test, model_name, dataset_name, results_dir):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    f1_average = 'weighted'
    f1 = f1_score(y_test, y_pred, average=f1_average)
    report = classification_report(y_test, y_pred)
    results_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

    with open(results_file, 'a+') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"F1-Score ({f1_average}): {f1}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Probabilities:\n{y_prob}\n")

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"F1-Score ({f1_average}): {f1}")
    print(f"Classification Report:\n{report}")
    print(f"Probabilities:\n{y_prob}")
    return f1, report, y_prob

def generate_plots(estimator, X, y, param_grid, title_prefix, results_dir, cv):
    try:
        param_labels = {
            'hidden_layer_sizes': 'Hidden Layer Sizes',
            'alpha': 'Regularization Parameter (Alpha)',
            'solver': 'Solver',
            'max_iter': 'Maximum Iterations',
            'activation': 'Activation Function',
            'learning_rate_init': 'Initial Learning Rate',
            'early_stopping': 'Early Stopping',
            'num_conv_layers': 'Number of Convolutional Layers',
            'filters': 'Number of Filters',
            'kernel_size': 'Kernel Size',
            'optimizer': 'Optimizer',
            'learning_rate': 'Learning Rate',
            'batch_size': 'Batch Size',
            'epochs': 'Number of Epochs',
            'dropout_rate': 'Dropout Rate',
            'C': 'Regularization Parameter (C)',
            'gamma': 'Kernel Coefficient (Gamma)',
            'kernel': 'Kernel Type',
            'class_weight': 'Class Weight',
            'decision_function_shape': 'Decision Function Shape',
            'n_neighbors': 'Number of Neighbors (k)',
            'weights': 'Weight Function',
            'algorithm': 'Algorithm Used for Nearest Neighbors',
            'p': 'Power Parameter for Minkowski Metric',
            'metric': 'Distance Metric',
            'criterion': 'Criterion',
            'max_depth': 'Maximum Depth',
            'min_samples_split': 'Minimum Samples to Split',
            'min_samples_leaf': 'Minimum Samples per Leaf',
            'max_features': 'Maximum Features',
            'class_weight': 'Class Weight',
            'max_leaf_nodes': 'Maximum Leaf Nodes',
            'min_impurity_decrease': 'Minimum Impurity Decrease',
            'n_estimators': 'Number of Estimators',
            'learning_rate': 'Learning Rate',
            'algorithm': 'Boosting Algorithm',
            'base_estimator__max_depth': 'Base Estimator Maximum Depth',
            'base_estimator__min_samples_split': 'Base Estimator Minimum Samples to Split',
            'base_estimator__min_samples_leaf': 'Base Estimator Minimum Samples per Leaf',
            'random_state': 'Random State',
            'subsample': 'Subsample Ratio'
        }
        training_times = []
        start_time = time.time()
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_weighted'
        )
        end_time = time.time()
        learning_curve_time = end_time - start_time
        training_times.append(('Learning Curve', learning_curve_time))
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.title(f"{title_prefix} Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training F1 Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation F1 Score")
        plt.legend(loc="best")
        plt.savefig(os.path.join(results_dir, f'{title_prefix.lower().replace(" ", "_")}_learning_curve.png'))
        plt.close()

        for param_name, param_range in param_grid.items():
            start_time = time.time()
            train_scores, test_scores = validation_curve(
                estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1, scoring='f1_weighted'
            )
            end_time = time.time()
            validation_curve_time = end_time - start_time
            training_times.append((f'Validation Curve for {param_name}', validation_curve_time))
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            plt.figure()
            plt.title(f"{title_prefix} Validation Curve for {param_labels.get(param_name, param_name)}")
            plt.xlabel(param_labels.get(param_name, param_name))
            plt.ylabel("F1 Score")
            plt.grid()
            param_range_str = [str(val) for val in param_range] if isinstance(param_range[0], (str, tuple)) else param_range
            plt.plot(param_range_str, train_scores_mean, 'o-', color='r', label="Training F1 Score")
            plt.plot(param_range_str, test_scores_mean, 'o-', color='g', label="Cross-validation F1 Score")
            plt.legend(loc="best")
            plt.savefig(os.path.join(results_dir, f'{title_prefix.lower().replace(" ", "_")}_validation_curve_{param_name}.png'))
            plt.close()

        with open(os.path.join(results_dir, f"{title_prefix.replace(' ', '_')}_training_times.txt"), 'w') as log_file:
            for desc, t in training_times:
                log_file.write(f"{desc}: {t:.2f} seconds\n")
                print(f"{desc}: {t:.2f} seconds")

    except Exception as e:
        print(f"An error occurred while generating plots: {e}")

if __name__ == "__main__":

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

    nn_hyperparameters_range_ufc = {
        'hidden_layer_sizes': [
            (25,), (50,), (64,), (128,), (256,), (512,),
            (100, 50), (150, 100, 50), (200, 150, 100, 50)
        ],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000],
        'early_stopping': [True, False],
    }

    dnn_hyperparameters_range = {
        'hidden_layer_sizes': [
            (64,), (128,), (256,),
            (64, 64), (128, 64), (256, 128, 64)
        ],
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam', 'sgd'],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64],
        'epochs': [50],
        'dropout_rate': [0.0, 0.2],
        'early_stopping': [True],
    }


    svm_hyperparameters_range_ufc = {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'class_weight': ['balanced'],
        'decision_function_shape': ['ovo', 'ovr'],
    }

    knn_hyperparameters_range = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'metric': ['minkowski', 'chebyshev']
    }

    dt_hyperparameters_range = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6, 8, 10, 15, 20, 25, 30],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'class_weight': ['balanced'],
        'max_leaf_nodes': [10, 20, 30, 40, 50],
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
    }

    boosting_hyperparameters_range_ufc = {
        'n_estimators': [50, 100, 200, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__max_depth': [1, 2, 3, 4],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4]
    }

    print("Loading UFC data...")
    X_ufc, y_ufc, preprocessor, fight_data, relevant_columns = load_ufc_data()
    if X_ufc is None:
        print("Failed to load UFC data.")
        exit(1)

    generate_plots(MLPClassifier(), X_ufc, y_ufc, nn_hyperparameters_range_ufc, "Neural Network (UFC)", output_path, cv)
    generate_plots(SVC(), X_ufc, y_ufc, svm_hyperparameters_range_ufc, "SVM (UFC)", output_path, cv)
    generate_plots(KNeighborsClassifier(), X_ufc, y_ufc, knn_hyperparameters_range, "KNN (UFC)", output_path, cv)
    generate_plots(DecisionTreeClassifier(), X_ufc, y_ufc, dt_hyperparameters_range, "Decision Tree (UFC)", output_path, cv)
    generate_plots(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, max_leaf_nodes=10)), X_ufc, y_ufc, boosting_hyperparameters_range_ufc, "AdaBoost (UFC)", output_path, cv)

    print("Splitting UFC dataset into training and testing sets...")
    X_train_ufc, X_test_ufc, y_train_ufc, y_test_ufc = train_test_split(X_ufc, y_ufc, test_size=0.2, random_state=21, stratify=y_ufc)
    print("Training on UFC data...")
    print("Training Fully Connected Neural Network...")
    nn_model_ufc = train_fully_connected_neural_network(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, nn_hyperparameters_range_ufc, 'NeuralNetwork', 'UFC', output_path)
    print("Training Deep Neural Network...")
    dnn_model_ufc = train_deep_neural_network(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, dnn_hyperparameters_range, 'DeepNeuralNetwork', 'UFC', output_path)
    print("Training Support Vector Machine...")
    svm_model_ufc = train_support_vector_machine(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, svm_hyperparameters_range_ufc, 'SVM', 'UFC', output_path)
    print("Training K-Nearest Neighbor...")
    knn_model_ufc = train_k_nearest_neighbor(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, knn_hyperparameters_range, 'KNN', 'UFC', output_path)
    print("Training Decision Tree...")
    dt_model_ufc = train_decision_tree(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, dt_hyperparameters_range, 'DecisionTree', 'UFC', output_path)
    print("Training Adaptive Boosting with Decision Trees...")
    ada_model_ufc = train_boosting(X_train_ufc, y_train_ufc, X_test_ufc, y_test_ufc, boosting_hyperparameters_range_ufc, 'AdaBoost', 'UFC', output_path)

    print(f"All tasks completed. Results saved in {output_path}.")
