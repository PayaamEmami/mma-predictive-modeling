# preprocessing.py
import pandas as pd
from collections import defaultdict
from utils import parse_height, parse_reach, parse_strike, parse_time
import re

def is_finish(method_str):
    if pd.isnull(method_str):
        return False
    return 'Decision' not in method_str

def calculate_fight_time(round_num, time_str):
    if pd.isnull(round_num) or pd.isnull(time_str):
        return 0
    round_num = int(round_num)
    time_in_current_round = parse_time(time_str)
    total_fight_time = (round_num - 1) * 300 + time_in_current_round
    return total_fight_time

def process_landed_attempted(fight_data, column_name):
    landed_attempted = fight_data[column_name].apply(parse_strike)
    landed_col = column_name + '_Landed'
    attempted_col = column_name + '_Attempted'
    fight_data[landed_col] = [x[0] for x in landed_attempted]
    fight_data[attempted_col] = [x[1] for x in landed_attempted]

def preprocess_fight_data(fight_data):
    # drop unnecessary columns and filter data
    fight_data = fight_data.drop(columns=['EventName'])
    fight_data = fight_data[~fight_data['Winner'].isin(['NC', 'D'])]

    # process height and reach
    fight_data['Fighter1_Height_cm'] = fight_data['Fighter1_Height'].apply(parse_height)
    fight_data['Fighter2_Height_cm'] = fight_data['Fighter2_Height'].apply(parse_height)
    fight_data['Fighter1_Reach_cm'] = fight_data['Fighter1_Reach'].apply(parse_reach)
    fight_data['Fighter2_Reach_cm'] = fight_data['Fighter2_Reach'].apply(parse_reach)

    # process control time
    fight_data['Fighter1_Control_Time_sec'] = fight_data['Fighter1_Control_Time'].apply(parse_time)
    fight_data['Fighter2_Control_Time_sec'] = fight_data['Fighter2_Control_Time'].apply(parse_time)

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
        process_landed_attempted(fight_data, col)

    # calculate total fight time
    fight_data['Fight_Time_sec'] = fight_data.apply(
        lambda row: calculate_fight_time(row['Round'], row['Time']), axis=1
    )

    return fight_data

def initialize_fighter_stats():
    return defaultdict(lambda: {
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

def update_fighter_statistics(fight_data, fighter_stats):
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

    # sort by eventdate
    fight_data = fight_data.sort_values('EventDate').reset_index(drop=True)

    # iterate through each fight to update statistics
    for idx, row in fight_data.iterrows():
        fight_time = row['Fight_Time_sec']
        event_date = row['EventDate']

        # update stats before the current fight
        for fighter_num, opponent_num in [('Fighter1', 'Fighter2'), ('Fighter2', 'Fighter1')]:
            fighter_id = row[f'{fighter_num}_ID']
            stats = fighter_stats[fighter_id]
            stats_before = stats.copy()

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
                # no previous fights
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

            # update win/loss/draw counts before the fight
            fight_data.at[idx, f'{fighter_num}_Wins'] = stats_before['Wins']
            fight_data.at[idx, f'{fighter_num}_Losses'] = stats_before['Losses']
            fight_data.at[idx, f'{fighter_num}_Draws'] = stats_before['Draws']
            fight_data.at[idx, f'{fighter_num}_NoContests'] = stats_before['NoContests']

        # update fighter stats with current fight data
        for fighter_num in ['Fighter1', 'Fighter2']:
            fighter_id = row[f'{fighter_num}_ID']
            stats = fighter_stats[fighter_id]

            stats['TotalFightTime'] += fight_time if not pd.isnull(fight_time) else 0
            stats['NumFights'] += 1
            stats['LastFightDate'] = event_date

            control_time = row[f'{fighter_num}_Control_Time_sec']
            stats['TotalControlTime'] += control_time if not pd.isnull(control_time) else 0

            sub_attempts = float(row[f'{fighter_num}_Submission_Attempts']) if not pd.isnull(row[f'{fighter_num}_Submission_Attempts']) else 0
            stats['TotalSubmissionAttempts'] += sub_attempts

            leg_strikes = row[f'{fighter_num}_Leg_Strikes_Landed'] if not pd.isnull(row[f'{fighter_num}_Leg_Strikes_Landed']) else 0
            stats['TotalLegStrikes'] += leg_strikes

            clinch_strikes = row[f'{fighter_num}_Clinch_Strikes_Landed'] if not pd.isnull(row[f'{fighter_num}_Clinch_Strikes_Landed']) else 0
            stats['TotalClinchStrikes'] += clinch_strikes

            strikes_landed = row[f'{fighter_num}_Significant_Strikes_Landed'] if not pd.isnull(row[f'{fighter_num}_Significant_Strikes_Landed']) else 0
            strikes_attempted = row[f'{fighter_num}_Significant_Strikes_Attempted'] if not pd.isnull(row[f'{fighter_num}_Significant_Strikes_Attempted']) else 0
            stats['TotalStrikesLanded'] += strikes_landed
            stats['TotalStrikesAttempted'] += strikes_attempted

            # handle takedowns
            takedowns_landed = row.get(f'{fighter_num}_Takedowns_Landed', row[f'{fighter_num}_Takedowns']) if not pd.isnull(row.get(f'{fighter_num}_Takedowns_Landed', row[f'{fighter_num}_Takedowns'])) else 0
            takedowns_attempted = row.get(f'{fighter_num}_Takedowns_Attempted', row[f'{fighter_num}_Takedowns']) if not pd.isnull(row.get(f'{fighter_num}_Takedowns_Attempted', row[f'{fighter_num}_Takedowns'])) else 0
            stats['TotalTakedownsLanded'] += takedowns_landed
            stats['TotalTakedownsAttempted'] += takedowns_attempted

            reversals = float(row[f'{fighter_num}_Reversals']) if not pd.isnull(row[f'{fighter_num}_Reversals']) else 0
            stats['TotalReversals'] += reversals

        # update win/loss/draw statistics based on the fight outcome
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

    return fight_data
