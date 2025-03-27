import torch
import pandas as pd
import numpy as np
from data import load_ufc_data
from models import initialize_models
from config import DEVICE, DATA_PATH
from preprocessing import parse_height, parse_reach, parse_strike, parse_control_time
import os

def get_fighter_stats(fighter_id):
    """
    Get fighter statistics from the UFC events CSV file.
    
    Args:
        fighter_id: The UFC stats profile URL of the fighter
        
    Returns:
        Dictionary containing the fighter's statistics
    """
    # Load the UFC events data
    fight_data_path = os.path.join(DATA_PATH, "ufc_events.csv")
    fight_data = pd.read_csv(fight_data_path)
    
    # Find all fights where this fighter participated
    fighter1_fights = fight_data[fight_data['Fighter1_ID'] == fighter_id]
    fighter2_fights = fight_data[fight_data['Fighter2_ID'] == fighter_id]
    
    if len(fighter1_fights) == 0 and len(fighter2_fights) == 0:
        raise ValueError(f"Fighter with ID {fighter_id} not found in the dataset")
    
    # Combine all fights
    all_fights = pd.concat([fighter1_fights, fighter2_fights])
    
    # Calculate statistics
    stats = {}
    
    # Get basic info from the most recent fight
    most_recent_fight = all_fights.iloc[-1]
    if most_recent_fight['Fighter1_ID'] == fighter_id:
        stats['Height'] = most_recent_fight['Fighter1_Height']
        stats['Reach'] = most_recent_fight['Fighter1_Reach']
        stats['Stance'] = most_recent_fight['Fighter1_Stance']
    else:
        stats['Height'] = most_recent_fight['Fighter2_Height']
        stats['Reach'] = most_recent_fight['Fighter2_Reach']
        stats['Stance'] = most_recent_fight['Fighter2_Stance']
    
    # Calculate age (using most recent fight date)
    stats['Age'] = (pd.to_datetime(most_recent_fight['EventDate']) - 
                   pd.to_datetime(most_recent_fight['Fighter1_DOB'] if most_recent_fight['Fighter1_ID'] == fighter_id 
                                else most_recent_fight['Fighter2_DOB'])).days / 365.25
    
    # Calculate fight statistics
    total_fights = len(all_fights)
    wins = 0
    losses = 0
    finishes = 0
    total_control_time = 0
    total_submission_attempts = 0
    total_leg_strikes = 0
    total_clinch_strikes = 0
    total_strikes_landed = 0
    total_strikes_attempted = 0
    total_takedowns_landed = 0
    total_takedowns_attempted = 0
    total_reversals = 0
    
    for _, fight in all_fights.iterrows():
        is_fighter1 = fight['Fighter1_ID'] == fighter_id
        
        # Count wins and losses
        if fight['Winner'] == '1' and is_fighter1:
            wins += 1
            if 'Decision' not in fight['Method']:
                finishes += 1
        elif fight['Winner'] == '2' and not is_fighter1:
            wins += 1
            if 'Decision' not in fight['Method']:
                finishes += 1
        else:
            losses += 1
        
        # Sum up fight statistics
        if is_fighter1:
            total_control_time += parse_control_time(fight['Fighter1_Control_Time'])
            total_submission_attempts += float(fight['Fighter1_Submission_Attempts'])
            leg_strikes = parse_strike(fight['Fighter1_Leg_Strikes'])[0]
            clinch_strikes = parse_strike(fight['Fighter1_Clinch_Strikes'])[0]
            strikes = parse_strike(fight['Fighter1_Significant_Strikes'])
            takedowns = parse_strike(fight['Fighter1_Takedowns'])
            total_reversals += float(fight['Fighter1_Reversals'])
        else:
            total_control_time += parse_control_time(fight['Fighter2_Control_Time'])
            total_submission_attempts += float(fight['Fighter2_Submission_Attempts'])
            leg_strikes = parse_strike(fight['Fighter2_Leg_Strikes'])[0]
            clinch_strikes = parse_strike(fight['Fighter2_Clinch_Strikes'])[0]
            strikes = parse_strike(fight['Fighter2_Significant_Strikes'])
            takedowns = parse_strike(fight['Fighter2_Takedowns'])
            total_reversals += float(fight['Fighter2_Reversals'])
        
        total_leg_strikes += leg_strikes
        total_clinch_strikes += clinch_strikes
        total_strikes_landed += strikes[0]
        total_strikes_attempted += strikes[1]
        total_takedowns_landed += takedowns[0]
        total_takedowns_attempted += takedowns[1]
    
    # Calculate averages
    stats['AvgFightTime'] = 600  # Default to 10 minutes if not available
    stats['TimeSinceLastFight'] = 90  # Default to 90 days if not available
    stats['FinishRate'] = finishes / wins if wins > 0 else 0
    stats['Wins'] = wins
    stats['Losses'] = losses
    stats['AvgControlTime'] = total_control_time / total_fights
    stats['AvgSubmissionAttempts'] = total_submission_attempts / total_fights
    stats['AvgLegStrikes'] = total_leg_strikes / total_fights
    stats['AvgClinchStrikes'] = total_clinch_strikes / total_fights
    stats['AvgStrikesLanded'] = total_strikes_landed / total_fights
    stats['AvgStrikesAttempted'] = total_strikes_attempted / total_fights
    stats['StrikeAccuracy'] = total_strikes_landed / total_strikes_attempted if total_strikes_attempted > 0 else 0
    stats['AvgTakedownsLanded'] = total_takedowns_landed / total_fights
    stats['AvgTakedownsAttempted'] = total_takedowns_attempted / total_fights
    stats['AvgReversals'] = total_reversals / total_fights
    
    return stats

def prepare_fighter_data(fighter1_data, fighter2_data):
    """
    Prepare fighter data for prediction by processing it in the same way as training data.
    
    Args:
        fighter1_data: Dictionary containing Fighter1's statistics
        fighter2_data: Dictionary containing Fighter2's statistics
        
    Returns:
        Processed feature vector ready for model prediction
    """
    # Create feature dictionary
    features = {}
    
    # Process Fighter1 data
    features['Fighter1_Height_cm'] = parse_height(fighter1_data.get('Height', '0ft 0in'))
    features['Fighter1_Reach_cm'] = parse_reach(fighter1_data.get('Reach', '0in'))
    features['Fighter1_Age'] = float(fighter1_data.get('Age', 30))
    features['Fighter1_AvgFightTime'] = float(fighter1_data.get('AvgFightTime', 0))
    features['Fighter1_TimeSinceLastFight'] = float(fighter1_data.get('TimeSinceLastFight', 0))
    features['Fighter1_FinishRate'] = float(fighter1_data.get('FinishRate', 0))
    features['Fighter1_Wins'] = int(fighter1_data.get('Wins', 0))
    features['Fighter1_Losses'] = int(fighter1_data.get('Losses', 0))
    features['Fighter1_AvgControlTime'] = float(fighter1_data.get('AvgControlTime', 0))
    features['Fighter1_AvgSubmissionAttempts'] = float(fighter1_data.get('AvgSubmissionAttempts', 0))
    features['Fighter1_AvgLegStrikes'] = float(fighter1_data.get('AvgLegStrikes', 0))
    features['Fighter1_AvgClinchStrikes'] = float(fighter1_data.get('AvgClinchStrikes', 0))
    features['Fighter1_AvgStrikesLanded'] = float(fighter1_data.get('AvgStrikesLanded', 0))
    features['Fighter1_AvgStrikesAttempted'] = float(fighter1_data.get('AvgStrikesAttempted', 0))
    features['Fighter1_StrikeAccuracy'] = float(fighter1_data.get('StrikeAccuracy', 0))
    features['Fighter1_AvgTakedownsLanded'] = float(fighter1_data.get('AvgTakedownsLanded', 0))
    features['Fighter1_AvgTakedownsAttempted'] = float(fighter1_data.get('AvgTakedownsAttempted', 0))
    features['Fighter1_AvgReversals'] = float(fighter1_data.get('AvgReversals', 0))
    
    # Process Fighter2 data
    features['Fighter2_Height_cm'] = parse_height(fighter2_data.get('Height', '0ft 0in'))
    features['Fighter2_Reach_cm'] = parse_reach(fighter2_data.get('Reach', '0in'))
    features['Fighter2_Age'] = float(fighter2_data.get('Age', 30))
    features['Fighter2_AvgFightTime'] = float(fighter2_data.get('AvgFightTime', 0))
    features['Fighter2_TimeSinceLastFight'] = float(fighter2_data.get('TimeSinceLastFight', 0))
    features['Fighter2_FinishRate'] = float(fighter2_data.get('FinishRate', 0))
    features['Fighter2_Wins'] = int(fighter2_data.get('Wins', 0))
    features['Fighter2_Losses'] = int(fighter2_data.get('Losses', 0))
    features['Fighter2_AvgControlTime'] = float(fighter2_data.get('AvgControlTime', 0))
    features['Fighter2_AvgSubmissionAttempts'] = float(fighter2_data.get('AvgSubmissionAttempts', 0))
    features['Fighter2_AvgLegStrikes'] = float(fighter2_data.get('AvgLegStrikes', 0))
    features['Fighter2_AvgClinchStrikes'] = float(fighter2_data.get('AvgClinchStrikes', 0))
    features['Fighter2_AvgStrikesLanded'] = float(fighter2_data.get('AvgStrikesLanded', 0))
    features['Fighter2_AvgStrikesAttempted'] = float(fighter2_data.get('AvgStrikesAttempted', 0))
    features['Fighter2_StrikeAccuracy'] = float(fighter2_data.get('StrikeAccuracy', 0))
    features['Fighter2_AvgTakedownsLanded'] = float(fighter2_data.get('AvgTakedownsLanded', 0))
    features['Fighter2_AvgTakedownsAttempted'] = float(fighter2_data.get('AvgTakedownsAttempted', 0))
    features['Fighter2_AvgReversals'] = float(fighter2_data.get('AvgReversals', 0))
    
    # Add stance features (one-hot encoded)
    stances = ['Orthodox', 'Southpaw', 'Switch']
    for stance in stances:
        features[f'Fighter1_Stance_{stance}'] = 1 if fighter1_data.get('Stance', 'Orthodox') == stance else 0
        features[f'Fighter2_Stance_{stance}'] = 1 if fighter2_data.get('Stance', 'Orthodox') == stance else 0
    
    # Convert to numpy array and reshape for model input
    feature_vector = np.array(list(features.values())).reshape(1, -1)
    
    return feature_vector

def predict_fight(fighter1_id=None, fighter2_id=None, fighter1_data=None, fighter2_data=None, model_name='Random Forest'):
    """
    Predict the outcome of a fight between two fighters.
    
    Args:
        fighter1_id: The UFC stats profile URL of Fighter1 (optional)
        fighter2_id: The UFC stats profile URL of Fighter2 (optional)
        fighter1_data: Dictionary containing Fighter1's statistics (optional)
        fighter2_data: Dictionary containing Fighter2's statistics (optional)
        model_name: Name of the model to use for prediction (default: 'Random Forest')
        
    Returns:
        Dictionary containing prediction results and probabilities
    """
    # Get fighter data if IDs are provided
    if fighter1_id:
        fighter1_data = get_fighter_stats(fighter1_id)
    if fighter2_id:
        fighter2_data = get_fighter_stats(fighter2_id)
    
    if not fighter1_data or not fighter2_data:
        raise ValueError("Either fighter IDs or fighter data must be provided")
    
    # Load the trained model
    models = initialize_models(len(prepare_fighter_data({}, {}).flatten()), DEVICE)
    model = models.get(model_name)
    
    if model is None:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    
    # Prepare the feature vector
    X = prepare_fighter_data(fighter1_data, fighter2_data)
    
    # Make prediction
    if model_name in ['FCNN', 'RNN', 'LSTM', 'Transformer']:
        # Load model state if it exists
        model_path = os.path.join(DATA_PATH, f"{model_name}.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.astype(np.float32)).to(DEVICE)
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            probabilities = probabilities.cpu().numpy()[0]
    else:
        # For scikit-learn models
        probabilities = model.predict_proba(X)[0]
        predicted_class = model.predict(X)[0]
    
    # Prepare result dictionary
    result = {
        'winner': 'Fighter 1' if predicted_class == 0 else 'Fighter 2',
        'probabilities': {
            'Fighter 1': float(probabilities[0]),
            'Fighter 2': float(probabilities[1])
        },
        'model_used': model_name
    }
    
    return result

def print_prediction_result(result):
    """
    Print the prediction result in a formatted way.
    
    Args:
        result: Dictionary containing prediction results
    """
    print("\nFight Prediction Results:")
    print("-" * 50)
    print(f"Model Used: {result['model_used']}")
    print(f"Predicted Winner: {result['winner']}")
    print("\nWin Probabilities:")
    print(f"Fighter 1: {result['probabilities']['Fighter 1']:.2%}")
    print(f"Fighter 2: {result['probabilities']['Fighter 2']:.2%}")
    print("-" * 50) 