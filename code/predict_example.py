from predict import predict_fight, print_prediction_result

# Example 1: Using fighter IDs (UFC stats profile URLs)
fighter1_id = "http://ufcstats.com/fighter-details/19f8a2f6eecd92ac"  # Example ID
fighter2_id = "http://ufcstats.com/fighter-details/9c442aaf149ea982"  # Example ID

print("\nPredicting fight using fighter IDs:")
try:
    result = predict_fight(fighter1_id=fighter1_id, fighter2_id=fighter2_id, model_name='Random Forest')
    print_prediction_result(result)
except Exception as e:
    print(f"Error: {str(e)}")

# Example 2: Using manual fighter data
fighter1 = {
    'Height': '5ft 11in',
    'Reach': '72in',
    'Age': 30,
    'Stance': 'Orthodox',
    'AvgFightTime': 600,  # 10 minutes
    'TimeSinceLastFight': 90,  # days
    'FinishRate': 0.7,
    'Wins': 15,
    'Losses': 3,
    'AvgControlTime': 120,  # seconds
    'AvgSubmissionAttempts': 2,
    'AvgLegStrikes': 15,
    'AvgClinchStrikes': 10,
    'AvgStrikesLanded': 80,
    'AvgStrikesAttempted': 150,
    'StrikeAccuracy': 0.53,
    'AvgTakedownsLanded': 2,
    'AvgTakedownsAttempted': 4,
    'AvgReversals': 1
}

fighter2 = {
    'Height': '6ft 0in',
    'Reach': '74in',
    'Age': 28,
    'Stance': 'Southpaw',
    'AvgFightTime': 540,  # 9 minutes
    'TimeSinceLastFight': 60,  # days
    'FinishRate': 0.6,
    'Wins': 12,
    'Losses': 2,
    'AvgControlTime': 90,  # seconds
    'AvgSubmissionAttempts': 1,
    'AvgLegStrikes': 12,
    'AvgClinchStrikes': 8,
    'AvgStrikesLanded': 70,
    'AvgStrikesAttempted': 130,
    'StrikeAccuracy': 0.54,
    'AvgTakedownsLanded': 3,
    'AvgTakedownsAttempted': 5,
    'AvgReversals': 2
}

print("\nPredicting fight using manual fighter data:")
try:
    result = predict_fight(fighter1_data=fighter1, fighter2_data=fighter2, model_name='Random Forest')
    print_prediction_result(result)
except Exception as e:
    print(f"Error: {str(e)}")

# Example 3: Using different models
models = ['Random Forest', 'Gradient Boosting', 'FCNN']

print("\nComparing predictions from different models:")
for model_name in models:
    try:
        result = predict_fight(fighter1_id=fighter1_id, fighter2_id=fighter2_id, model_name=model_name)
        print_prediction_result(result)
    except Exception as e:
        print(f"\nError using {model_name}: {str(e)}") 