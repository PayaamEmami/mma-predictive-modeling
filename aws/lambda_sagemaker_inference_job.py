import sys
import argparse

# Add the code directory to the Python path
sys.path.append("/opt/ml/code")

from inference import main as inference_main


def main():
    """
    Entry point for SageMaker inference job.
    This function is specifically for inference only
    """
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument("--mode", type=str, default="inference")
    parser.add_argument("--s3_bucket", type=str, required=True)
    parser.add_argument(
        "--upcoming_fights_key", type=str, default="upcoming_fights.json"
    )
    parser.add_argument("--historical_data_key", type=str, default="fight_events.csv")
    parser.add_argument(
        "--predictions_key", type=str, default="predictions/latest_predictions.json"
    )

    args = parser.parse_args()

    if args.mode == "inference":
        print("Running in inference mode...")
        inference_main(
            args.s3_bucket,
            args.upcoming_fights_key,
            args.historical_data_key,
            args.predictions_key,
        )
    else:
        print(
            f"ERROR: Invalid mode '{args.mode}'. This Lambda function only supports inference mode."
        )
        print("Available modes: 'inference'")
        sys.exit(1)


if __name__ == "__main__":
    main()
