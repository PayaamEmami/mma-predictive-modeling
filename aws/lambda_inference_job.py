"""
AWS Lambda Function: SageMaker Inference Job Trigger
====================================================

This Lambda function triggers SageMaker training jobs for MMA prediction inference
when new upcoming fights data is uploaded to S3. It monitors S3 events and starts
the machine learning inference pipeline to generate predictions.

Triggered by:
- S3 object creation events for data/upcoming_fights.json

Key Features:
- Validates S3 event is for upcoming fights data
- Prevents duplicate predictions by checking if event already has archived predictions
- Creates unique training job names with timestamps
- Configures SageMaker with appropriate instance types and resources
- Passes S3 bucket information as hyperparameters
- Monitors job execution with timeout controls

Duplicate Prevention:
- Reads upcoming fights data to extract event name
- Checks if archived prediction already exists for this event
- Skips inference job if predictions already exist to prevent waste and duplication

Dependencies:
- Environment variables: SAGEMAKER_ROLE_ARN, TRAINING_IMAGE_URI, S3_BUCKET
- SageMaker execution role with appropriate permissions
- Docker image URI for the training container
"""

import json
import boto3
import os
import re
from datetime import datetime


def convert_event_name_to_filename(event_name):
    """
    Convert event name to a filename-safe format (same logic as prediction archiver).
    """
    # Convert to lowercase
    filename = event_name.lower()

    # Replace common separators with hyphens
    filename = re.sub(r"[:\-–—]+", "-", filename)

    # Replace spaces and other non-alphanumeric characters with hyphens
    filename = re.sub(r"[^a-z0-9]+", "-", filename)

    # Remove leading/trailing hyphens and multiple consecutive hyphens
    filename = re.sub(r"^-+|-+$", "", filename)
    filename = re.sub(r"-+", "-", filename)

    return filename


def check_predictions_already_exist(s3_bucket, event_name):
    """
    Check if predictions for this event already exist in the archived predictions.
    Returns True if predictions already exist, False otherwise.
    """
    try:
        s3_client = boto3.client("s3")

        # Convert event name to filename format (same as archiver)
        filename = convert_event_name_to_filename(event_name)
        archive_key = f"predictions/archive/{filename}.json"

        # Check if the archived prediction file exists
        s3_client.head_object(Bucket=s3_bucket, Key=archive_key)

        print(f"Predictions already exist for event '{event_name}' at {archive_key}")
        return True

    except Exception as e:
        # Handle both NoSuchKey and other S3 exceptions
        error_str = str(e)
        if "404" in error_str or "NoSuchKey" in error_str or "Not Found" in error_str:
            print(
                f"No existing predictions found for event '{event_name}' - proceeding with inference"
            )
            return False
        else:
            print(f"Error checking for existing predictions: {e}")
            # If we can't check for other reasons, err on the side of caution and don't run inference
            return True


def lambda_handler(event, context):
    """
    Lambda function to trigger SageMaker training job for MMA predictions inference.
    Triggered when upcoming fights data is uploaded to S3.
    """

    # Configuration
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    training_image = os.environ["TRAINING_IMAGE_URI"]
    s3_bucket = os.environ["S3_BUCKET"]

    # Get the S3 object information from the event
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]

        # Only trigger if it's the upcoming fights file
        if key != "data/upcoming_fights.json":
            print(f"Ignoring S3 event for {key}")
            return {
                "statusCode": 200,
                "body": json.dumps("Ignored - not upcoming fights file"),
            }

        print(f"Processing upcoming fights file: {key}")

    except Exception as e:
        print(f"Error parsing S3 event: {e}")
        return {"statusCode": 400, "body": json.dumps(f"Error parsing S3 event: {e}")}

    # Check if predictions already exist for this event to prevent duplicates
    try:
        s3_client = boto3.client("s3")

        # Read the upcoming fights data to get the event name
        response = s3_client.get_object(Bucket=bucket, Key=key)
        upcoming_fights_content = response["Body"].read().decode("utf-8")
        upcoming_fights_data = json.loads(upcoming_fights_content)

        event_name = upcoming_fights_data.get("EventName", "")
        if not event_name:
            print("Warning: No EventName found in upcoming fights data")
        else:
            print(f"Event name: {event_name}")

            # Check if predictions already exist
            if check_predictions_already_exist(s3_bucket, event_name):
                return {
                    "statusCode": 200,
                    "body": json.dumps(
                        {
                            "message": "Predictions already exist for this event - skipping inference job",
                            "event_name": event_name,
                            "skipped": True,
                        }
                    ),
                }

    except Exception as e:
        print(f"Error checking for existing predictions: {e}")
        # If we can't read the file or check for existing predictions, proceed with caution
        # This ensures the system still works even if there's an issue with the duplicate check
        print("Proceeding with inference job due to error in duplicate check")

    # Initialize SageMaker client after duplicate check
    sagemaker_client = boto3.client("sagemaker")

    # Create unique training job name
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    training_job_name = f"mpm-inference-{timestamp}"

    # Define training job parameters
    training_params = {
        "TrainingJobName": training_job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{s3_bucket}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None",
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{s3_bucket}/inference-output/"},
        "ResourceConfig": {
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},  # 1 hour
        "HyperParameters": {
            "sagemaker_program": "code/inference.py",
            "sagemaker_submit_directory": f"s3://{s3_bucket}/code.tar.gz",
            "s3_bucket": s3_bucket,
        },
    }

    try:
        # Start the training job
        response = sagemaker_client.create_training_job(**training_params)

        print(f"Started SageMaker inference job: {training_job_name}")
        print(f"Job ARN: {response['TrainingJobArn']}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "SageMaker inference job started successfully",
                    "training_job_name": training_job_name,
                    "training_job_arn": response["TrainingJobArn"],
                }
            ),
        }

    except Exception as e:
        print(f"Error starting SageMaker job: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error starting SageMaker job: {e}"),
        }
