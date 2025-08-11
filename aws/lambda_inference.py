import json
import boto3
import os
from datetime import datetime


def lambda_handler(event, context):
    """
    Lambda function to trigger SageMaker training job for MMA predictions inference.
    Triggered when upcoming fights data is uploaded to S3.
    """

    sagemaker_client = boto3.client("sagemaker")

    # Configuration
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    training_image = os.environ["TRAINING_IMAGE_URI"]
    s3_bucket = os.environ["S3_BUCKET"]

    # Get the S3 object information from the event
    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]

        # Only trigger if it's the upcoming fights file
        if key != "upcoming_fights.json":
            print(f"Ignoring S3 event for {key}")
            return {
                "statusCode": 200,
                "body": json.dumps("Ignored - not upcoming fights file"),
            }

        print(f"Processing upcoming fights file: {key}")

    except Exception as e:
        print(f"Error parsing S3 event: {e}")
        return {"statusCode": 400, "body": json.dumps(f"Error parsing S3 event: {e}")}

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
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},  # 1 hour
        "HyperParameters": {
            "mode": "inference",
            "s3_bucket": s3_bucket,
            "upcoming_fights_key": "upcoming_fights.json",
            "historical_data_key": "fight_events.csv",
            "predictions_key": "predictions/latest_predictions.json",
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
