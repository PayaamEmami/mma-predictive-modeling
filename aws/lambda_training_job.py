"""
AWS Lambda Function: SageMaker Training Job Trigger
===================================================

This Lambda function triggers SageMaker training jobs for MMA prediction model
training when new data is uploaded to S3. It supports both main and experimental
training workflows with separate configurations.

Triggered by:
- S3 object creation events in data/ folder (main training)
- S3 object creation events in experiments/ folder (experimental training)

Key Features:
- Dual-mode operation: main vs experimental training
- Environment-based configuration management
- Automatic job naming with timestamps
- GPU-optimized instance configuration (ml.g4dn.xlarge)
- Comprehensive hyperparameter passing to training containers

Training Modes:
- Main: Uses standard environment variables for production training
- Experimental: Uses EXPERIMENTAL_ prefixed variables for testing

Dependencies:
- Environment variables: ROLE_ARN, SAGEMAKER_PROGRAM, SAGEMAKER_SUBMIT_DIRECTORY, etc.
- Experimental variables: EXPERIMENTAL_ROLE_ARN, EXPERIMENTAL_SAGEMAKER_PROGRAM, etc.
- SageMaker execution role with appropriate permissions
- PyTorch training container image
"""

import boto3
import datetime
import os
from datetime import timezone


def get_config_from_env(is_experimental=False):
    """Get configuration from environment variables"""
    prefix = "EXPERIMENTAL_" if is_experimental else ""

    config = {
        "role_arn": os.environ.get(f"{prefix}ROLE_ARN"),
        "sagemaker_program": os.environ.get(f"{prefix}SAGEMAKER_PROGRAM"),
        "sagemaker_submit_directory": os.environ.get(
            f"{prefix}SAGEMAKER_SUBMIT_DIRECTORY"
        ),
        "s3_bucket": os.environ.get(f"{prefix}S3_BUCKET"),
        "s3_data_key": os.environ.get(f"{prefix}S3_DATA_KEY"),
        "s3_results_prefix": os.environ.get(f"{prefix}S3_RESULTS_PREFIX"),
        "s3_output_path": os.environ.get(f"{prefix}S3_OUTPUT_PATH"),
    }

    # Validate required environment variables
    required_vars = [
        "role_arn",
        "sagemaker_submit_directory",
        "s3_bucket",
        "s3_results_prefix",
        "s3_output_path",
    ]
    missing_vars = [var for var in required_vars if not config[var]]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    return config


def lambda_handler(event, context):
    """
    Main handler for SageMaker training job creation.

    Process:
    1. Determine training mode (main vs experimental) from S3 key
    2. Load appropriate environment configuration
    3. Generate unique training job name with timestamp
    4. Configure SageMaker job parameters and hyperparameters
    5. Submit training job to SageMaker

    Returns:
    - Success: Training job ARN and job details
    - Error: Configuration or SageMaker submission errors
    """
    sagemaker = boto3.client("sagemaker")
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    # Determine if this is experimental based on the S3 key
    s3_key = event["Records"][0]["s3"]["object"]["key"]
    is_experimental = s3_key.startswith("experiments/")

    try:
        # Get configuration from environment variables
        config = get_config_from_env(is_experimental)

        job_name = (
            f"mpm-train-job-{'experimental' if is_experimental else 'main'}-{timestamp}"
        )

        # Prepare hyperparameters
        hyperparameters = {
            "sagemaker_program": config["sagemaker_program"],
            "sagemaker_submit_directory": config["sagemaker_submit_directory"],
            "s3_bucket": config["s3_bucket"],
            "s3_data_key": config["s3_data_key"],
            "s3_results_prefix": config["s3_results_prefix"],
        }

        # Create the training job
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39",
                "TrainingInputMode": "File",
            },
            RoleArn=config["role_arn"],
            OutputDataConfig={"S3OutputPath": config["s3_output_path"]},
            ResourceConfig={
                "InstanceType": "ml.g4dn.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 21600},
            HyperParameters=hyperparameters,
        )

        return {
            "statusCode": 200,
            "body": "Training job started: " + response["TrainingJobArn"],
        }

    except ValueError as e:
        print(f"Configuration error: {e}")
        return {"statusCode": 500, "body": f"Configuration error: {str(e)}"}
    except Exception as e:
        print(f"Error starting training job: {e}")
        return {"statusCode": 500, "body": f"Error starting training job: {str(e)}"}
