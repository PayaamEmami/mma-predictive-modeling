import boto3
import datetime
import json
from datetime import timezone
from botocore.exceptions import ClientError


def get_secret(secret_name, region_name="us-west-1"):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    return json.loads(get_secret_value_response["SecretString"])


def lambda_handler(event, context):
    sagemaker = boto3.client("sagemaker")
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    secret_name = "mpm-secrets"
    region_name = "us-west-1"
    secrets = get_secret(secret_name, region_name)

    response = sagemaker.create_training_job(
        TrainingJobName=f"mma-train-job-{timestamp}",
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu116",
            "TrainingInputMode": "File",
        },
        RoleArn=secrets["role_arn"],
        OutputDataConfig={"S3OutputPath": secrets["s3_output_path"]},
        ResourceConfig={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        HyperParameters={
            "sagemaker_program": secrets["sagemaker_program"],
            "sagemaker_submit_directory": secrets["sagemaker_submit_directory"],
            "s3_bucket": secrets["s3_bucket"],
            "s3_data_key": secrets["s3_data_key"],
            "s3_results_prefix": secrets["s3_results_prefix"],
        },
    )

    return {
        "statusCode": 200,
        "body": "Training job started: " + response["TrainingJobArn"],
    }
