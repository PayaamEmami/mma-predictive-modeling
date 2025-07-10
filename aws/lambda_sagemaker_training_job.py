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

    # Determine the secret name based on the S3 key
    region_name = "us-west-1"
    s3_key = event["Records"][0]["s3"]["object"]["key"]
    if s3_key.startswith("experiments/"):
        secret_name = "mpm-secrets-experimental"
    else:
        secret_name = "mpm-secrets"
    secrets = get_secret(secret_name, region_name)

    job_name = f"mma-train-job-{'experimental' if secret_name == 'mpm-secrets-experimental' else 'main'}-{timestamp}"

    # Create the training job
    response = sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39",
            "TrainingInputMode": "File",
        },
        RoleArn=secrets["role_arn"],
        OutputDataConfig={"S3OutputPath": secrets["s3_output_path"]},
        ResourceConfig={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 21600},
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
