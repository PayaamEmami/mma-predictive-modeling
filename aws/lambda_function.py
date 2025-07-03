import boto3
import datetime
import json
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
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    secret_name = "mma-predictive-modeling-secrets"
    region_name = "us-west-1"
    secrets = get_secret(secret_name, region_name)

    response = sagemaker.create_training_job(
        TrainingJobName=f"mma-train-job-{timestamp}",
        AlgorithmSpecification={
            "TrainingImage": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu116",
            "TrainingInputMode": "File",
        },
        RoleArn=secrets["mpm-RoleArn"],
        OutputDataConfig={"S3OutputPath": secrets["mpm-S3OutputPath"]},
        ResourceConfig={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        HyperParameters={
            "sagemaker_program": secrets["mpm-sagemaker_program"],
            "sagemaker_submit_directory": secrets["mpm-sagemaker_submit_directory"],
            "s3_bucket": secrets["mpm-s3_bucket"],
            "s3_data_key": secrets["mpm-s3_data_key"],
            "s3_results_prefix": secrets["mpm-s3_results_prefix"],
        },
    )

    return {
        "statusCode": 200,
        "body": "Training job started: " + response["TrainingJobArn"],
    }
