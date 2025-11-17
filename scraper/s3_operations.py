"""
AWS S3 operations.

Functions for uploading and downloading files from S3 storage.
Handles the data persistence layer for scraped fight data.
"""

import boto3
from pathlib import Path
from botocore.exceptions import ClientError


def download_from_s3(
    bucket: str,
    s3_key: str,
    local_path: str,
    region: str = 'us-west-1'
) -> bool:
    """
    Download file from S3 to local filesystem.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)
        local_path: Local file path to save to
        region: AWS region

    Returns:
        True if successful, False if file doesn't exist in S3

    Raises:
        Exception: For other S3 errors (permissions, network, etc.)
    """
    s3_client = boto3.client('s3', region_name=region)

    try:
        # Check if file exists in S3
        s3_client.head_object(Bucket=bucket, Key=s3_key)

        # Delete local file if it exists
        local_file = Path(local_path)
        if local_file.exists():
            local_file.unlink()
            print(f"Deleted existing local file: {local_path}")

        # Ensure parent directory exists
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # Download from S3
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"Downloaded from S3: s3://{bucket}/{s3_key} -> {local_path}")
        return True

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"File not found in S3: s3://{bucket}/{s3_key}")
            return False
        else:
            print(f"S3 error downloading file: {e}")
            raise


def upload_to_s3(
    local_path: str,
    bucket: str,
    s3_key: str,
    region: str = 'us-west-1'
) -> bool:
    """
    Upload file from local filesystem to S3.

    Args:
        local_path: Path to local file
        bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)
        region: AWS region

    Returns:
        True if successful, False if local file doesn't exist

    Raises:
        Exception: For S3 errors (permissions, network, etc.)
    """
    local_file = Path(local_path)

    if not local_file.exists():
        print(f"Local file not found, skipping upload: {local_path}")
        return False

    try:
        s3_client = boto3.client('s3', region_name=region)
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded to S3: {local_path} -> s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise


def check_s3_object_exists(bucket: str, s3_key: str, region: str = 'us-west-1') -> bool:
    """
    Check if an object exists in S3.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key
        region: AWS region

    Returns:
        True if object exists, False otherwise
    """
    s3_client = boto3.client('s3', region_name=region)

    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False
        else:
            raise


def list_s3_objects(bucket: str, prefix: str = '', region: str = 'us-west-1') -> list[str]:
    """
    List objects in S3 bucket with optional prefix.

    Args:
        bucket: S3 bucket name
        prefix: Optional prefix to filter objects
        region: AWS region

    Returns:
        List of object keys
    """
    s3_client = boto3.client('s3', region_name=region)

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            return []

        return [obj['Key'] for obj in response['Contents']]

    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        raise

