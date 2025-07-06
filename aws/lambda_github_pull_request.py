import boto3
import tempfile
import os
import shutil
from datetime import datetime, timezone
from github import Github
from git import Repo
import json
from botocore.exceptions import ClientError


def get_secret(secret_name, region_name="us-west-1"):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except ClientError as e:
        raise e


def lambda_handler(event, context):
    secret_name = "mpm-secrets"
    region_name = "us-west-1"
    secrets = get_secret(secret_name, region_name)

    # GitHub repo config
    GITHUB_REPO = "PayaamEmami/mma-predictive-modeling"
    GITHUB_BRANCH = "main"
    GITHUB_TOKEN = secrets["github_token"]

    # S3 config
    S3_BUCKET = secrets["s3_bucket"]
    S3_PREFIX = secrets["s3_results_prefix"]
    S3 = boto3.client("s3")

    # Temp directories
    tmp_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(tmp_dir, "repo")
    s3_results_dir = os.path.join(tmp_dir, "s3_results")
    os.makedirs(s3_results_dir, exist_ok=True)

    # Clone repo
    repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
    repo = Repo.clone_from(repo_url, repo_dir, branch=GITHUB_BRANCH)

    # Create new branch
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    new_branch = f"s3-results-update-{timestamp}"
    repo.git.checkout("-b", new_branch)

    # Download files from S3
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, S3_PREFIX)
            local_path = os.path.join(s3_results_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            S3.download_file(S3_BUCKET, key, local_path)

    # Copy into repo's results/ directory
    repo_results_path = os.path.join(repo_dir, "results")
    if os.path.exists(repo_results_path):
        shutil.rmtree(repo_results_path)
    shutil.copytree(s3_results_dir, repo_results_path)

    # Git commit and push
    repo.git.add("results/")
    repo.index.commit(f"Automated update from S3 at {timestamp}")
    repo.remote().push(refspec=f"{new_branch}:{new_branch}")

    # Create Pull Request
    gh = Github(GITHUB_TOKEN)
    gh_repo = gh.get_repo(GITHUB_REPO)
    gh_repo.create_pull(
        title=f"Automated results update - {timestamp}",
        body="This pull request updates the results/ folder with new outputs from S3.",
        head=new_branch,
        base=GITHUB_BRANCH,
    )

    return {"statusCode": 200, "body": f"Pull request created for branch: {new_branch}"}
