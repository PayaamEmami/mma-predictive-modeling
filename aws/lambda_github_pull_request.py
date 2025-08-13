import boto3
import tempfile
import os
import shutil
from datetime import datetime, timezone
from github import Github


def get_github_token_from_parameter_store(parameter_name, region_name="us-west-1"):
    """Get GitHub token from Parameter Store"""
    ssm = boto3.client("ssm", region_name=region_name)
    try:
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except Exception as e:
        print(f"Error retrieving GitHub token from Parameter Store: {e}")
        raise e


def lambda_handler(event, context):
    # Use single GitHub token parameter for both main and experimental
    s3_key = event["Records"][0]["s3"]["object"]["key"]
    if s3_key.startswith("experiments/"):
        github_branch = "experimental"
        s3_results_prefix = "experiments/results/"
    else:
        github_branch = "main"
        s3_results_prefix = "results/"

    # Configuration from environment variables
    S3_BUCKET = os.environ.get("S3_BUCKET")
    S3_PREFIX = s3_results_prefix
    S3 = boto3.client("s3")

    # Download files from S3 to temp dir
    tmp_dir = tempfile.mkdtemp()
    s3_results_dir = os.path.join(tmp_dir, "s3_results")
    os.makedirs(s3_results_dir, exist_ok=True)

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

    # Authenticate with GitHub using single Parameter Store token
    GITHUB_TOKEN = get_github_token_from_parameter_store("mpm-github-token")
    GITHUB_REPO = os.environ.get("GITHUB_REPO")
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)

    # Create a new branch from the target branch
    base = repo.get_branch(github_branch)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    new_branch = f"s3-results-update-{timestamp}"
    repo.create_git_ref(ref=f"refs/heads/{new_branch}", sha=base.commit.sha)

    # For each file in s3_results_dir, update or create in results/ in the repo
    for root, _, files in os.walk(s3_results_dir):
        for file in files:
            if file == "done.json":
                continue
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, s3_results_dir)
            repo_path = f"results/{rel_path.replace(os.sep, '/')}"

            with open(local_path, "rb") as f:
                content = f.read()
            try:
                contents = repo.get_contents(repo_path, ref=new_branch)
                repo.update_file(
                    repo_path,
                    f"Automated update from S3 at {timestamp}",
                    content,
                    contents.sha,
                    branch=new_branch,
                )
            except Exception:
                repo.create_file(
                    repo_path,
                    f"Automated update from S3 at {timestamp}",
                    content,
                    branch=new_branch,
                )

    # Create Pull Request
    pr = repo.create_pull(
        title=f"Automated results update - {timestamp}",
        body="This pull request updates the results/ folder with new outputs from S3.",
        head=new_branch,
        base=github_branch,
    )

    shutil.rmtree(tmp_dir)
    return {
        "statusCode": 200,
        "body": f"Pull request created for branch: {new_branch}, PR URL: {pr.html_url}",
    }
