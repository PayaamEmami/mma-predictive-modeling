"""Unit tests for SageMaker training-trigger key allowlisting."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

AWS_DIR = Path(__file__).resolve().parents[1] / "aws"
MODULE_PATH = AWS_DIR / "lambda_training_job.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("lambda_training_job", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["lambda_training_job"] = module
    spec.loader.exec_module(module)
    return module


training = _load_training_module()


@pytest.mark.parametrize(
    "s3_key,expected",
    [
        ("data/fight_events.csv", True),
        ("experiments/fight_events.csv", True),
        ("experiments/code.tar.gz", False),
        ("code.tar.gz", False),
        ("data/upcoming_fights.json", False),
        ("data/other.csv", False),
    ],
)
def test_should_start_training_allowlist(s3_key, expected):
    assert training.should_start_training(s3_key) is expected


def test_lambda_handler_ignores_code_tarball():
    event = {
        "Records": [
            {"s3": {"object": {"key": "experiments/code.tar.gz"}}}
        ]
    }

    with patch.object(training, "boto3") as mock_boto3:
        result = training.lambda_handler(event, None)

    assert result["statusCode"] == 200
    assert "Ignored" in result["body"]
    mock_boto3.client.assert_not_called()


def test_lambda_handler_starts_job_for_allowlisted_csv():
    event = {
        "Records": [
            {"s3": {"object": {"key": "data/fight_events.csv"}}}
        ]
    }
    fake_sm = MagicMock()
    fake_sm.create_training_job.return_value = {
        "TrainingJobArn": "arn:aws:sagemaker:us-west-1:123:training-job/test"
    }

    env = {
        "ROLE_ARN": "arn:aws:iam::123:role/sagemaker",
        "SAGEMAKER_PROGRAM": "train.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "s3://bucket/code.tar.gz",
        "S3_BUCKET": "bucket",
        "S3_DATA_KEY": "data/fight_events.csv",
        "S3_RESULTS_PREFIX": "results/",
        "S3_OUTPUT_PATH": "s3://bucket/output/",
    }

    with patch.object(training, "boto3") as mock_boto3, patch.dict(
        training.os.environ, env, clear=False
    ):
        mock_boto3.client.return_value = fake_sm
        result = training.lambda_handler(event, None)

    assert result["statusCode"] == 200
    fake_sm.create_training_job.assert_called_once()
