"""Unit tests for training plots Lambda edge cases."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

AWS_DIR = Path(__file__).resolve().parents[1] / "aws"
MODULE_PATH = AWS_DIR / "lambda_api_training_plots.py"


def _load_module():
    env = {"S3_BUCKET": "test-bucket"}
    with patch.dict("os.environ", env, clear=False):
        # Force a fresh import so env is read with BUCKET_NAME set.
        sys.modules.pop("lambda_api_training_plots", None)
        spec = importlib.util.spec_from_file_location(
            "lambda_api_training_plots", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["lambda_api_training_plots"] = module
        with patch("boto3.client", return_value=MagicMock()):
            spec.loader.exec_module(module)
        return module


def test_null_query_and_path_params_do_not_500():
    plots = _load_module()
    event = {
        "httpMethod": "GET",
        "pathParameters": None,
        "queryStringParameters": None,
    }

    with patch.object(plots, "get_training_overview") as mock_overview:
        mock_overview.return_value = {
            "statusCode": 200,
            "headers": {},
            "body": "{}",
        }
        result = plots.lambda_handler(event, None)

    assert result["statusCode"] == 200
    mock_overview.assert_called_once()


def test_missing_plot_head_object_404_returns_404():
    plots = _load_module()
    error = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "HeadObject",
    )
    plots.s3_client.head_object.side_effect = error

    result = plots.get_plot_image("missing_plot.png")

    assert result["statusCode"] == 404
    body = json.loads(result["body"])
    assert body["error"] == "Plot not found"
