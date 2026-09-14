"""Unit tests for past predictions matching and pagination guards."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

AWS_DIR = Path(__file__).resolve().parents[1] / "aws"
MODULE_PATH = AWS_DIR / "lambda_api_past_predictions.py"


def _load_module():
    env = {
        "BUCKET_NAME": "test-bucket",
        "FIGHT_EVENTS_KEY": "data/fight_events.csv",
    }
    with patch.dict("os.environ", env, clear=False):
        sys.modules.pop("lambda_api_past_predictions", None)
        spec = importlib.util.spec_from_file_location(
            "lambda_api_past_predictions", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["lambda_api_past_predictions"] = module
        with patch("boto3.client", return_value=MagicMock()):
            spec.loader.exec_module(module)
        return module


def test_find_fight_result_strips_event_and_fighter_names():
    past = _load_module()
    fight_events = [
        {
            "EventName": "UFC 300",
            "Fighter1_Name": "Fighter A",
            "Fighter2_Name": "Fighter B",
            "Winner": "1",
            "Method": "KO",
            "Round": "1",
            "Time": "1:00",
        }
    ]

    result = past.find_fight_result(
        fight_events, "UFC 300 ", " Fighter A ", "Fighter B"
    )

    assert result is not None
    assert result["winner_name"] == "Fighter A"


def test_handle_list_past_predictions_clamps_nonpositive_limit():
    past = _load_module()
    past.s3_client.list_objects_v2.return_value = {"Contents": []}

    result = past.handle_list_past_predictions({"page": "0", "limit": "0"}, {})

    assert result["statusCode"] == 200
    body = __import__("json").loads(result["body"])
    assert body["page"] == 1
    assert body["limit"] == 1
    assert body["total_pages"] == 0
