"""What a partial generation run tells the reader.

A run that stops partway used to show one English sentence from the worker
protocol and nothing about where it stopped. The reader could not tell a rate
limit from a broken configuration without opening the saved file.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import presentation  # noqa: E402


def _attempt(*items):
    return {"status": "partial", "result": {"item_failures": list(items)}}


def _stage(stage, message=None, status=None):
    item = {"stage": stage}
    if message is not None:
        item["failure"] = {"category": "ValueError", "message": message}
    if status is not None:
        item["status"] = status
    return item


class ServiceMessagesAreTranslatedTests(unittest.TestCase):
    def test_no_service_message_reaches_the_screen_in_english(self):
        import generation_service as service

        for code, message in service._ERRORS.items():
            with self.subTest(code=code):
                korean = presentation.reason_text(message)
                self.assertNotEqual(korean, message, f"{code} is still shown in English")
                self.assertTrue(korean)

    def test_an_unknown_message_is_shown_as_it_came(self):
        """A blank where a reason belongs is the thing this avoids."""
        self.assertEqual(presentation.reason_text("something nobody mapped"),
                         "something nobody mapped")


class StageFailureTests(unittest.TestCase):
    """The reader is told which stage stopped, and why where we can say why."""

    def test_a_rate_limit_is_named_as_one(self):
        message = ("every item in stage 'routes' failed; first failure was "
                   "OpenAICompatibleError: provider returned HTTP 429 for model "
                   "'openai/gpt-oss-120b': Rate limit reached for model ... "
                   "tokens per minute (TPM): Limit 8000")
        pairs = presentation.stage_failures(_attempt(_stage("routes", message)))
        self.assertEqual(len(pairs), 1)
        stage, reason, raw = pairs[0]
        self.assertEqual(stage, "경로 탐색")
        self.assertIn("분당 한도", reason)
        self.assertEqual(raw, message)

    def test_a_schema_failure_is_named_as_one(self):
        message = ("provider returned HTTP 400: {\"error\":{\"code\":"
                   "\"json_validate_failed\"}}")
        _, reason, _ = presentation.stage_failures(_attempt(_stage("routes", message)))[0]
        self.assertIn("JSON", reason)

    def test_a_stage_with_no_message_still_names_the_stage(self):
        pairs = presentation.stage_failures(
            _attempt(_stage("stitched_routes", status="no_completed_routes")))
        stage, reason, raw = pairs[0]
        self.assertEqual(stage, "경로 이어붙이기")
        self.assertIsNone(reason)
        self.assertIn("끝까지 간 경로가 없습니다", raw)

    def test_a_reason_we_cannot_name_keeps_the_original_text(self):
        """Guessing wrong about a failure is worse than showing it as it came."""
        pairs = presentation.stage_failures(_attempt(_stage("analyses", "something new")))
        stage, reason, raw = pairs[0]
        self.assertEqual(stage, "분석")
        self.assertIsNone(reason)
        self.assertEqual(raw, "something new")

    def test_every_stage_the_worker_can_report_has_a_korean_name(self):
        import generation_worker as worker

        for stage in worker.STAGES + ("cli",):
            with self.subTest(stage=stage):
                self.assertIn(stage, presentation.STAGE_NAMES)

    def test_a_clean_attempt_reports_nothing(self):
        self.assertEqual(presentation.stage_failures({"status": "succeeded", "result": None}), [])
        self.assertEqual(presentation.stage_failures(_attempt()), [])


if __name__ == "__main__":
    unittest.main()
