"""The model-provider panel on the generation screen.

These need Streamlit, not SynthEx. They used to sit in
test_openai_compatible_unit.py, which skips itself whenever the interpreter
has no pydantic, so they ran in neither environment: the suite's interpreter
has no pydantic, and the SynthEx runtime has no Streamlit.

What they are for: a provider whose address and model are written down in
PROVIDERS must not ask the reader to type either of them back in, and a
provider that names no model of its own still has to ask.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

KEY = "offline-provider-key-sentinel-0123456789"


class ProviderControlTests(unittest.TestCase):
    """The screen offers free providers and never promises a paid call for one."""

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()

    def synthex(self):
        app = self.app()
        app.selectbox(key="generation_engine").set_value("synthex").run()
        self.assertEqual([str(error) for error in app.exception], [])
        return app

    def test_the_default_provider_asks_for_nothing_but_a_key(self):
        """Groq documents one address and one model, so neither is a question.

        Both are written in PROVIDERS. Drawing them as empty boxes asked the
        reader to retype a constant, and an empty box also blocked the run.
        """
        app = self.synthex()
        self.assertEqual(app.selectbox(key="llm_provider").value, "groq")
        keys = [item.key for item in app.text_input]
        self.assertNotIn("llm_base_url", keys)
        self.assertNotIn("llm_model", keys)
        self.assertIn("llm_api_key", keys)
        self.assertEqual(app.text_input(key="llm_api_key").value, "")
        # Nothing is hidden: the caption says which model the structure goes to.
        captions = " ".join(caption.value for caption in app.caption)
        self.assertIn("Groq", captions)
        self.assertIn("openai/gpt-oss-120b", captions)
        self.assertNotIn("유료 호출", captions)
        self.assertNotIn("유료", app.checkbox(key="generation_consent").label)
        # A key is all that stands between this screen and a run.
        self.assertFalse(any("채우세요" in warning.value for warning in app.warning))

    def test_choosing_gemini_restores_the_paid_wording_and_hides_the_endpoint(self):
        app = self.synthex()
        app.selectbox(key="llm_provider").set_value("gemini").run()
        self.assertEqual([str(error) for error in app.exception], [])
        keys = [item.key for item in app.text_input]
        self.assertNotIn("llm_base_url", keys)
        self.assertNotIn("llm_model", keys)
        captions = " ".join(caption.value for caption in app.caption)
        self.assertIn("Gemini", captions)
        self.assertIn("gemini-3.1-pro-preview", captions)
        self.assertIn("유료 호출", captions)
        self.assertIn("유료", app.checkbox(key="generation_consent").label)

    def test_a_provider_without_a_verified_default_model_asks_for_one(self):
        app = self.synthex()
        app.selectbox(key="llm_provider").set_value("openrouter").run()
        self.assertEqual([str(error) for error in app.exception], [])
        self.assertEqual(app.text_input(key="llm_model").value, "")
        # OpenRouter publishes its address, so only the model is asked for.
        self.assertNotIn("llm_base_url", [item.key for item in app.text_input])
        self.assertTrue(app.button(key="generation_run").disabled)
        warnings = " ".join(warning.value for warning in app.warning)
        self.assertIn("모델 이름을 채우세요", warnings)
        self.assertNotIn("API 주소", warnings)

    def test_switching_provider_clears_the_key_and_blocks_the_run(self):
        app = self.synthex()
        app.text_input(key="llm_api_key").set_value(KEY).run()
        self.assertEqual(app.text_input(key="llm_api_key").value, KEY)
        app.selectbox(key="llm_provider").set_value("openrouter").run()
        self.assertEqual(app.text_input(key="llm_api_key").value, "")
        self.assertFalse(app.session_state["generation_consent"])
        self.assertTrue(app.button(key="generation_run").disabled)

    def test_a_provider_that_publishes_nothing_asks_for_both(self):
        app = self.synthex()
        app.selectbox(key="llm_provider").set_value("custom").run()
        self.assertEqual([str(error) for error in app.exception], [])
        keys = [item.key for item in app.text_input]
        self.assertIn("llm_base_url", keys)
        self.assertIn("llm_model", keys)
        self.assertTrue(app.button(key="generation_run").disabled)
        warnings = " ".join(warning.value for warning in app.warning)
        self.assertIn("API 주소와 모델 이름을 채우세요", warnings)


if __name__ == "__main__":
    unittest.main()
