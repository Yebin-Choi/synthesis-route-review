"""Offline tests for the OpenAI-compatible provider path.

No live provider is contacted and no API key exists here. Every exchange is a
fake HTTP response served by ``httpx.MockTransport``, so request assembly,
response parsing, retry conditions and key containment are all checked without
a real call. These are software tests, not chemistry or live-run evidence.
"""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "synthex" / "src"))
sys.path.insert(0, str(ROOT / "test"))
import env_support

# See the note in test_generation_worker_unit: SynthEx runs in its own
# runtime, so the suite's interpreter need not have pydantic.
if not env_support.importable("pydantic"):
    raise unittest.SkipTest(
        "pydantic이 없어 SynthEx를 불러올 수 없습니다. SynthEx는 "
        "generation.local.json에 설정된 자체 런타임에서 돌아갑니다.")


import generation_service as service
import generation_worker as worker
from synthex.application import create_llm_client
from synthex.config import LLMConfig, ManuscriptConfig
from synthex.integrations.openai_compatible import (
    API_KEY_ENVIRONMENT_VARIABLE, OpenAICompatibleClient, OpenAICompatibleError,
)
from synthex.llm import LLMRequest, Message, ToolCall, ToolSpec

KEY = "offline-provider-key-sentinel-0123456789"
BASE_URL = "https://api.groq.test/openai/v1"
MODEL = "openai/gpt-oss-120b"


async def _no_sleep(delay, *args, **kwargs):
    return None


def completion(content="ok", tool_calls=None, usage=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body = {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return body


def adapter(handler, **kwargs):
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("base_url", BASE_URL)
    return OpenAICompatibleClient(MODEL, transport=httpx.MockTransport(handler), **kwargs)


def generate(client, request):
    return asyncio.run(client.generate(request))


def plain_request(**kwargs):
    kwargs.setdefault("system", "you are a planner")
    kwargs.setdefault("messages", (Message(role="user", content="hello"),))
    kwargs.setdefault("temperature", 0.3)
    return LLMRequest(**kwargs)


class RequestAssemblyTests(unittest.TestCase):
    def test_system_messages_tools_and_temperature_become_openai_fields(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["authorization"] = request.headers.get("authorization")
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=completion())

        response = generate(adapter(handler), plain_request(
            messages=(
                Message(role="user", content="plan it"),
                Message(role="assistant", content="calling",
                        tool_calls=(ToolCall(id="call-1", name="read_plan", arguments={"index": 2}),)),
                Message(role="tool", name="read_plan", tool_call_id="call-1", content="step text"),
            ),
            tools=(ToolSpec(name="read_plan", description="read it", parameters={"type": "object"}),)))

        self.assertEqual(seen["url"], f"{BASE_URL}/chat/completions")
        self.assertEqual(seen["authorization"], f"Bearer {KEY}")
        payload = seen["payload"]
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "you are a planner"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "plan it"})
        self.assertEqual(payload["messages"][2]["role"], "assistant")
        self.assertEqual(payload["messages"][2]["tool_calls"], [
            {"id": "call-1", "type": "function",
             "function": {"name": "read_plan", "arguments": '{"index": 2}'}}])
        self.assertEqual(payload["messages"][3],
                         {"role": "tool", "tool_call_id": "call-1", "content": "step text"})
        self.assertEqual(payload["tools"], [
            {"type": "function", "function": {"name": "read_plan", "description": "read it",
                                              "parameters": {"type": "object"}}}])
        self.assertNotIn("response_format", payload)
        self.assertEqual(response.text, "ok")

    def test_response_schema_is_sent_as_strict_json_schema(self):
        seen = {}
        schema = {"type": "object", "properties": {"strategies": {"type": "array"}}}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json=completion())

        generate(adapter(handler), plain_request(response_schema=schema))
        self.assertEqual(seen["payload"]["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "synthex_response", "schema": schema, "strict": True}})

    def test_strict_schema_refusal_downgrades_to_json_object_once_and_says_so(self):
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append(payload["response_format"]["type"])
            if attempts[-1] == "json_schema":
                return httpx.Response(400, json={"error": {"message": "response_format is unsupported"}})
            return httpx.Response(200, json=completion())

        client = adapter(handler)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = generate(client, plain_request(response_schema={"type": "object"}))

        self.assertEqual(attempts, ["json_schema", "json_object"])
        self.assertTrue(client.response_format_downgraded)
        self.assertIn("json_object", response.model)
        self.assertTrue(any("json_object" in str(item.message) for item in caught))
        # Latched: the next call does not re-probe strict json_schema.
        generate(client, plain_request(response_schema={"type": "object"}))
        self.assertEqual(attempts, ["json_schema", "json_object", "json_object"])

    def test_thinking_level_is_reported_not_silently_dropped(self):
        client = adapter(lambda request: httpx.Response(200, json=completion()))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = generate(client, plain_request(thinking_level="medium"))

        self.assertTrue(any("thinking_level" in str(item.message) for item in caught))
        self.assertTrue(client.ignored_features)
        self.assertIn("thinking_level", client.ignored_features[0])
        self.assertIn("thinking_level", response.model)
        self.assertNotIn("thinking", json.dumps(
            client._payload(plain_request(thinking_level="medium"))))


class ResponseParsingTests(unittest.TestCase):
    def test_text_tool_calls_and_usage_map_onto_the_neutral_contract(self):
        body = completion(content="", tool_calls=[
            {"id": "call-9", "type": "function",
             "function": {"name": "apply_edit", "arguments": '{"node": "n1", "keep": true}'}}],
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                   "prompt_tokens_details": {"cached_tokens": 4},
                   "completion_tokens_details": {"reasoning_tokens": 2}})
        response = generate(adapter(lambda request: httpx.Response(200, json=body)), plain_request())

        self.assertEqual(response.text, "")
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual((call.id, call.name), ("call-9", "apply_edit"))
        self.assertEqual(call.arguments, {"node": "n1", "keep": True})
        self.assertIsNone(call.thought_signature)
        self.assertEqual(response.usage.input_tokens, 11)
        self.assertEqual(response.usage.output_tokens, 7)
        self.assertEqual(response.usage.total_tokens, 18)
        self.assertEqual(response.usage.cached_input_tokens, 4)
        self.assertEqual(response.usage.thought_tokens, 2)
        self.assertEqual(response.model, MODEL)

    def test_unusable_output_raises_instead_of_returning_empty_or_invented_results(self):
        cases = {
            "empty": completion(content=""),
            "no_choices": {"choices": []},
            "bad_arguments": completion(content="", tool_calls=[
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{not json"}}]),
            "array_arguments": completion(content="", tool_calls=[
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "[1, 2]"}}]),
            "unsupported_call": completion(content="", tool_calls=[
                {"id": "c", "type": "web_search", "function": {"name": "f"}}]),
            "nameless_call": completion(content="", tool_calls=[
                {"id": "c", "type": "function", "function": {"arguments": "{}"}}]),
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(OpenAICompatibleError):
                    generate(adapter(lambda request, body=body: httpx.Response(200, json=body)),
                             plain_request())

    def test_non_json_body_raises(self):
        with self.assertRaises(OpenAICompatibleError):
            generate(adapter(lambda request: httpx.Response(200, text="<html>nope</html>")),
                     plain_request())

    def test_tool_result_without_a_tool_call_id_fails_clearly(self):
        with self.assertRaises(OpenAICompatibleError):
            generate(adapter(lambda request: httpx.Response(200, json=completion())),
                     plain_request(messages=(Message(role="tool", name="f", content="x"),)))


class RetryTests(unittest.TestCase):
    def test_only_429_and_5xx_are_retried(self):
        with patch.object(asyncio, "sleep", _no_sleep):
            for status in (429, 503):
                with self.subTest(status=status):
                    calls = []

                    def handler(request, calls=calls, status=status):
                        calls.append(status)
                        if len(calls) < 3:
                            return httpx.Response(status, text="slow down")
                        return httpx.Response(200, json=completion())

                    self.assertEqual(generate(adapter(handler), plain_request()).text, "ok")
                    self.assertEqual(len(calls), 3)
            for status in (401, 403, 422):
                with self.subTest(status=status):
                    calls = []

                    def handler(request, calls=calls, status=status):
                        calls.append(status)
                        return httpx.Response(status, text="refused")

                    with self.assertRaises(OpenAICompatibleError):
                        generate(adapter(handler), plain_request())
                    self.assertEqual(len(calls), 1)

    def test_exhausted_retries_raise_rather_than_return_nothing(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(500, text="boom")

        with patch.object(asyncio, "sleep", _no_sleep), self.assertRaises(OpenAICompatibleError):
            generate(adapter(handler, max_attempts=3), plain_request())
        self.assertEqual(len(calls), 3)

    def test_transport_failures_are_retried_then_raised(self):
        calls = []

        def handler(request):
            calls.append(1)
            raise httpx.ConnectError("refused", request=request)

        with patch.object(asyncio, "sleep", _no_sleep), self.assertRaises(OpenAICompatibleError):
            generate(adapter(handler, max_attempts=2), plain_request())
        self.assertEqual(len(calls), 2)


class KeyContainmentTests(unittest.TestCase):
    def test_the_key_stays_out_of_urls_exceptions_and_feature_notes(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            # A provider that echoes the key back must not leak it onward.
            return httpx.Response(401, text=f"invalid api key {KEY}")

        client = adapter(handler)
        with self.assertRaises(OpenAICompatibleError) as caught:
            generate(client, plain_request(thinking_level="high"))

        self.assertNotIn(KEY, seen["url"])
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertTrue(all(KEY not in note for note in client.ignored_features))

    def test_a_missing_key_fails_loudly(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenAICompatibleError):
                OpenAICompatibleClient(MODEL, base_url=BASE_URL)
        with patch.dict("os.environ", {API_KEY_ENVIRONMENT_VARIABLE: KEY}, clear=True):
            self.assertEqual(OpenAICompatibleClient(MODEL, base_url=BASE_URL).model, MODEL)

    def test_a_blank_model_or_base_url_fails_loudly(self):
        for arguments in ({"model": "", "base_url": BASE_URL}, {"model": MODEL, "base_url": ""}):
            with self.subTest(**arguments):
                with self.assertRaises(OpenAICompatibleError):
                    OpenAICompatibleClient(arguments["model"], base_url=arguments["base_url"],
                                           api_key=KEY)


class ConfigurationTests(unittest.TestCase):
    def test_base_url_is_required_for_openai_compatible_and_refused_for_gemini(self):
        with self.assertRaises(ValueError):
            LLMConfig(provider="openai_compatible")
        with self.assertRaises(ValueError):
            LLMConfig(provider="gemini", base_url=BASE_URL)
        self.assertEqual(LLMConfig(provider="openai_compatible", base_url=BASE_URL,
                                   model=MODEL).base_url, BASE_URL)

    def test_the_default_config_is_still_the_unchanged_gemini_one(self):
        default = LLMConfig()
        self.assertEqual(default.provider, "gemini")
        self.assertIsNone(default.base_url)
        self.assertEqual(default.model, "gemini-3.1-pro-preview")

    def test_create_llm_client_follows_the_provider(self):
        config = ManuscriptConfig.model_validate({
            "llm": {"provider": "openai_compatible", "base_url": BASE_URL, "model": MODEL},
            "completion": {"enabled": False}})
        with patch.dict("os.environ", {API_KEY_ENVIRONMENT_VARIABLE: KEY}, clear=True):
            client = create_llm_client(config)
        self.assertIsInstance(client, OpenAICompatibleClient)
        self.assertEqual(client.url, f"{BASE_URL}/chat/completions")
        self.assertEqual(client.model, MODEL)


class SelectionPlumbingTests(unittest.TestCase):
    """The nonsecret provider choice travels with the request; the key does not."""

    def selection(self, **overrides):
        value = {"provider": "openai_compatible", "base_url": BASE_URL, "model": MODEL}
        value.update(overrides)
        return value

    def test_valid_selections_are_normalized(self):
        self.assertIsNone(service._llm_selection(None))
        self.assertEqual(service._llm_selection(self.selection()), self.selection())
        gemini = {"provider": "gemini", "base_url": None, "model": "gemini-3.1-pro-preview"}
        self.assertEqual(service._llm_selection(gemini), gemini)
        local = self.selection(base_url="http://localhost:11434/v1")
        self.assertEqual(service._llm_selection(local), local)

    def test_unsafe_or_malformed_selections_are_refused(self):
        cases = {
            "unknown_provider": self.selection(provider="anthropic"),
            "missing_base_url": self.selection(base_url=None),
            "gemini_with_base_url": {"provider": "gemini", "base_url": BASE_URL, "model": "m"},
            "plain_http_remote": self.selection(base_url="http://api.groq.test/openai/v1"),
            "credential_in_url": self.selection(base_url="https://user:pw@api.groq.test/v1"),
            "space_in_url": self.selection(base_url="https://api groq.test/v1"),
            "blank_model": self.selection(model=""),
            "newline_model": self.selection(model="a\nb"),
            "extra_field": {**self.selection(), "api_key": KEY},
        }
        for name, value in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(service._Failure):
                    service._llm_selection(value)

    def test_each_provider_reads_one_credential_variable_in_the_child_only(self):
        for provider, variable in service.KEY_VARIABLES.items():
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                env = service._child_environment(sys.executable, directory, KEY, variable)
                self.assertEqual(env[variable], KEY)
                self.assertFalse(service.KEY_VARIABLE_NAMES - {variable} & set(env))
        with tempfile.TemporaryDirectory() as directory:
            env = service._child_environment(sys.executable, directory)
            self.assertFalse(service.KEY_VARIABLE_NAMES & set(env))

    def test_an_unknown_credential_variable_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(service._Failure):
                service._child_environment(sys.executable, directory, KEY, "OPENAI_API_KEY")

    def test_the_worker_request_never_carries_the_key(self):
        request = {"schema_version": "1.0", "operation": "generate", "engine": "synthex",
                   "request_id": "r1", "context_id": "ctx", "target": "CCO",
                   "configuration_id": "a" * 64, "llm": self.selection(),
                   "settings": {"fingerprint": "b" * 64, "config": str(ROOT / "x.json"),
                                "config_sha256": "c" * 64}}
        text = service._json_bytes(request).decode("utf-8")
        self.assertNotIn(KEY, text)
        self.assertFalse(service._contains_secret(text, KEY))
        self.assertEqual(worker._key_variable(request), "SYNTHEX_LLM_API_KEY")
        self.assertEqual(worker._key_variable({"llm": None}), "GEMINI_API_KEY")
        self.assertEqual(worker._key_variable({}), "GEMINI_API_KEY")

    def test_the_worker_rejects_a_selection_for_the_wrong_engine_or_shape(self):
        base = {"schema_version": "1.0", "operation": "generate", "engine": "synthex",
                "request_id": "r1", "context_id": "ctx", "target": "CCO",
                "configuration_id": "a" * 64,
                "settings": {"fingerprint": "b" * 64, "config": str(ROOT / "x.json"),
                             "config_sha256": "c" * 64}}
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(worker._validate_request(deepcopy(base)), base)
            with self.assertRaises(service._Failure):
                worker._validate_request({**deepcopy(base), "engine": "aizynthfinder",
                                          "llm": self.selection()})
            with self.assertRaises(service._Failure):
                worker._validate_request({**deepcopy(base), "llm": {"provider": "openai_compatible"}})

    def test_only_the_selected_providers_variable_may_be_present(self):
        base = {"schema_version": "1.0", "operation": "generate", "engine": "synthex",
                "request_id": "r1", "context_id": "ctx", "target": "CCO",
                "configuration_id": "a" * 64, "llm": self.selection(),
                "settings": {"fingerprint": "b" * 64, "config": str(ROOT / "x.json"),
                             "config_sha256": "c" * 64}}
        with patch.dict("os.environ", {"SYNTHEX_LLM_API_KEY": KEY}, clear=True):
            self.assertEqual(worker._validate_request(deepcopy(base))["llm"], self.selection())
        with patch.dict("os.environ", {"GEMINI_API_KEY": KEY}, clear=True):
            with self.assertRaises(service._Failure):
                worker._validate_request(deepcopy(base))

    def test_the_worker_applies_only_the_provider_choice_to_the_loaded_config(self):
        # Reuse the offline worker fixture rather than rebuild its asset tree.
        from test_generation_worker_unit import WorkerFixture

        case = WorkerFixture("run")
        case.setUp()
        self.addCleanup(case.doCleanups)
        settings = case.request["settings"]

        unchanged, _ = worker._full_config(settings)
        self.assertEqual(unchanged.llm.provider, "gemini")
        self.assertIsNone(unchanged.llm.base_url)
        self.assertEqual(unchanged.llm.model, "gemini-3.1-pro-preview")

        switched, _ = worker._full_config(settings, self.selection())
        self.assertEqual(switched.llm.provider, "openai_compatible")
        self.assertEqual(switched.llm.base_url, BASE_URL)
        self.assertEqual(switched.llm.model, MODEL)
        # Nothing outside the provider triple moved.
        ignore = {"provider", "base_url", "model"}
        self.assertEqual(switched.llm.model_dump(exclude=ignore),
                         unchanged.llm.model_dump(exclude=ignore))
        self.assertEqual(switched.agents.model_dump(), unchanged.agents.model_dump())
        self.assertEqual(switched.completion.model_dump(), unchanged.completion.model_dump())

        for bad in (self.selection(base_url="http://api.groq.test/v1"), self.selection(model="")):
            with self.subTest(selection=bad), self.assertRaises(service._Failure):
                worker._full_config(settings, bad)

    def test_a_provider_change_is_a_different_readiness_configuration(self):
        first = service.GenerationService._cache_key("synthex", "f" * 64, self.selection())
        second = service.GenerationService._cache_key("synthex", "f" * 64,
                                                      self.selection(model="other"))
        none = service.GenerationService._cache_key("synthex", "f" * 64, None)
        self.assertEqual(len({first, second, none}), 3)


if __name__ == "__main__":
    unittest.main()
