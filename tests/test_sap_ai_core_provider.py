"""SAP AI Core integration at the provider factory and LangChain boundaries."""

import asyncio
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tradingagents.default_config import build_default_config
from tradingagents.llm_clients.factory import create_llm_client, create_tier_client


@pytest.fixture
def sap_sdk(monkeypatch):
    """Simulate the optional SDK boundary without authentication or network calls."""
    captured = {}
    monkeypatch.setattr(
        "tradingagents.llm_clients.sap_ai_core_client.version", lambda name: "7.4.1"
    )

    class ChatOpenAI(BaseChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

        @property
        def _llm_type(self):
            return "fake-sap"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["request"] = kwargs
            if kwargs.get("tools"):
                message = AIMessage(content="", tool_calls=[{
                    "name": "stock_price", "args": {"ticker": "SAP"}, "id": "call-1",
                }])
            else:
                message = AIMessage(content=[{"type": "text", "text": "SAP report"}])
            return ChatResult(generations=[ChatGeneration(message=message)])

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=tools, **kwargs)

    module = ModuleType("gen_ai_hub.proxy.langchain.openai")
    module.ChatOpenAI = ChatOpenAI
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.langchain.openai", module)
    return captured


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["quick", "deep"])
def test_sap_model_from_environment_supports_reports_and_tools(monkeypatch, sap_sdk, tier):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "sap_ai_core")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "gpt-6.1-sol")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "gpt-6.1-sol")
    monkeypatch.setenv("TRADINGAGENTS_OPENAI_REASONING_EFFORT", "high")
    monkeypatch.setenv("TRADINGAGENTS_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("TRADINGAGENTS_MAX_TOKENS", "2048")

    llm = create_tier_client(build_default_config(), tier).get_llm()
    assert sap_sdk["proxy_model_name"] == "gpt-6.1-sol"
    assert sap_sdk["use_responses_api"] is True
    assert sap_sdk["reasoning_effort"] == "high"
    assert sap_sdk["max_retries"] == 0  # One application-owned retry budget.
    assert sap_sdk["max_tokens"] == 2048
    assert "api_key" not in sap_sdk
    assert llm.invoke("Analyze SAP").content == "SAP report"

    response = llm.bind_tools([{"name": "stock_price"}]).invoke("Look up SAP")
    assert response.tool_calls[0]["name"] == "stock_price"
    assert response.tool_calls[0]["args"] == {"ticker": "SAP"}


@pytest.mark.unit
def test_sap_cli_offers_requested_model_without_openai_key(monkeypatch):
    from cli.prompts import ensure_api_key, select_llm_provider
    from tradingagents.llm_clients.model_catalog import get_model_options

    def select(message, **kwargs):
        choice = next(c for c in kwargs["choices"] if c.value == ("sap_ai_core", None))
        return type("Answer", (), {"ask": lambda self: choice.value})()

    monkeypatch.setattr("cli.prompts.questionary.select", select)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert select_llm_provider("sap_ai_core") == ("sap_ai_core", None)
    ensure_api_key("sap_ai_core")
    for tier in ("quick", "deep"):
        assert ("GPT-6.1 Sol (SAP deployment)", "gpt-6.1-sol") in get_model_options(
            "sap_ai_core", tier
        )


@pytest.mark.unit
def test_missing_sap_sdk_explains_installation(monkeypatch):
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.langchain.openai", None)
    with pytest.raises(ImportError, match=r"pip install.*sap_ai_core"):
        create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm()


@pytest.mark.unit
def test_outdated_sap_sdk_explains_upgrade_before_initialization(monkeypatch, sap_sdk):
    monkeypatch.setattr(
        "tradingagents.llm_clients.sap_ai_core_client.version", lambda name: "5.9.0"
    )
    with pytest.raises(ImportError, match=r"5\.9\.0.*7\.4\.1.*pip install --upgrade"):
        create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm()
    assert sap_sdk == {}


@pytest.mark.unit
def test_sap_endpoint_configuration_is_not_silently_ignored():
    with pytest.raises(ValueError, match="AICORE_BASE_URL"):
        create_llm_client(
            "sap_ai_core", "gpt-6.1-sol", base_url="https://wrong-endpoint.example/v1"
        ).get_llm()


@pytest.fixture
def installed_sap_proxy(monkeypatch):
    pytest.importorskip("gen_ai_hub.proxy.langchain.openai")
    from gen_ai_hub.proxy.core.base import BaseProxyClient
    from gen_ai_hub.proxy.langchain.base import BaseAuth

    selections = []

    class Proxy(BaseProxyClient):
        @property
        def request_header(self):
            return {"AI-Resource-Group": "test", "Authorization": "Bearer test"}

        @property
        def deployments(self):
            return []

        @property
        def deployment_class(self):
            return SimpleNamespace

        def select_deployment(self, **kwargs):
            selections.append(kwargs)
            return SimpleNamespace(
                model_name="gpt-6.1-sol", deployment_id="test-deployment",
                url="https://sap.example/deployments/test-deployment", prediction_url=None,
                additional_request_body_kwargs=lambda: {},
            )

    proxy = Proxy()
    monkeypatch.setattr(BaseAuth, "_get_proxy_client", classmethod(lambda cls, values: proxy))
    return selections


@pytest.mark.unit
def test_installed_sap_sdk_selects_model_and_binds_real_tools(installed_sap_proxy):
    llm = create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm()
    assert installed_sap_proxy[0]["model_name"] == "gpt-6.1-sol"
    assert llm.proxy_model_name == "gpt-6.1-sol"
    assert llm.deployment_id == "test-deployment"
    assert llm.use_responses_api is True

    def stock_price(ticker: str) -> str:
        """Look up a stock's price."""
        return "100"

    bound = llm.bind_tools([stock_price])
    assert bound.kwargs["tools"][0]["function"]["name"] == "stock_price"


@pytest.mark.unit
def test_sap_async_reports_are_normalized(sap_sdk):
    llm = create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm()
    assert asyncio.run(llm.ainvoke("Analyze SAP")).content == "SAP report"


@pytest.mark.unit
def test_older_sap_deployments_keep_chat_completions(sap_sdk):
    create_llm_client("sap_ai_core", "gpt-4o-mini").get_llm()
    assert sap_sdk["use_responses_api"] is False


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-astra"])
@pytest.mark.parametrize("async_call", [False, True])
def test_sap_reasoning_and_function_tools_use_responses_api(
    installed_sap_proxy, monkeypatch, model, async_call
):
    from gen_ai_hub.proxy.native.openai.clients import get_current_deployment
    from openai.resources.responses import AsyncResponses, Responses
    from openai.types.responses import Response

    llm = create_llm_client("sap_ai_core", model, reasoning_effort="high").get_llm()

    def stock_price(ticker: str) -> str:
        """Look up a stock's price."""
        return "100"

    create_signature = inspect.signature(Responses.create)

    def responses_create(self, **payload):
        # Validate the real OpenAI SDK signature after SAP deployment routing.
        create_signature.bind(self, **payload)
        assert get_current_deployment().deployment_id == "test-deployment"
        assert payload["model"] == "gpt-6.1-sol"
        assert payload["reasoning"]["effort"] == "high"
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["name"] == "stock_price"
        response = Response.model_validate({
            "id": "resp-test", "created_at": 0, "object": "response", "model": model,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"type": "function_call", "id": "fc-test", "call_id": "call-test",
                        "name": "stock_price", "arguments": '{"ticker":"SAP"}'}],
        })
        return SimpleNamespace(parse=lambda: response)

    def chat_create(**payload):
        pytest.fail("Function tools with reasoning_effort are not supported in /v1/chat/completions")

    async_signature = inspect.signature(AsyncResponses.create)

    async def async_responses_create(self, **payload):
        async_signature.bind(self, **payload)
        return responses_create(self, **payload)

    monkeypatch.setattr(Responses, "create", responses_create)
    monkeypatch.setattr(AsyncResponses, "create", async_responses_create)
    monkeypatch.setattr(llm.client.with_raw_response, "create", chat_create)
    bound = llm.bind_tools([stock_price])
    response = (
        asyncio.run(bound.ainvoke("Look up SAP")) if async_call else bound.invoke("Look up SAP")
    )
    assert response.tool_calls[0]["name"] == "stock_price"
    assert response.tool_calls[0]["args"] == {"ticker": "SAP"}


@pytest.mark.unit
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-4o-mini"])
def test_sap_capacity_waits_then_replays_same_request(
    installed_sap_proxy, monkeypatch, caplog, async_call, model
):
    from unittest.mock import AsyncMock

    import tradingagents.llm_clients.sap_ai_core_client as sap

    waits = []
    monkeypatch.setattr(sap, "sleep", waits.append, raising=False)
    monkeypatch.setattr(sap, "async_sleep", AsyncMock(side_effect=waits.append), raising=False)
    monkeypatch.setattr("openai._base_client.BaseClient._calculate_retry_timeout", lambda *a: 0)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) < 3:
            return httpx.Response(429, headers={"Retry-After": "75"}, json={
                "error": {"code": "no_capacity", "type": "too_many_requests",
                          "message": "High demand"},
            })
        if model.startswith("gpt-6"):
            body = {
                "id": "resp-test", "created_at": 0, "object": "response", "model": model,
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "output": [{"type": "function_call", "id": "fc-test", "call_id": "call-test",
                            "name": "stock_price", "arguments": '{"ticker":"SAP"}'}],
            }
        else:
            body = {
                "id": "chat-test", "created": 0, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": "call-test", "type": "function", "function": {
                            "name": "stock_price", "arguments": '{"ticker":"SAP"}',
                        },
                    }],
                }}],
            }
        return httpx.Response(200, json=body)

    llm = create_llm_client("sap_ai_core", model, max_retries=2).get_llm()
    transport = httpx.MockTransport(respond)
    # Exercise the real SDK retry loop and LangChain exception translation.
    monkeypatch.setattr(llm.root_client, "_client", httpx.Client(transport=transport))
    monkeypatch.setattr(llm.root_async_client, "_client", httpx.AsyncClient(transport=transport))

    def stock_price(ticker: str) -> str:
        """Look up a stock's price."""
        return "100"

    bound = llm.bind_tools([stock_price])
    result = asyncio.run(bound.ainvoke("Look up SAP")) if async_call else bound.invoke("Look up SAP")
    assert result.tool_calls[0]["args"] == {"ticker": "SAP"}
    assert waits == [75, 75]
    assert requests == [requests[0]] * 3
    assert "retry 1/2" in caplog.text


def _sap_error(status=429, code="no_capacity", headers=None):
    from openai import AuthenticationError, InternalServerError, RateLimitError

    error_class = {401: AuthenticationError, 429: RateLimitError, 503: InternalServerError}[status]
    return error_class(
        "test API error", body={"code": code},
        response=httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://sap.example")),
    )


@pytest.mark.unit
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("budget,expected", [(None, 6), (0, 0), (2, 2)])
def test_sap_retry_budget_is_bounded_and_preserves_error(sap_sdk, monkeypatch, async_call, budget, expected):
    from unittest.mock import AsyncMock, Mock

    import tradingagents.llm_clients.sap_ai_core_client as sap

    waits = []
    monkeypatch.setattr(sap, "sleep", waits.append)
    monkeypatch.setattr(sap, "async_sleep", AsyncMock(side_effect=waits.append))
    monkeypatch.setattr(sap, "uniform", lambda *a: 0)
    error = _sap_error()
    generate = Mock(side_effect=error)
    monkeypatch.setattr(sys.modules["gen_ai_hub.proxy.langchain.openai"].ChatOpenAI, "_generate", generate)
    kwargs = {} if budget is None else {"max_retries": budget}
    llm = create_llm_client("sap_ai_core", "gpt-6.1-sol", **kwargs).get_llm()
    with pytest.raises(type(error)) as caught:
        if async_call:
            asyncio.run(llm.ainvoke("SAP"))
        else:
            llm.invoke("SAP")
    assert caught.value is error
    assert generate.call_count == expected + 1
    assert waits == [10, 20, 40, 80, 120, 120][:expected]


@pytest.mark.unit
@pytest.mark.parametrize("status,code", [(401, "invalid_api_key"), (429, "insufficient_quota")])
def test_sap_permanent_errors_are_not_retried(sap_sdk, monkeypatch, status, code):
    from unittest.mock import Mock

    import tradingagents.llm_clients.sap_ai_core_client as sap

    wait = Mock()
    monkeypatch.setattr(sap, "sleep", wait)
    error = _sap_error(status, code)
    generate = Mock(side_effect=error)
    monkeypatch.setattr(sys.modules["gen_ai_hub.proxy.langchain.openai"].ChatOpenAI, "_generate", generate)
    with pytest.raises(type(error)):
        create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm().invoke("SAP")
    assert generate.call_count == 1
    wait.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("headers,expected", [
    ({"Retry-After": "90"}, 90),
    ({"retry-after-ms": "1500", "Retry-After": "10"}, 1.5),
    ({"Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"}, 120),
    ({"Retry-After": "not-a-date"}, 10),
    ({"Retry-After": "NaN"}, 10),
    ({"Retry-After": "inf"}, 10),
    ({"Retry-After": "-5"}, 10),
])
def test_sap_retry_after_formats_and_invalid_values(monkeypatch, headers, expected):
    import tradingagents.llm_clients.sap_ai_core_client as sap

    monkeypatch.setattr(sap, "time", lambda: 0)
    monkeypatch.setattr(sap, "uniform", lambda *a: 0)
    assert sap._retry_delay(_sap_error(headers=headers), 0, 6) == expected


@pytest.mark.unit
def test_sap_transient_server_errors_remain_retryable(monkeypatch):
    import tradingagents.llm_clients.sap_ai_core_client as sap

    monkeypatch.setattr(sap, "uniform", lambda *a: 0)
    assert sap._retry_delay(_sap_error(503, "unavailable"), 0, 6) == 10


@pytest.mark.unit
@pytest.mark.parametrize("async_call", [False, True])
def test_sap_wait_can_be_cancelled(sap_sdk, monkeypatch, async_call):
    from unittest.mock import AsyncMock, Mock

    import tradingagents.llm_clients.sap_ai_core_client as sap

    error = asyncio.CancelledError if async_call else KeyboardInterrupt
    monkeypatch.setattr(sap, "sleep", Mock(side_effect=error))
    monkeypatch.setattr(sap, "async_sleep", AsyncMock(side_effect=error))
    generate = Mock(side_effect=_sap_error())
    monkeypatch.setattr(sys.modules["gen_ai_hub.proxy.langchain.openai"].ChatOpenAI, "_generate", generate)
    llm = create_llm_client("sap_ai_core", "gpt-6.1-sol").get_llm()
    with pytest.raises(error):
        if async_call:
            asyncio.run(llm.ainvoke("SAP"))
        else:
            llm.invoke("SAP")
    assert generate.call_count == 1
