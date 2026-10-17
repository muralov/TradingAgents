"""OpenAI chat models deployed in SAP AI Core's Generative AI Hub."""

import logging
import math
from asyncio import sleep as async_sleep
from email.utils import parsedate_to_datetime
from importlib.metadata import version
from random import uniform
from time import sleep, time
from typing import Any

from packaging.version import Version

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)


def _retry_delay(error, attempt, limit):
    """Return a delay for recoverable SDK errors, or None to propagate them."""
    from openai import APIConnectionError

    status = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    if attempt >= limit or code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return None
    if not isinstance(error, APIConnectionError) and not (
        status in {408, 409, 429} or status is not None and status >= 500
    ):
        return None

    response = getattr(error, "response", None)
    headers = response.headers if response is not None else {}
    delay = None
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            candidate = float(value) * scale
        except ValueError:
            try:
                candidate = parsedate_to_datetime(value).timestamp() - time()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(candidate) and candidate > 0:
            delay = candidate
            break
    if delay is None:
        # Capacity windows need longer waits than the SDK's sub-second retries.
        base = min(10 * 2 ** min(attempt, 4), 120)
        delay = base + uniform(0, base * 0.1)
    logger.warning(
        "SAP AI Core %s; waiting %.1fs before retry %d/%d.",
        "rate limit/capacity (429)" if status == 429 else "temporary API failure",
        delay, attempt + 1, limit,
    )
    return delay


class SAPAICoreClient(BaseLLMClient):
    """Use the SAP SDK for OAuth, resource groups, and deployment selection."""

    def get_llm(self) -> Any:
        if self.base_url:
            raise ValueError(
                "SAP AI Core uses AICORE_BASE_URL for its service endpoint. "
                "Unset backend_url / TRADINGAGENTS_LLM_BACKEND_URL for this provider."
            )

        try:
            from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "SAP AI Core support requires the optional 'sap-ai-sdk-gen' dependency. "
                'Install it with: pip install ".[sap_ai_core]"'
            ) from exc

        sdk_version = version("sap-ai-sdk-gen")
        if Version(sdk_version) < Version("7.4.1"):
            raise ImportError(
                f"Installed sap-ai-sdk-gen {sdk_version} is unsupported; "
                "SAP AI Core requires >=7.4.1 for Responses API deployment routing. "
                'Upgrade in the Python environment running TradingAgents: '
                'python -m pip install --upgrade "sap-ai-sdk-gen[amazon]>=7.4.1,<8"'
            )

        from openai import APIError

        from .factory import _coerce_max_retries

        retry_limit = _coerce_max_retries(self.kwargs.get("max_retries", 6))

        class NormalizedSAPChatOpenAI(ChatOpenAI):
            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                if self.use_responses_api:
                    # SAP's wrapper always supplies n=1, but Responses has no n.
                    payload.pop("n", None)
                return payload

            def invoke(self, input, config=None, **kwargs):
                for attempt in range(retry_limit + 1):
                    try:
                        return normalize_content(super().invoke(input, config, **kwargs))
                    except APIError as exc:
                        delay = _retry_delay(exc, attempt, retry_limit)
                        if delay is None:
                            raise
                        sleep(delay)

            async def ainvoke(self, input, config=None, **kwargs):
                for attempt in range(retry_limit + 1):
                    try:
                        return normalize_content(await super().ainvoke(input, config, **kwargs))
                    except APIError as exc:
                        delay = _retry_delay(exc, attempt, retry_limit)
                        if delay is None:
                            raise
                        await async_sleep(delay)

        # GPT-6 models require Responses for function tools with reasoning.
        # Keep older/custom deployments on Chat Completions.
        llm_kwargs = {
            "proxy_model_name": self.model,
            "use_responses_api": self.model.startswith("gpt-6"),
            # Avoid multiplying the application budget by the SDK retry loop.
            "max_retries": 0,
        }
        for key in ("temperature", "max_tokens", "reasoning_effort", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedSAPChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Model availability is determined by the user's SAP deployments."""
        return validate_model("sap_ai_core", self.model)
