"""
Provider abstraction for chat, vision and embedding models.

WHY THIS INDIRECTION EXISTS
---------------------------
It would be shorter to import `ChatGoogleGenerativeAI` directly wherever a model
is needed. That shortcut has three concrete costs, all of which bite during a
one-day build:

1. RATE LIMITS. Free-tier Gemini throttles aggressively. Summarising ~25 tables
   means ~25 calls in quick succession, and a 429 halfway through wastes every
   call already made. Retry-with-backoff belongs in ONE place, not copy-pasted
   into every call site.
2. PROVIDER RISK. If the Gemini key fails on submission day, switching to
   OpenAI must be a config change, not a refactor. Vendor lock-in inside a
   deadline is an avoidable risk.
3. TESTABILITY. Every module downstream depends on this interface, not on a
   vendor SDK, so tests can substitute a fake without network access.

WHY NOT LANGCHAIN'S `init_chat_model`
-------------------------------------
It covers the chat case but not the multimodal-message shape we need for figure
captioning, and it hides which model is actually being used -- something the
methodology write-up has to state precisely.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Protocol

from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import AppConfig
from utils.logger import get_logger

log = get_logger(__name__)


class LLMProvider(Protocol):
    """
    The contract every provider must satisfy.

    WHY `Protocol` rather than an ABC: structural typing means a test double
    needs only to define these three methods -- no inheritance, no registration.
    It documents the interface for a reader while staying zero-cost at runtime.
    """

    def complete(self, prompt: str) -> str: ...
    def describe_image(self, image_path: Path, prompt: str) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Retry policy -- shared by both providers
# ---------------------------------------------------------------------------
# WHY exponential backoff starting at 2s: provider rate limiters use a sliding
# window. Retrying immediately burns another quota slot and is near-guaranteed to
# fail again; doubling the wait lets the window drain. Capped at 30s so a single
# stuck call cannot stall ingestion for minutes.
#
# WHY `reraise=True`: after the final attempt we want the ORIGINAL provider
# exception (which carries the useful message: quota exhausted vs bad key vs
# malformed request), not tenacity's generic RetryError wrapper.


# WHY WE CLASSIFY ERRORS INSTEAD OF RETRYING EVERYTHING
# -----------------------------------------------------
# This was not a theoretical concern -- it cost a 15-minute stalled ingestion run.
#
# The configured model (`gemini-2.0-flash`) had been retired, so every call
# returned a permanent 404. Retrying a 404 cannot succeed: the model will not
# come back between attempts. But the original policy retried EVERY exception,
# and langchain_google_genai runs its own internal 5-attempt backoff underneath
# ours. The two compounded to roughly 3 minutes per table, times 30 tables.
#
# Worse than the delay, the failure was disguised. A loop that retries forever
# looks like a slow network, not a misconfiguration, so the actual error message
# -- which named the replacement model -- stayed buried in a retry log.
#
# Transient failures (429 rate limit, 5xx, timeouts, connection resets) are worth
# retrying. Permanent ones (404 unknown model, 401/403 bad key, 400 malformed
# request) must fail immediately and loudly, because the fix is a config change
# and no amount of waiting substitutes for it.
_PERMANENT_MARKERS = (
    "not found", "404",
    "no longer available", "is not supported",
    "api key not valid", "permission denied", "unauthenticated",
    "invalid argument", "400",
)


def _is_transient(exc: BaseException) -> bool:
    """
    Decide whether an exception is worth another attempt.

    WHY string matching on the message rather than exception types: the provider
    SDKs raise their own exception hierarchies (google.api_core.exceptions,
    openai.APIError) and wrap them differently depending on the call path. A
    type-based check would have to import and enumerate both SDKs' trees and
    would still miss wrapped cases. Matching the normalised message is less
    elegant but covers every path uniformly.

    WHY we default to True (retry) on an unrecognised error: an unknown failure
    is more likely to be a transient network problem than a permanent config
    error, and the attempt cap bounds the cost of being wrong.
    """
    text = f"{type(exc).__name__} {exc}".casefold()
    return not any(marker in text for marker in _PERMANENT_MARKERS)


_retry_policy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


class _RateLimiter:
    """
    Enforce a minimum interval between outbound API calls.

    WHY THIS IS NECESSARY, AND WHY RETRIES ARE NOT A SUBSTITUTE
    -----------------------------------------------------------
    Gemini's free tier allows 5 generate-content requests per minute. Enrichment
    makes ~31 calls back to back, so without pacing the 6th call onward returns
    429 and the run degrades: in the first attempt, 5 of 30 tables fell back to
    header-only summaries. That failure is quiet -- ingestion "succeeds", the
    index is built, and retrieval quality is simply worse, which is the most
    dangerous kind of bug because nothing looks broken.

    Retrying does not fix it. A retry waits and tries again while the *next*
    table's call is already queued behind it, so the burst rate is unchanged and
    the limiter keeps rejecting. Retries handle a transient spike; they cannot
    fix a sustained rate that exceeds the budget. The only real fix is to slow
    the producer down.

    WHY A BLOCKING SLEEP RATHER THAN A TOKEN BUCKET OR ASYNC QUEUE: the pipeline
    is single-threaded and sequential by design (see summarizer.enrich). The
    entire requirement is "do not issue calls faster than N per minute", which a
    monotonic timestamp and a sleep express exactly. A token bucket would add
    machinery for burst allowance we have no use for.

    WHY `time.monotonic` and not `time.time`: the wall clock can jump backwards
    (NTP correction, DST on some platforms), which would make the computed wait
    negative or absurdly long. Monotonic time only moves forward.
    """

    def __init__(self, requests_per_minute: int):
        # WHY guard against <= 0: a misconfigured 0 would mean an infinite
        # interval and the pipeline would hang forever rather than fail.
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if (remaining := self._min_interval - elapsed) > 0:
            log.debug("rate limiter: sleeping %.1fs", remaining)
            time.sleep(remaining)
        self._last_call = time.monotonic()


def _encode_image(image_path: Path) -> tuple[str, str]:
    """
    Read an image as base64 plus its MIME type, for inline multimodal messages.

    WHY inline base64 rather than uploading to a file API: the images here are
    small (a few hundred KB at most) and inlining avoids a second network
    round-trip, an upload quota, and lifecycle management of remote file handles.

    WHY we infer MIME from the suffix: the vision APIs reject a payload whose
    declared type contradicts its bytes. Our extractor always writes PNG, but
    reading it from the path keeps this function honest if that ever changes.
    """
    suffix = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        suffix, "image/png"
    )
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return data, mime


class GoogleProvider:
    """Gemini-backed implementation."""

    def __init__(self, cfg: AppConfig):
        # WHY import inside __init__ rather than at module top: this keeps the
        # OpenAI SDK from being imported when the user runs with Google (and
        # vice versa). A missing optional dependency then only fails if you
        # actually select that provider.
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        self.cfg = cfg
        key = cfg.api_key  # raises early with a clear message if unset
        # WHY TWO LIMITERS: rate limits are enforced per model, so the cheap
        # bulk model and the stronger answering model have independent budgets.
        self._limiter = _RateLimiter(cfg.llm.requests_per_minute)
        self._answer_limiter = _RateLimiter(cfg.llm.answer_requests_per_minute)

        # WHY two separate chat clients: summarisation and answering are
        # configured independently in config.yaml so the cheap model can do bulk
        # enrichment while a stronger one can be swapped in for answering
        # without touching code.
        self._summarizer = ChatGoogleGenerativeAI(
            model=cfg.llm.summarizer_model,
            google_api_key=key,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.request_timeout_s,
            # WHY max_retries=0: langchain_google_genai retries internally with
            # its own 5-attempt backoff. Left on, it multiplies with our tenacity
            # policy (3 x 5 = 15 attempts) and buries the real error. One retry
            # layer, owned here, is the whole point of this module.
            max_retries=0,
        )
        self._answerer = ChatGoogleGenerativeAI(
            model=cfg.llm.answer_model,
            google_api_key=key,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.request_timeout_s,
            # WHY max_retries=0: langchain_google_genai retries internally with
            # its own 5-attempt backoff. Left on, it multiplies with our tenacity
            # policy (3 x 5 = 15 attempts) and buries the real error. One retry
            # layer, owned here, is the whole point of this module.
            max_retries=0,
        )
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=cfg.llm.embedding_model,
            google_api_key=key,
        )

    @_retry_policy
    def complete(self, prompt: str, *, answering: bool = False) -> str:
        """
        Single-turn text completion.

        WHY the `answering` flag instead of two public methods: the call shape is
        identical and only the underlying model differs. One method with a flag
        keeps the Protocol small while still routing bulk work to the cheap
        model.
        """
        (self._answer_limiter if answering else self._limiter).wait()
        model = self._answerer if answering else self._summarizer
        return model.invoke(prompt).content

    @_retry_policy
    def describe_image(self, image_path: Path, prompt: str) -> str:
        """
        Caption an image using the vision-capable chat model.

        WHY the content list of {type: ...} dicts: this is LangChain's canonical
        multimodal message format. Text and image parts travel in ONE message so
        the model sees the instruction and the pixels together -- sending them as
        separate turns measurably degrades grounding.
        """
        self._limiter.wait()
        data, mime = _encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:{mime};base64,{data}"},
            ]
        )
        return self._summarizer.invoke([message]).content

    @_retry_policy
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        WHY batch (`embed_documents`) rather than looping one text at a time:
        one HTTP round-trip instead of N. For ~60 elements that is the
        difference between a couple of seconds and a couple of minutes, and it
        consumes one rate-limit slot instead of sixty.
        """
        self._limiter.wait()
        return self._embeddings.embed_documents(texts)

    @_retry_policy
    def embed_query(self, text: str) -> list[float]:
        """
        WHY a SEPARATE method from `embed`: Google's embedding model is
        task-aware -- it encodes documents with `RETRIEVAL_DOCUMENT` and queries
        with `RETRIEVAL_QUERY`, producing deliberately asymmetric vectors that
        match better. Using `embed_documents` for a query silently degrades
        recall, and it is a very easy mistake to make.
        """
        self._limiter.wait()
        return self._embeddings.embed_query(text)


class OpenAIProvider:
    """OpenAI-backed fallback. Same contract, different vendor."""

    def __init__(self, cfg: AppConfig):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        self.cfg = cfg
        key = cfg.api_key
        # WHY TWO LIMITERS: rate limits are enforced per model, so the cheap
        # bulk model and the stronger answering model have independent budgets.
        self._limiter = _RateLimiter(cfg.llm.requests_per_minute)
        self._answer_limiter = _RateLimiter(cfg.llm.answer_requests_per_minute)

        # WHY model names are not read from config here: config.yaml's defaults
        # are Gemini model IDs, which are meaningless to OpenAI. Mapping to
        # sensible OpenAI equivalents keeps the fallback genuinely one-switch.
        self._summarizer = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=key,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.request_timeout_s,
            # WHY max_retries=0: langchain_google_genai retries internally with
            # its own 5-attempt backoff. Left on, it multiplies with our tenacity
            # policy (3 x 5 = 15 attempts) and buries the real error. One retry
            # layer, owned here, is the whole point of this module.
            max_retries=0,
        )
        self._answerer = ChatOpenAI(
            model="gpt-4o",
            api_key=key,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.request_timeout_s,
            # WHY max_retries=0: langchain_google_genai retries internally with
            # its own 5-attempt backoff. Left on, it multiplies with our tenacity
            # policy (3 x 5 = 15 attempts) and buries the real error. One retry
            # layer, owned here, is the whole point of this module.
            max_retries=0,
        )
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)

    @_retry_policy
    def complete(self, prompt: str, *, answering: bool = False) -> str:
        (self._answer_limiter if answering else self._limiter).wait()
        model = self._answerer if answering else self._summarizer
        return model.invoke(prompt).content

    @_retry_policy
    def describe_image(self, image_path: Path, prompt: str) -> str:
        self._limiter.wait()
        data, mime = _encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ]
        )
        return self._summarizer.invoke([message]).content

    @_retry_policy
    def embed(self, texts: list[str]) -> list[list[float]]:
        self._limiter.wait()
        return self._embeddings.embed_documents(texts)

    @_retry_policy
    def embed_query(self, text: str) -> list[float]:
        # WHY this exists despite OpenAI's embeddings being symmetric: the rest
        # of the codebase calls `embed_query` unconditionally. Keeping the
        # method here means no call site needs to know which provider is active.
        self._limiter.wait()
        return self._embeddings.embed_query(text)


def get_provider(cfg: AppConfig) -> LLMProvider:
    """
    Factory selecting the configured provider.

    WHY a factory function rather than importing the class directly: it is the
    single place that knows the provider -> class mapping, so adding a third
    provider later touches exactly one line here instead of every call site.
    """
    provider = GoogleProvider(cfg) if cfg.provider_is_google else OpenAIProvider(cfg)
    log.info(
        "LLM provider=%s summarizer=%s answerer=%s",
        cfg.llm.provider,
        cfg.llm.summarizer_model if cfg.provider_is_google else "gpt-4o-mini",
        cfg.llm.answer_model if cfg.provider_is_google else "gpt-4o",
    )
    return provider
