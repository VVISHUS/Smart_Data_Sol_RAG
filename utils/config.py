"""
Typed configuration loader.

WHY NOT JUST `yaml.safe_load()` INTO A DICT
-------------------------------------------
A raw dict fails *late* and *quietly*. Misspell `min_tables_in_context` as
`min_tables_in_ctx` and you get a `KeyError` twenty minutes into an ingestion run,
or -- worse -- a silent fallback to a default that makes results irreproducible.

Pydantic validates the whole file at startup: wrong type, missing key, or
nonsense value (negative top_k) raises immediately with a precise message. For
a pipeline whose runs cost API calls and minutes, failing in the first 50ms is
worth the extra file.

LAYERING RULE (highest priority wins)
-------------------------------------
    1. environment variables   <- machine-specific reality, never committed
    2. config.yaml             <- the committed, reviewable intent
    3. field defaults below    <- last resort so the code runs without any file

WHY that order: secrets and per-machine overrides must never require editing a
tracked file, or someone will eventually commit their API key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# WHY load .env at import: every entrypoint (CLI, Streamlit, eval harness) needs
# it, and doing it here means no entrypoint can forget to.
load_dotenv()

# WHY resolve the project root from __file__ rather than os.getcwd(): the CLI
# must behave identically whether invoked from the repo root, from app/, or by
# Streamlit (which changes the working directory). Paths in config.yaml are
# relative to the project root, and this is what makes that promise true.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    raw_pdf: Path = Path("data/raw/apple_10q_2022_q3.pdf")
    processed_dir: Path = Path("data/processed")
    figures_dir: Path = Path("data/processed/figures")
    vector_store_dir: Path = Path("chroma_db")
    docstore_path: Path = Path("data/processed/docstore.sqlite3")

    def absolute(self, field: str) -> Path:
        """
        Resolve a configured relative path against the project root.

        WHY a method instead of resolving at validation time: keeping the stored
        values relative means they stay readable in logs and in the methodology
        write-up ("data/processed/figures", not "C:/Users/.../figures"), while
        callers still get an unambiguous absolute path.
        """
        return (PROJECT_ROOT / getattr(self, field)).resolve()


class LLMConfig(BaseModel):
    provider: Literal["google", "openai"] = "google"
    summarizer_model: str = "gemini-3.5-flash-lite"
    answer_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout_s: int = 60
    # WHY THIS EXISTS: Gemini's free tier caps generate-content at a handful of
    # requests per minute (measured: 5/min for gemini-3.6-flash). Enrichment
    # issues ~31 calls back to back, so without pacing most of them 429 and fall
    # back to a degraded summary -- silently. See _RateLimiter in llm/provider.py.
    requests_per_minute: int = 12
    # WHY A SECOND, LOWER BUDGET: quota is per MODEL, not per key. The lite
    # summariser tolerates ~12/min; the stronger answering model is capped at 5.
    # One shared limiter would have to use the lower number and would then slow
    # the 31-call enrichment stage by 2.5x for no reason.
    answer_requests_per_minute: int = 10

    @field_validator("temperature")
    @classmethod
    def _temperature_must_be_low(cls, v: float) -> float:
        """
        Guard the single most damaging misconfiguration in this project.

        WHY: this is a factual-retrieval system over an SEC filing. A non-zero
        temperature makes the model paraphrase numbers it should be copying.
        We allow a small band rather than hard-pinning 0.0 so the value stays
        available as a deliberate experiment, but anything above 0.3 is almost
        certainly a mistake and we refuse to start.
        """
        if not 0.0 <= v <= 0.3:
            raise ValueError(
                f"temperature={v} is unsafe for factual financial QA; "
                "expected 0.0-0.3 (see utils/config.py for rationale)"
            )
        return v


class IngestionConfig(BaseModel):
    min_figure_width_px: int = 40
    min_figure_height_px: int = 40
    max_figure_aspect_ratio: float = 8.0
    min_table_rows: int = 2
    min_table_cols: int = 2




class RetrievalConfig(BaseModel):
    top_k: int = Field(default=6, gt=0)
    min_tables_in_context: int = Field(default=2, ge=0)


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @property
    def api_key(self) -> str:
        """
        Fetch the credential for the selected provider.

        WHY a property rather than a config field: secrets must never be
        loadable from (and therefore accidentally writable to) the YAML file.
        Reading from the environment on demand keeps them out of the committed
        surface entirely, and out of any `repr()` of this object that might be
        logged.
        """
        # WHY WE ACCEPT TWO NAMES FOR THE GOOGLE KEY: Google's own console issues
        # the credential as GEMINI_API_KEY, while `langchain_google_genai` and
        # most tutorials read GOOGLE_API_KEY. Both names are in wide use for the
        # identical secret. Accepting either removes a genuinely confusing
        # first-run failure -- "the key is right there in my .env" -- at the cost
        # of one tuple.
        candidates = (
            ("GOOGLE_API_KEY", "GEMINI_API_KEY")
            if self.provider_is_google
            else ("OPENAI_API_KEY",)
        )

        for var in candidates:
            if key := os.getenv(var, "").strip():
                return key

        raise RuntimeError(
            f"None of {' / '.join(candidates)} is set. "
            "Copy .env.example to .env and fill it in."
        )

    @property
    def provider_is_google(self) -> bool:
        return self.llm.provider == "google"


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Build the validated config object.

    Reads config.yaml if present, then lets `LLM_PROVIDER` from the environment
    override the provider choice.

    WHY the file is optional: the defaults encoded above are a complete, working
    configuration. A reviewer who deletes or mangles config.yaml still gets a
    running system rather than a crash -- the YAML is for *tuning*, not for
    basic operation.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"

    raw: dict = {}
    if cfg_path.exists():
        # WHY safe_load and not load: `yaml.load` can instantiate arbitrary
        # Python objects from a crafted file. Never relevant here, always
        # correct to avoid.
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # WHY this override is applied manually rather than via pydantic-settings:
    # only one field is environment-overridable, and an explicit two lines is
    # clearer to a reviewer than a settings-source customisation hook.
    if env_provider := os.getenv("LLM_PROVIDER", "").strip():
        raw.setdefault("llm", {})["provider"] = env_provider

    return AppConfig(**raw)
