"""Redact secrets from LLM inputs/outputs and block obvious exfiltration prompts."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Pattern

REDACTED = "[REDACTED]"

# Env vars whose values are treated as literal secrets (substring redaction).
_SENSITIVE_ENV_KEYS = (
    "HF_TOKEN",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "POSTGRES_PASSWORD",
    "MYSQL_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "MINIO_ROOT_PASSWORD",
    "MINIO_ROOT_USER",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_FLASK_SERVER_SECRET_KEY",
    "AIRFLOW_FERNET_KEY",
    "AIRFLOW_DB_PASSWORD",
    "AIRFLOW_WWW_USER_PASSWORD",
    "PGVECTOR_PASSWORD",
    "TRINO_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

# Common secret shapes (applied even when not in env).
_SECRET_PATTERNS: list[tuple[str, Pattern[str]]] = [
    (
        "hf_token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    ),
    (
        "langfuse_public",
        re.compile(r"\bpk-lf-[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "langfuse_secret",
        re.compile(r"\bsk-lf-[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "aws_secret_key",
        re.compile(
            r"(?i)(aws_secret_access_key|secret_access_key)\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{20,}"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{20,}\b", re.IGNORECASE),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "postgres_url",
        re.compile(r"postgres(?:ql)?://[^\s:@/]+:[^\s@/]+@[^\s/]+", re.IGNORECASE),
    ),
    (
        "env_assignment",
        re.compile(
            r"(?i)\b("
            r"HF_TOKEN|LANGSMITH_API_KEY|LANGCHAIN_API_KEY|LANGFUSE_(?:PUBLIC|SECRET)_KEY|"
            r"POSTGRES_PASSWORD|MYSQL_(?:ROOT_)?PASSWORD|MINIO_ROOT_(?:USER|PASSWORD)|"
            r"AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY)|MLFLOW_(?:TRACKING_)?PASSWORD|"
            r"MLFLOW_FLASK_SERVER_SECRET_KEY|AIRFLOW_(?:FERNET_KEY|DB_PASSWORD|WWW_USER_PASSWORD)|"
            r"PGVECTOR_PASSWORD|TRINO_PASSWORD"
            r")\s*[=:]\s*['\"]?[^\s'\"#,]{4,}"
        ),
    ),
    (
        "dotenv_line",
        re.compile(r"(?im)^[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*\s*=\s*.+$"),
    ),
]

_BLOCKED_QUESTION_PATTERNS: list[Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(api[_ -]?key|secret[_ -]?key|access[_ -]?token|auth[_ -]?token)\b",
        r"\b(hf[_ -]?token|huggingface[_ -]?token)\b",
        r"\b(langsmith|langfuse|langchain)[_-]?(api[_ -]?)?key\b",
        r"\b(env(ironment)?[_ -]?(file|var(s)?)|\.env\b)",
        r"\b(show|print|list|reveal|dump|expose|give|tell).{0,40}\b(credential|password|secret|token|api key)\b",
        r"\bwhat (is|are) (the |your )?(password|credentials|secrets|api keys)\b",
        r"\bkubernetes secret\b",
        r"\bplatform-secrets\b",
    )
]

REFUSAL_MESSAGE = (
    "I can't help with revealing credentials, API keys, passwords, or environment configuration. "
    "Ask about your indexed documents or data questions instead."
)


def guardrails_enabled() -> bool:
    raw = os.getenv("GUARDRAILS_ENABLED", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _literal_secrets() -> tuple[str, ...]:
    values: set[str] = set()
    for key in _SENSITIVE_ENV_KEYS:
        value = (os.getenv(key) or "").strip()
        if len(value) >= 4:
            values.add(value)
    extra = os.getenv("GUARDRAILS_EXTRA_SECRETS", "")
    for part in extra.split(","):
        part = part.strip()
        if len(part) >= 4:
            values.add(part)
    return tuple(sorted(values, key=len, reverse=True))


def is_blocked_question(question: str) -> bool:
    if not guardrails_enabled():
        return False
    text = (question or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _BLOCKED_QUESTION_PATTERNS)


def sanitize(text: str | None) -> str:
    """Redact known secrets and common credential patterns from text."""
    if not text or not guardrails_enabled():
        return text or ""

    redacted = text
    for secret in _literal_secrets():
        redacted = redacted.replace(secret, REDACTED)

    for _, pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)

    return redacted


def sanitize_mapping(data: dict) -> dict:
    """Sanitize string values in a shallow dict (API / trace payloads)."""
    if not guardrails_enabled():
        return data
    return {key: sanitize(value) if isinstance(value, str) else value for key, value in data.items()}
