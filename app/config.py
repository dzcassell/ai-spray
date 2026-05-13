"""Runtime configuration, entirely driven by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Set


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on", "y")


VALID_CATEGORIES = {
    "llm_api", "chatbot_ui", "media_gen", "aggregator", "real_response",
    "mcp_synthetic", "mcp_authed", "mcp_session_sim", "mcp_responder",
}


@dataclass(frozen=True)
class Config:
    # Pacing
    min_interval: float
    max_interval: float

    # Bursts — occasionally fire a cluster of requests back-to-back
    burst_probability: float
    burst_min_size: int
    burst_max_size: int
    burst_gap_min: float
    burst_gap_max: float

    # Scope
    categories: Set[str]
    enable_real_responses: bool

    # HTTP
    http_timeout: float
    max_concurrent: int

    # TLS — DEFAULT OFF because this tool is specifically designed to run
    # behind SASE fabrics / NGFWs that decrypt TLS by re-signing certs
    # with their own CA. Verifying against the container's bundled Mozilla
    # trust store would fail on every inspected flow. Flip to True only
    # if you're running hAIrspray outside any MitM inspection (not the
    # typical case) and want the extra validation.
    tls_verify: bool

    # Observability
    log_level: str
    health_port: int

    @classmethod
    def from_env(cls) -> "Config":
        raw_cats = os.getenv(
            "CATEGORIES",
            "llm_api,chatbot_ui,media_gen,aggregator,real_response,"
            "mcp_synthetic,mcp_authed,mcp_session_sim",
        )
        cats = {c.strip() for c in raw_cats.split(",") if c.strip()}
        bad = cats - VALID_CATEGORIES
        if bad:
            raise ValueError(
                f"Invalid CATEGORIES values {bad}; valid: {sorted(VALID_CATEGORIES)}"
            )

        return cls(
            min_interval=float(os.getenv("MIN_INTERVAL_SEC", "30")),
            max_interval=float(os.getenv("MAX_INTERVAL_SEC", "180")),
            burst_probability=float(os.getenv("BURST_PROBABILITY", "0.15")),
            burst_min_size=int(os.getenv("BURST_MIN_SIZE", "3")),
            burst_max_size=int(os.getenv("BURST_MAX_SIZE", "7")),
            burst_gap_min=float(os.getenv("BURST_GAP_MIN_SEC", "1")),
            burst_gap_max=float(os.getenv("BURST_GAP_MAX_SEC", "4")),
            categories=cats,
            enable_real_responses=_bool(os.getenv("ENABLE_REAL_RESPONSES"), True),
            http_timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "30")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT", "1")),
            tls_verify=_bool(os.getenv("TLS_VERIFY"), False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            health_port=int(os.getenv("HEALTH_PORT", "8080")),
        )
