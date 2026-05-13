"""Prompt mode — send a real user prompt to specific models and return
structured responses.

This is a different execution path from the scheduler + fire-all. Those
paths pick a random benign prompt from ``prompts.py`` and treat any
response as success (the point is flow generation, not content). The
prompt mode here takes a user-supplied prompt, targets *specific*
model/provider pairs, and returns the full response body so the UI can
render it.

Two kinds of providers:

* **Keyless** — Pollinations (text + image) and DuckDuckGo AI Chat.
  Work out of the box, no account needed. Rate-limited by the upstream.
* **Keyed** — Free-tier API endpoints from major LLM vendors. The user
  signs up on the vendor's site, gets a free API key, pastes it into
  the hAIrspray UI. Key is persisted to the key store and passed as
  ``Authorization: Bearer`` (or vendor-specific header) on each
  request.

The keyed-provider catalog is in ``KEYED_PROVIDERS`` below; the runner
dispatches on ``provider``. ``needs_key`` + ``signup_url`` surface in
``targets_catalogue()`` so the UI can render a "Get a key at ..." link
for unconfigured providers.
"""
from __future__ import annotations

import random
import time
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from .providers import BROWSER_UAS, stream_read_capped


# ---------------------------------------------------------------------------
# Model catalogue — the only keyless, prompt-capable targets.
# ---------------------------------------------------------------------------

# Each entry: (provider id, model id, kind)
# - provider id: "pollinations-text" | "pollinations-image" | "duckduckgo"
# - model id: the specific model within that provider
# - kind: "text" | "image"
#
# Pollinations model list verified against their 2026 APIDOCS.md
# (github.com/pollinations/pollinations/blob/main/APIDOCS.md). The
# older aliases `llama`, `claude`, `qwen`, `flux-realism` were retired
# when they moved to the gen.pollinations.ai gateway. `kimi` / `deepseek`
# / `glm` are the current community-listed keyless models.
PROMPT_TARGETS: list[dict[str, str]] = [
    # Pollinations text — current keyless-tolerant models
    {"provider": "pollinations-text", "model": "openai",   "kind": "text"},
    {"provider": "pollinations-text", "model": "mistral",  "kind": "text"},
    {"provider": "pollinations-text", "model": "kimi",     "kind": "text"},
    {"provider": "pollinations-text", "model": "deepseek", "kind": "text"},
    {"provider": "pollinations-text", "model": "glm",      "kind": "text"},
    # Pollinations image — 2026 roster, keyless-tolerant subset
    {"provider": "pollinations-image", "model": "flux",       "kind": "image"},
    {"provider": "pollinations-image", "model": "turbo",      "kind": "image"},
    {"provider": "pollinations-image", "model": "nanobanana", "kind": "image"},
    # DuckDuckGo AI Chat (4 models via their duckchat gateway)
    {"provider": "duckduckgo", "model": "gpt-4o-mini",                                "kind": "text"},
    {"provider": "duckduckgo", "model": "claude-3-haiku-20240307",                    "kind": "text"},
    {"provider": "duckduckgo", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",    "kind": "text"},
    {"provider": "duckduckgo", "model": "mistralai/Mistral-Small-24B-Instruct-2501",  "kind": "text"},
]


# ---------------------------------------------------------------------------
# Keyed providers — free-tier API endpoints that need a user-supplied key.
# ---------------------------------------------------------------------------
#
# Each entry:
#   provider    — stable slug used as the key-store id and UI target
#                 prefix. Matches PROMPT_TARGETS.provider.
#   label       — human display name
#   signup_url  — direct link the UI offers for getting a key
#   models      — list of model ids to expose as individual targets
#   kind        — "text" (all keyed providers are chat/text only; images
#                 are keyless-only for now)
#   shape       — which request runner to use. See _run_keyed_* below.
#                 "openai-compatible" covers ~80% of modern LLM APIs.
#                 Vendor-specific shapes (gemini, cohere, anthropic)
#                 have dedicated runners.
#   host        — upstream host, used for per-host serialization in
#                 the fan-out.
#   extra       — optional provider-specific knobs (base_url override,
#                 auth header name, api-version, etc.)

KEYED_PROVIDERS: list[dict[str, Any]] = [
    # --- OpenAI-compatible (Authorization: Bearer <key>, /v1/chat/completions)
    {
        "provider": "google",
        "label":    "Google Gemini",
        "signup_url": "https://aistudio.google.com/apikey",
        "shape":    "gemini",   # uses ?key=<key>, custom body shape
        "host":     "generativelanguage.googleapis.com",
        "models":   ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "kind":     "text",
    },
    {
        "provider": "groq",
        "label":    "Groq",
        "signup_url": "https://console.groq.com/keys",
        "shape":    "openai-compatible",
        "host":     "api.groq.com",
        "extra":    {"base_url": "https://api.groq.com/openai/v1"},
        "models":   ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "kind":     "text",
    },
    {
        "provider": "mistral",
        "label":    "Mistral AI",
        "signup_url": "https://console.mistral.ai/api-keys",
        "shape":    "openai-compatible",
        "host":     "api.mistral.ai",
        "extra":    {"base_url": "https://api.mistral.ai/v1"},
        "models":   ["mistral-small-latest", "mistral-large-latest",
                     "open-mistral-nemo"],
        "kind":     "text",
    },
    {
        "provider": "cohere",
        "label":    "Cohere",
        "signup_url": "https://dashboard.cohere.com/api-keys",
        "shape":    "cohere",   # /v2/chat, message field different
        "host":     "api.cohere.com",
        "models":   ["command-r", "command-r-plus", "command-r7b"],
        "kind":     "text",
    },
    {
        "provider": "openrouter",
        "label":    "OpenRouter",
        "signup_url": "https://openrouter.ai/keys",
        "shape":    "openai-compatible",
        "host":     "openrouter.ai",
        "extra":    {"base_url": "https://openrouter.ai/api/v1"},
        "models":   ["google/gemini-flash-1.5:free",
                     "meta-llama/llama-3.3-70b-instruct:free",
                     "openai/gpt-4o-mini"],
        "kind":     "text",
    },
    {
        "provider": "huggingface",
        "label":    "Hugging Face",
        "signup_url": "https://huggingface.co/settings/tokens",
        "shape":    "hf-router",   # router.huggingface.co chat completions
        "host":     "router.huggingface.co",
        "extra":    {"base_url": "https://router.huggingface.co/v1"},
        "models":   ["meta-llama/Llama-3.3-70B-Instruct",
                     "mistralai/Mistral-7B-Instruct-v0.3",
                     "Qwen/Qwen2.5-72B-Instruct"],
        "kind":     "text",
    },
    {
        "provider": "together",
        "label":    "Together AI",
        "signup_url": "https://api.together.ai/settings/api-keys",
        "shape":    "openai-compatible",
        "host":     "api.together.xyz",
        "extra":    {"base_url": "https://api.together.xyz/v1"},
        # Mixtral-8x7B-Instruct-v0.1 was retired after Mistral pulled
        # Mixtral hosting rights — removed.
        "models":   ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "kind":     "text",
    },
    {
        "provider": "cerebras",
        "label":    "Cerebras",
        "signup_url": "https://cloud.cerebras.ai/platform/",
        "shape":    "openai-compatible",
        "host":     "api.cerebras.ai",
        "extra":    {"base_url": "https://api.cerebras.ai/v1"},
        # llama-3.3-70b (hyphenated) was never a Cerebras slug — their
        # naming convention uses no dot-separator on the size.
        # Discovery will populate the real catalog.
        "models":   ["llama3.1-8b", "llama3.1-70b"],
        "kind":     "text",
    },
    {
        "provider": "sambanova",
        "label":    "SambaNova",
        "signup_url": "https://cloud.sambanova.ai/apis",
        "shape":    "openai-compatible",
        "host":     "api.sambanova.ai",
        "extra":    {"base_url": "https://api.sambanova.ai/v1"},
        # Llama 3.1 Instruct models were deprecated in the 2025 Samba
        # sweep. Current catalog varies by account tier — leaving a
        # placeholder that discovery replaces immediately.
        "models":   ["Meta-Llama-3.3-70B-Instruct"],
        "kind":     "text",
    },
    {
        "provider": "hyperbolic",
        "label":    "Hyperbolic",
        "signup_url": "https://app.hyperbolic.xyz/settings",
        "shape":    "openai-compatible",
        "host":     "api.hyperbolic.xyz",
        "extra":    {"base_url": "https://api.hyperbolic.xyz/v1"},
        "models":   ["meta-llama/Meta-Llama-3.1-70B-Instruct",
                     "Qwen/Qwen2.5-72B-Instruct"],
        "kind":     "text",
    },
    {
        "provider": "deepseek",
        "label":    "DeepSeek",
        "signup_url": "https://platform.deepseek.com/api_keys",
        "shape":    "openai-compatible",
        "host":     "api.deepseek.com",
        "extra":    {"base_url": "https://api.deepseek.com"},
        "models":   ["deepseek-chat", "deepseek-reasoner"],
        "kind":     "text",
    },
    {
        "provider": "xai",
        "label":    "xAI Grok",
        "signup_url": "https://console.x.ai/",
        "shape":    "openai-compatible",
        "host":     "api.x.ai",
        "extra":    {"base_url": "https://api.x.ai/v1"},
        # grok-2-latest and grok-beta were retired when Grok 3 shipped.
        # Best-guess current slugs; discovery will correct them.
        "models":   ["grok-3", "grok-3-mini"],
        "kind":     "text",
    },
    {
        "provider": "ai21",
        "label":    "AI21",
        "signup_url": "https://studio.ai21.com/account/api-key",
        "shape":    "openai-compatible",
        "host":     "api.ai21.com",
        "extra":    {"base_url": "https://api.ai21.com/studio/v1"},
        "models":   ["jamba-large", "jamba-mini"],
        "kind":     "text",
    },
    {
        "provider": "fireworks",
        "label":    "Fireworks AI",
        "signup_url": "https://fireworks.ai/account/api-keys",
        "shape":    "openai-compatible",
        "host":     "api.fireworks.ai",
        "extra":    {"base_url": "https://api.fireworks.ai/inference/v1"},
        # Mixtral-8x7B retired for the same reason as Together.
        "models":   ["accounts/fireworks/models/llama-v3p3-70b-instruct"],
        "kind":     "text",
    },
    {
        # NVIDIA NIM — build.nvidia.com hosted catalog. Signup gets you
        # 1,000 free inference credits via the NVIDIA Developer Program;
        # the catalog includes 100+ models (Llama 3/4, Nemotron, DeepSeek
        # R1, Kimi K2, Qwen 3.5, Mistral). OpenAI-compatible.
        "provider": "nvidia",
        "label":    "NVIDIA NIM",
        "signup_url": "https://build.nvidia.com/",
        "shape":    "openai-compatible",
        "host":     "integrate.api.nvidia.com",
        "extra":    {"base_url": "https://integrate.api.nvidia.com/v1"},
        "models":   [
            "meta/llama-3.3-70b-instruct",
            "deepseek-ai/deepseek-r1",
            "moonshotai/kimi-k2",
        ],
        "kind":     "text",
    },
    {
        # GitHub Models — free tier for every GitHub user (PAT with the
        # models:read scope). Chat endpoint and catalog live on DIFFERENT
        # paths: chat at /inference/chat/completions, catalog at
        # /catalog/models (plain JSON array, not {data:[...]} wrapped).
        # discovery.py handles both response shapes.
        "provider": "github",
        "label":    "GitHub Models",
        "signup_url": "https://github.com/marketplace/models",
        "shape":    "openai-compatible",
        "host":     "models.github.ai",
        "extra":    {
            "base_url":      "https://models.github.ai/inference",
            "discovery_url": "https://models.github.ai/catalog/models",
        },
        "models":   [
            "openai/gpt-4.1",
            "openai/gpt-4o",
            "meta/Meta-Llama-3.1-70B-Instruct",
        ],
        "kind":     "text",
    },
    {
        # Anthropic — used by the Agents tab (random-sprinkle coder
        # prompts) via app/agents.py, which knows how to set
        # 'x-api-key' + 'anthropic-version' headers on /v1/messages.
        #
        # IMPORTANT: this entry is marked openai-compatible, but
        # Anthropic is NOT actually OpenAI-compatible — they use
        # x-api-key (not Bearer auth) and a different request body
        # shape. Discovery against /v1/models will currently fail
        # because discovery.py only sets 'Authorization: Bearer'.
        # Prompt & Fire fan-out against this entry will likewise
        # fail. The Agents tab works fine because it has its own
        # fire path that handles Anthropic's auth correctly.
        # Marking this as a TODO: add an 'anthropic' shape to
        # discovery.py so Prompt & Fire / Models discovery work too.
        "provider": "anthropic",
        "label":    "Anthropic",
        "signup_url": "https://console.anthropic.com/settings/keys",
        "shape":    "openai-compatible",
        "host":     "api.anthropic.com",
        "extra":    {"base_url": "https://api.anthropic.com/v1"},
        "models":   [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
        ],
        "kind":     "text",
    },
    {
        # Cursor — User API Key from the Cursor Integrations Dashboard.
        # The official API is at api.cursor.com/v0/* (verified April
        # 2026 — earlier hAIrspray builds had api2.cursor.sh which is
        # the legacy session-token endpoint, not the public API key
        # surface). Auth: Authorization: Bearer <token>. Used primarily
        # by the Agents tab; appears in Prompt & Fire for parity though
        # Cursor's API is closer to its own bespoke surface than to
        # OpenAI chat-completions, so Prompt & Fire fan-outs against
        # it may behave oddly until/unless we add a per-shape runner.
        "provider": "cursor",
        "label":    "Cursor",
        "signup_url": "https://cursor.com/dashboard?tab=integrations",
        "shape":    "openai-compatible",
        "host":     "api.cursor.com",
        "extra":    {"base_url": "https://api.cursor.com/v0"},
        "models":   [
            "auto",  # Cursor's default routing
            "claude-sonnet-4-5",
            "gpt-4o",
        ],
        "kind":     "text",
    },
]


def keyed_providers() -> list[dict[str, Any]]:
    """Return the keyed-provider catalog (shape used by UI)."""
    return [
        {
            "provider":   p["provider"],
            "label":      p["label"],
            "signup_url": p["signup_url"],
            "models":     list(p["models"]),
            "kind":       p["kind"],
        }
        for p in KEYED_PROVIDERS
    ]


def _keyed_entry(provider: str) -> dict[str, Any] | None:
    for p in KEYED_PROVIDERS:
        if p["provider"] == provider:
            return p
    return None


def target_id(provider: str, model: str) -> str:
    """Stable opaque id for (provider, model) pairs used by the UI."""
    return f"{provider}::{model}"


def targets_catalogue(
    key_presence: dict[str, bool] | None = None,
    models_override: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Shape returned from GET /api/prompt/targets.

    Includes both keyless and keyed provider×model entries. Keyed
    entries carry ``needs_key=True``, a ``signup_url``, and (if
    ``key_presence`` is supplied) a ``present`` flag the UI uses to
    decide whether the checkbox is enabled.

    ``models_override`` lets callers (typically web.py, fed from the
    KeyStore's cached catalog) substitute the hard-coded model lists
    in ``KEYED_PROVIDERS`` with freshly-discovered ones. Only
    providers with a non-empty override list are replaced; others
    fall back to the KEYED_PROVIDERS defaults.
    """
    kp = key_presence or {}
    mo = models_override or {}
    out: list[dict[str, Any]] = []
    # Keyless entries — always present, no key needed.
    for t in PROMPT_TARGETS:
        out.append({
            "id":        target_id(t["provider"], t["model"]),
            "provider":  t["provider"],
            "model":     t["model"],
            "kind":      t["kind"],
            "label":     _display_label(t["provider"], t["model"]),
            "needs_key": False,
            "present":   True,
        })
    # Keyed entries — one per (provider, model) pair. If a discovered
    # catalog exists for this provider, use it; otherwise use the
    # hard-coded defaults.
    for p in KEYED_PROVIDERS:
        provider = p["provider"]
        discovered = mo.get(provider)
        if discovered:
            models = discovered
            source = "discovered"
        else:
            models = p["models"]
            source = "default"
        for m in models:
            out.append({
                "id":         target_id(provider, m),
                "provider":   provider,
                "model":      m,
                "kind":       p["kind"],
                "label":      _display_label(provider, m),
                "needs_key":  True,
                "signup_url": p["signup_url"],
                "present":    bool(kp.get(provider, False)),
                "source":     source,
            })
    return out


def _display_label(provider: str, model: str) -> str:
    if provider == "pollinations-text":
        return f"Pollinations-Text · {model}"
    if provider == "pollinations-image":
        return f"Pollinations-Image · {model}"
    if provider == "duckduckgo":
        # DDG model strings are long (e.g. "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        short = model.split("/")[-1]
        return f"DuckDuckGo · {short}"
    # Keyed providers use their display label if we can find one.
    entry = _keyed_entry(provider)
    if entry is not None:
        # Trim long vendor-style model ids for readability in the UI.
        short = model.split("/")[-1] if "/" in model else model
        return f"{entry['label']} · {short}"
    return f"{provider} · {model}"


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    target_id: str
    provider: str
    model: str
    kind: str            # "text" | "image" | "error"
    label: str
    ok: bool
    status: int | None
    latency_ms: int
    url: str | None      # for images, this IS the image; for text, the source URL
    body: str | None     # text response or error message
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-provider execution
# ---------------------------------------------------------------------------

async def _run_pollinations_text(
    client: httpx.AsyncClient, prompt: str, model: str,
) -> PromptResult:
    started = time.monotonic()
    url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}"
    params = {"model": model}
    headers = {
        "User-Agent": random.choice(BROWSER_UAS),
        "Accept": "text/plain, */*",
    }
    tid = target_id("pollinations-text", model)
    label = _display_label("pollinations-text", model)
    try:
        r = await client.get(url, headers=headers, params=params)
        body = r.text if r.status_code == 200 else None
        return PromptResult(
            target_id=tid, provider="pollinations-text", model=model,
            kind="text", label=label,
            ok=r.status_code == 200, status=r.status_code,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=str(r.url), body=_trim_text(body),
            content_type=r.headers.get("content-type"),
        )
    except httpx.HTTPError as e:
        return PromptResult(
            target_id=tid, provider="pollinations-text", model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=url, body=f"{type(e).__name__}: {e}",
        )


async def _run_pollinations_image(
    client: httpx.AsyncClient, prompt: str, model: str,
) -> PromptResult:
    started = time.monotonic()
    base = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
    params = {"model": model, "width": "512", "height": "512", "nologo": "true"}
    headers = {
        "User-Agent": random.choice(BROWSER_UAS),
        "Accept": "image/*",
    }
    tid = target_id("pollinations-image", model)
    label = _display_label("pollinations-image", model)
    # We issue the GET so the image is actually generated and Cato sees
    # the flow. We do NOT read the body (images can be ~200KB) — we just
    # return the final URL. Use HEAD-like behavior: stream and close.
    try:
        async with client.stream(
            "GET", base, headers=headers, params=params,
        ) as r:
            # Consume a small chunk to ensure TLS + first body bytes
            # actually flow (for Cato app-ID). Then abort.
            async for _chunk in r.aiter_bytes(1024):
                break
            final_url = str(r.url)
            status = r.status_code
            ctype = r.headers.get("content-type")
        return PromptResult(
            target_id=tid, provider="pollinations-image", model=model,
            kind="image", label=label,
            ok=status == 200, status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=final_url, body=None,
            content_type=ctype,
        )
    except httpx.HTTPError as e:
        return PromptResult(
            target_id=tid, provider="pollinations-image", model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=base, body=f"{type(e).__name__}: {e}",
        )


async def _run_duckduckgo(
    client: httpx.AsyncClient, prompt: str, model: str,
) -> PromptResult:
    started = time.monotonic()
    STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
    CHAT_URL = "https://duckduckgo.com/duckchat/v1/chat"
    ua = random.choice(BROWSER_UAS)
    tid = target_id("duckduckgo", model)
    label = _display_label("duckduckgo", model)

    def err(msg: str) -> PromptResult:
        return PromptResult(
            target_id=tid, provider="duckduckgo", model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=STATUS_URL, body=msg,
        )

    try:
        s = await client.get(
            STATUS_URL,
            headers={
                "User-Agent": ua, "Accept": "*/*",
                "x-vqd-accept": "1", "Cache-Control": "no-store",
            },
        )
    except httpx.HTTPError as e:
        return err(f"status handshake failed: {type(e).__name__}: {e}")

    vqd = s.headers.get("x-vqd-4") or s.headers.get("x-vqd-hash-1")
    if not vqd:
        return PromptResult(
            target_id=tid, provider="duckduckgo", model=model,
            kind="error", label=label,
            ok=False, status=s.status_code,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=STATUS_URL,
            body=("no vqd token in status response — DDG likely rotated "
                  "their anti-abuse header scheme"),
        )

    try:
        # 256 KiB caps the DDG SSE body — chat completions are
        # typically well under 16 KiB; anything larger is either a
        # runaway stream or a misbehaving upstream.
        status, hdr, raw_text, _ = await stream_read_capped(
            client, "POST", CHAT_URL,
            headers={
                "User-Agent": ua,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "x-vqd-4": vqd,
                "Origin": "https://duckduckgo.com",
                "Referer": "https://duckduckgo.com/",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    except httpx.HTTPError as e:
        return err(f"chat post failed: {type(e).__name__}: {e}")

    body = _parse_ddg_stream(raw_text) if status == 200 else None
    return PromptResult(
        target_id=tid, provider="duckduckgo", model=model,
        kind="text", label=label,
        ok=status == 200, status=status,
        latency_ms=int((time.monotonic() - started) * 1000),
        url=CHAT_URL, body=_trim_text(body),
        content_type=hdr.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_TEXT_BYTES = 8_000


def _trim_text(s: str | None) -> str | None:
    if s is None:
        return None
    if len(s) <= _MAX_TEXT_BYTES:
        return s
    return s[:_MAX_TEXT_BYTES] + "\n\n[... truncated]"


def _parse_ddg_stream(raw: str) -> str:
    """DDG returns a text/event-stream where each `data:` line is a JSON
    object like ``{"message": "Hello"}`` or the sentinel ``[DONE]``. We
    concatenate all ``message`` fields in order to get the full reply.
    """
    import json as _json
    chunks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]" or not payload:
            continue
        try:
            obj = _json.loads(payload)
        except _json.JSONDecodeError:
            continue
        msg = obj.get("message")
        if isinstance(msg, str):
            chunks.append(msg)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Keyed-provider runners
# ---------------------------------------------------------------------------
#
# All four of these follow the same pattern as the keyless runners: run
# the request, wall-clock-time it, return a PromptResult. The main
# difference is that they accept an ``api_key`` argument and set the
# appropriate auth header / query param. A missing key short-circuits
# to an error result so the UI can show a clear "add your key" message
# instead of making a doomed upstream request.


def _keyed_missing_key(
    entry: dict[str, Any], model: str,
) -> PromptResult:
    label = f"{entry['label']} · {model}"
    return PromptResult(
        target_id=target_id(entry["provider"], model),
        provider=entry["provider"], model=model,
        kind="error", label=label,
        ok=False, status=None, latency_ms=0,
        url=None,
        body=(f"no API key configured for {entry['label']}. "
              f"Click 'add key' in the Keys panel to paste one."),
    )


async def _run_keyed_openai_compatible(
    client: httpx.AsyncClient,
    prompt: str,
    entry: dict[str, Any],
    model: str,
    api_key: str,
) -> PromptResult:
    """Covers Groq, Mistral, OpenRouter, Together, Cerebras, SambaNova,
    Hyperbolic, DeepSeek, xAI, AI21, Fireworks — anything speaking the
    OpenAI /v1/chat/completions shape with Authorization: Bearer."""
    started = time.monotonic()
    base_url = entry["extra"]["base_url"]
    url = f"{base_url}/chat/completions"
    label = f"{entry['label']} · {model}"
    tid = target_id(entry["provider"], model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    # OpenRouter asks for an identifying header for free-tier rankings.
    # Harmless on everywhere else.
    if entry["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/dzcassell/hAIrspray"
        headers["X-Title"]      = "hAIrspray"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    try:
        r = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=url, body=f"{type(e).__name__}: {e}",
        )

    latency = int((time.monotonic() - started) * 1000)
    if r.status_code != 200:
        # Try to extract a useful error message from the JSON body.
        msg = _extract_error_msg(r.text) or f"HTTP {r.status_code}"
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url, body=msg,
        )

    try:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url,
            body=f"could not parse response: {type(e).__name__}: {e}",
        )

    return PromptResult(
        target_id=tid, provider=entry["provider"], model=model,
        kind="text", label=label,
        ok=True, status=r.status_code, latency_ms=latency,
        url=url, body=_trim_text(text),
        content_type=r.headers.get("content-type"),
    )


async def _run_keyed_gemini(
    client: httpx.AsyncClient,
    prompt: str,
    entry: dict[str, Any],
    model: str,
    api_key: str,
) -> PromptResult:
    """Google Gemini uses ?key= query param + a generateContent body."""
    started = time.monotonic()
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model}:generateContent")
    label = f"{entry['label']} · {model}"
    tid = target_id(entry["provider"], model)
    headers = {"Content-Type": "application/json"}
    body = {
        "contents":          [{"parts": [{"text": prompt}]}],
        "generationConfig":  {"maxOutputTokens": 512},
    }
    try:
        r = await client.post(
            url, headers=headers, params={"key": api_key}, json=body,
        )
    except httpx.HTTPError as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=url, body=f"{type(e).__name__}: {e}",
        )

    latency = int((time.monotonic() - started) * 1000)
    if r.status_code != 200:
        msg = _extract_error_msg(r.text) or f"HTTP {r.status_code}"
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url, body=msg,
        )

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url,
            body=f"could not parse response: {type(e).__name__}: {e}",
        )

    return PromptResult(
        target_id=tid, provider=entry["provider"], model=model,
        kind="text", label=label,
        ok=True, status=r.status_code, latency_ms=latency,
        url=url, body=_trim_text(text),
        content_type=r.headers.get("content-type"),
    )


async def _run_keyed_cohere(
    client: httpx.AsyncClient,
    prompt: str,
    entry: dict[str, Any],
    model: str,
    api_key: str,
) -> PromptResult:
    """Cohere /v2/chat uses Bearer auth but a different body shape."""
    started = time.monotonic()
    url = "https://api.cohere.com/v2/chat"
    label = f"{entry['label']} · {model}"
    tid = target_id(entry["provider"], model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    body = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    try:
        r = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=None,
            latency_ms=int((time.monotonic() - started) * 1000),
            url=url, body=f"{type(e).__name__}: {e}",
        )

    latency = int((time.monotonic() - started) * 1000)
    if r.status_code != 200:
        msg = _extract_error_msg(r.text) or f"HTTP {r.status_code}"
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url, body=msg,
        )

    try:
        data = r.json()
        # Cohere v2 response: { message: { content: [ { text: "..." } ] } }
        text = data["message"]["content"][0]["text"]
    except (KeyError, IndexError, ValueError) as e:
        return PromptResult(
            target_id=tid, provider=entry["provider"], model=model,
            kind="error", label=label,
            ok=False, status=r.status_code, latency_ms=latency,
            url=url,
            body=f"could not parse response: {type(e).__name__}: {e}",
        )

    return PromptResult(
        target_id=tid, provider=entry["provider"], model=model,
        kind="text", label=label,
        ok=True, status=r.status_code, latency_ms=latency,
        url=url, body=_trim_text(text),
        content_type=r.headers.get("content-type"),
    )


async def _run_keyed_hf_router(
    client: httpx.AsyncClient,
    prompt: str,
    entry: dict[str, Any],
    model: str,
    api_key: str,
) -> PromptResult:
    """Hugging Face router speaks OpenAI chat-completions over
    router.huggingface.co/v1. Near-identical to openai-compatible but
    kept separate in case HF adds headers or routing bits."""
    # The shape is identical enough that we can delegate.
    return await _run_keyed_openai_compatible(
        client, prompt, entry, model, api_key,
    )


def _extract_error_msg(raw_text: str) -> str | None:
    """Pull a helpful message out of a provider error JSON body.

    Providers disagree on the error shape, so we try several common
    paths before giving up. Returns None if nothing useful is found.
    """
    if not raw_text:
        return None
    try:
        import json as _json
        data = _json.loads(raw_text)
    except (ValueError, TypeError):
        # Not JSON — truncate and return the raw response.
        return raw_text[:300] if raw_text else None

    # Common locations for error messages across providers.
    for path in (
        ("error", "message"),
        ("error",),
        ("message",),
        ("detail",),
        ("errors", 0, "message"),
    ):
        cur: Any = data
        try:
            for key in path:
                cur = cur[key]
            if isinstance(cur, str):
                return cur[:300]
        except (KeyError, IndexError, TypeError):
            continue

    # Last resort: stringify the whole thing.
    return _json.dumps(data)[:300]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_target_id(tid: str) -> dict[str, str] | None:
    """Return the target entry matching this id, or None.

    Covers both keyless ``PROMPT_TARGETS`` (strict match) and keyed
    provider×model combinations. For keyed targets, only the provider
    half has to match a known ``KEYED_PROVIDERS`` entry — the model
    half is accepted as-is, because dynamic discovery means the
    catalog the UI renders at any given moment is a superset of the
    hard-coded defaults, and we mustn't reject a model that /models
    legitimately returned.
    """
    for t in PROMPT_TARGETS:
        if target_id(t["provider"], t["model"]) == tid:
            return t
    if "::" in tid:
        provider, _, model = tid.partition("::")
        entry = _keyed_entry(provider)
        if entry is not None and model:
            return {
                "provider": provider,
                "model":    model,
                "kind":     entry["kind"],
                "keyed":    True,
            }
    return None


async def run_prompt_target(
    client: httpx.AsyncClient,
    prompt: str,
    provider: str,
    model: str,
    api_key: str | None = None,
) -> PromptResult:
    """Dispatch to the right provider runner.

    ``api_key`` is required for keyed providers; if missing the runner
    returns a clear error PromptResult instead of making the request.
    """
    # Keyless first — fast path.
    if provider == "pollinations-text":
        return await _run_pollinations_text(client, prompt, model)
    if provider == "pollinations-image":
        return await _run_pollinations_image(client, prompt, model)
    if provider == "duckduckgo":
        return await _run_duckduckgo(client, prompt, model)

    # Keyed providers.
    entry = _keyed_entry(provider)
    if entry is None:
        return PromptResult(
            target_id=target_id(provider, model),
            provider=provider, model=model,
            kind="error",
            label=_display_label(provider, model),
            ok=False, status=None, latency_ms=0,
            url=None, body=f"unknown provider: {provider}",
        )

    if not api_key:
        return _keyed_missing_key(entry, model)

    shape = entry["shape"]
    if shape == "openai-compatible":
        return await _run_keyed_openai_compatible(
            client, prompt, entry, model, api_key,
        )
    if shape == "gemini":
        return await _run_keyed_gemini(client, prompt, entry, model, api_key)
    if shape == "cohere":
        return await _run_keyed_cohere(client, prompt, entry, model, api_key)
    if shape == "hf-router":
        return await _run_keyed_hf_router(
            client, prompt, entry, model, api_key,
        )

    return PromptResult(
        target_id=target_id(provider, model),
        provider=provider, model=model,
        kind="error",
        label=f"{entry['label']} · {model}",
        ok=False, status=None, latency_ms=0,
        url=None, body=f"unknown request shape: {shape}",
    )
