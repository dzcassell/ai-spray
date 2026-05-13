"""Request providers.

Each provider represents one way of generating outbound traffic:

* ``WebProbe``        — browser-style GET to a public SaaS URL. Cato
  identifies these via SNI + Host, so even a simple GET is enough.
* ``ApiProbe``        — realistically shaped POST to a vendor API endpoint
  without credentials. The response is almost always 401/403, but the URL
  path, SDK-style User-Agent, and JSON body are the exact shape Cato's app
  signatures expect to see.
* ``PollinationsText`` — keyless real LLM completions via pollinations.ai.
* ``PollinationsImage``— keyless image generation via pollinations.ai.
* ``DuckDuckGoChat``  — keyless proxy to GPT-4o-mini / Claude Haiku / Llama
  / Mistral via duckduckgo.com/duckchat. Best effort; DDG rotates API
  shape occasionally, so failures are logged and swallowed.
"""
from __future__ import annotations

import abc
import asyncio
import random
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import structlog

from . import prompts

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# DDG VQD cache
# ---------------------------------------------------------------------------
# DuckDuckGo's chat backend rotates a CSRF-like "vqd" token via a
# preflight GET to /duckchat/v1/status. The token is valid for some
# minutes; firing one every chat post doubles the HTTP RTTs. Cache
# per-UA so concurrent fires with different browser UAs each keep
# their own valid token, but successive fires from the same fake
# browser reuse it.

import time as _time
import threading as _threading

_VQD_TTL_SEC = 240.0  # DDG observed validity window is ~5 min; refresh early
_vqd_cache: dict[str, tuple[str, float]] = {}
_vqd_lock = _threading.Lock()


def cached_vqd(user_agent: str) -> str | None:
    """Return a still-fresh VQD for ``user_agent`` or None if missing/stale."""
    with _vqd_lock:
        entry = _vqd_cache.get(user_agent)
        if not entry:
            return None
        token, fetched_at = entry
        if _time.monotonic() - fetched_at > _VQD_TTL_SEC:
            _vqd_cache.pop(user_agent, None)
            return None
        return token


def store_vqd(user_agent: str, token: str) -> None:
    with _vqd_lock:
        _vqd_cache[user_agent] = (token, _time.monotonic())


# ---------------------------------------------------------------------------
# Bounded response read
# ---------------------------------------------------------------------------

# Cap on bytes we'll read from a streamed response. Pollinations and
# DuckDuckGo can hold a long-poll open until the http_timeout fires;
# without a byte cap, a misbehaving upstream pins memory equal to
# whatever it can pump in that window. 256 KiB is generous for the
# DDG SSE replies (chat completions rarely exceed ~10 KB) while still
# refusing pathological responses.
RESPONSE_READ_CAP_BYTES = 256 * 1024


async def stream_read_capped(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_bytes: int = RESPONSE_READ_CAP_BYTES,
    **kwargs: Any,
) -> tuple[int, httpx.Headers, str, str]:
    """Issue an HTTP request and read at most ``max_bytes`` of the body.

    Returns ``(status_code, headers, decoded_text, final_url)``. The
    underlying TCP connection is closed as soon as the cap is hit,
    avoiding the unbounded-memory exposure of httpx's default behaviour
    of buffering the entire response before returning.
    """
    async with client.stream(method, url, **kwargs) as r:
        chunks: list[bytes] = []
        total = 0
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += len(chunks[-1])
            if total >= max_bytes:
                break
        raw = b"".join(chunks)
        text = raw.decode("utf-8", errors="replace")
        return r.status_code, r.headers, text, str(r.url)


# ---------------------------------------------------------------------------
# Realistic user-agent pools
# ---------------------------------------------------------------------------

BROWSER_UAS: list[str] = [
    # Recent-ish Chrome on Windows
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    # Recent-ish Chrome on macOS
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    # Firefox on Linux
    ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"),
    # Safari on macOS
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"),
    # Edge on Windows
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"),
]

BROWSER_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8"
)
BROWSER_ACCEPT_LANG = "en-US,en;q=0.9"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    name: str
    category: str
    method: str
    url: str
    status_code: int | None
    ok: bool
    error: str | None = None
    response_snippet: str | None = None


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class Provider(abc.ABC):
    name: str
    category: str

    @abc.abstractmethod
    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        ...


# ---------------------------------------------------------------------------
# Web probe — simple browser-style GET
# ---------------------------------------------------------------------------

class WebProbe(Provider):
    def __init__(self, name: str, url: str, category: str = "chatbot_ui"):
        self.name = name
        self.url = url
        self.category = category

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        headers = {
            "User-Agent": random.choice(BROWSER_UAS),
            "Accept": BROWSER_ACCEPT,
            "Accept-Language": BROWSER_ACCEPT_LANG,
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }
        try:
            r = await client.get(self.url, headers=headers, follow_redirects=True)
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="GET",
                url=str(r.url),
                status_code=r.status_code,
                ok=r.status_code < 500,
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="GET",
                url=self.url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# API probe — realistic unauthenticated POST (or GET) to a vendor endpoint
# ---------------------------------------------------------------------------

@dataclass
class ApiProbe(Provider):
    name: str
    url: str
    user_agent: str
    category: str = "llm_api"
    method: str = "POST"
    body_builder: Callable[[str], dict[str, Any]] | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    # We still send a fake Authorization so the request looks 'real' to
    # app-ID engines that key off the header's presence.
    send_fake_auth: bool = True

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.send_fake_auth and "Authorization" not in headers:
            headers["Authorization"] = "Bearer sk-lab-sim-no-real-credential"

        body = None
        if self.body_builder is not None:
            prompt = random.choice(prompts.TEXT_PROMPTS)
            body = self.body_builder(prompt)

        try:
            if self.method == "POST":
                r = await client.post(self.url, headers=headers, json=body)
            else:
                r = await client.request(self.method, self.url, headers=headers)
            # For unauth probes, anything that isn't a transport error counts
            # as success — the whole point is that Cato saw the flow.
            return ProviderResult(
                name=self.name,
                category=self.category,
                method=self.method,
                url=str(r.url),
                status_code=r.status_code,
                ok=True,
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method=self.method,
                url=self.url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# Pollinations — keyless real responses (text + image)
# ---------------------------------------------------------------------------

class PollinationsText(Provider):
    category = "real_response"
    MODELS = ["openai", "mistral", "llama", "claude", "gemini", "qwen"]

    def __init__(self) -> None:
        self.name = "Pollinations-Text"

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        prompt = random.choice(prompts.TEXT_PROMPTS)
        model = random.choice(self.MODELS)
        url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}"
        params = {"model": model}
        headers = {
            "User-Agent": random.choice(BROWSER_UAS),
            "Accept": "text/plain, */*",
        }
        try:
            status, _hdr, text, final_url = await stream_read_capped(
                client, "GET", url, headers=headers, params=params,
                max_bytes=4096,  # snippet only needs first ~180 chars
            )
            snippet = text[:180] if status == 200 else None
            return ProviderResult(
                name=f"{self.name} ({model})",
                category=self.category,
                method="GET",
                url=final_url,
                status_code=status,
                ok=status == 200,
                response_snippet=snippet,
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="GET",
                url=url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )


class PollinationsImage(Provider):
    category = "real_response"
    MODELS = ["flux", "flux-realism", "turbo"]

    def __init__(self) -> None:
        self.name = "Pollinations-Image"

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        prompt = random.choice(prompts.IMAGE_PROMPTS)
        model = random.choice(self.MODELS)
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
        # Ask for a small image and suppress the logo to keep bandwidth sane.
        params = {"model": model, "width": "512", "height": "512", "nologo": "true"}
        headers = {
            "User-Agent": random.choice(BROWSER_UAS),
            "Accept": "image/*",
        }
        try:
            r = await client.get(url, headers=headers, params=params)
            snippet = None
            if r.status_code == 200:
                snippet = f"image/{r.headers.get('content-type', 'unknown')} {len(r.content)} bytes"
            return ProviderResult(
                name=f"{self.name} ({model})",
                category=self.category,
                method="GET",
                url=str(r.url),
                status_code=r.status_code,
                ok=r.status_code == 200,
                response_snippet=snippet,
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="GET",
                url=url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# DuckDuckGo AI Chat — keyless proxy to GPT-4o-mini / Claude / Llama / Mistral
# ---------------------------------------------------------------------------

class DuckDuckGoChat(Provider):
    category = "real_response"
    MODELS = [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mistralai/Mistral-Small-24B-Instruct-2501",
    ]
    STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
    CHAT_URL = "https://duckduckgo.com/duckchat/v1/chat"

    def __init__(self) -> None:
        self.name = "DuckDuckGo-AIChat"

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        model = random.choice(self.MODELS)
        ua = random.choice(BROWSER_UAS)

        # Step 1: VQD challenge token. Per-UA cache avoids the
        # preflight GET on every fire (token is good for several
        # minutes); on cache miss, fetch a fresh one.
        vqd = cached_vqd(ua)
        if not vqd:
            status_headers = {
                "User-Agent": ua,
                "Accept": "*/*",
                "x-vqd-accept": "1",
                "Cache-Control": "no-store",
            }
            try:
                s = await client.get(self.STATUS_URL, headers=status_headers)
            except httpx.HTTPError as e:
                return ProviderResult(
                    name=self.name,
                    category=self.category,
                    method="GET",
                    url=self.STATUS_URL,
                    status_code=None,
                    ok=False,
                    error=f"status handshake failed: {type(e).__name__}: {e}",
                )

            vqd = s.headers.get("x-vqd-4") or s.headers.get("x-vqd-hash-1")
            if not vqd:
                return ProviderResult(
                    name=self.name,
                    category=self.category,
                    method="GET",
                    url=self.STATUS_URL,
                    status_code=s.status_code,
                    ok=False,
                    error="no vqd token in status response",
                )
            store_vqd(ua, vqd)

        # Step 2: chat POST.
        chat_headers = {
            "User-Agent": ua,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-vqd-4": vqd,
            "Origin": "https://duckduckgo.com",
            "Referer": "https://duckduckgo.com/",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": random.choice(prompts.TEXT_PROMPTS)}
            ],
        }
        try:
            status, _hdr, text, _url = await stream_read_capped(
                client, "POST", self.CHAT_URL,
                headers=chat_headers, json=body,
                max_bytes=4096,  # snippet only needs first ~180 chars
            )
            snippet = text[:180] if status == 200 else None
            return ProviderResult(
                name=f"{self.name} ({model})",
                category=self.category,
                method="POST",
                url=self.CHAT_URL,
                status_code=status,
                ok=status == 200,
                response_snippet=snippet,
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="POST",
                url=self.CHAT_URL,
                status_code=None,
                ok=False,
                error=f"chat post failed: {type(e).__name__}: {e}",
            )

# ---------------------------------------------------------------------------
# MCPAuthedProbe — slice B-static
# ---------------------------------------------------------------------------
#
# Fires an authenticated MCP initialize POST against a real public MCP
# server. The API key is fetched from the KeyStore at execute() time
# (not constructor time) so we don't have to rebuild the registry
# every time a user saves/refreshes a key — the same probe instance
# transparently goes from "no key, returns error" to "key saved,
# fires real authed traffic" without any state mutation on the probe
# itself.
#
# Bodies and headers come from the mcp module so this class stays
# thin: it's just the bridge between the registry and the key store.

class MCPAuthedProbe(Provider):
    """Authed MCP server probe — see MCP_KEYED_SERVERS in app/mcp.py."""

    def __init__(self, server: dict[str, Any], key_provider: Callable[[str], Any]):
        # `server` is one MCP_KEYED_SERVERS entry.
        # `key_provider` is an async callable that returns the stored
        # API key for a provider slug, or None if not saved yet.
        # In production it's KeyStore.get; in tests it's whatever the
        # test wires up.
        self._server = server
        self._key_provider = key_provider
        self.name = f"{server['label']} (authed)"
        self.url = server["url"]
        self.category = "mcp_authed"

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        # Late import — providers.py is imported very early (before
        # the app package is fully loaded) and importing app.mcp at
        # module load time would create a cycle on some startup paths.
        from . import mcp as _mcp

        api_key = await self._key_provider(self._server["provider"])
        if not api_key:
            # No key saved → return a clear, non-network error so
            # the operator sees exactly what's missing in the UI.
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="POST",
                url=self.url,
                status_code=None,
                ok=False,
                error=(f"no API key saved for "
                       f"{self._server.get('provider', '?')}"
                       + (f" (see {self._server['signup_url']})"
                          if self._server.get("signup_url") else "")),
            )

        headers = _mcp.headers_for_keyed(self._server, api_key)
        body = _mcp.build_authed_probe_body()

        try:
            r = await client.post(self.url, headers=headers, json=body)
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="POST",
                url=str(r.url),
                status_code=r.status_code,
                ok=True,  # any non-transport response is "fabric saw it"
            )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="POST",
                url=self.url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# MCPSessionProbe — end-to-end synthetic session (slice A2)
# ---------------------------------------------------------------------------
#
# One execute() runs a sequence of POSTs against a single MCP destination
# on a single connection: initialize → notifications/initialized →
# tools/list → tools/call → resources/read → DELETE. The point is to
# produce a *connected* MCP flow on the wire, distinct from the
# one-shot synthetic probes whose individual messages can look like
# malformed JSON-RPC noise to a stateful classifier.
#
# The server will likely 401/403 after initialize for unauth targets;
# we keep firing the rest of the sequence anyway because the SASE
# fabric sees the message *attempts*, which is what the classifier
# inspects. Per-message pacing is intentionally human-ish (200-800ms)
# so the connection-reuse pattern is itself a fingerprint.

class MCPSessionProbe(Provider):
    """Run a multi-message MCP session against one destination.

    The full sequence per execute():
        POST initialize          → captures Mcp-Session-Id if returned
        POST notifications/initialized
        POST tools/list          → notes how many tools were returned
        POST tools/call          (random tool-call profile)
        POST resources/read
        DELETE <url>             (with the session id, to terminate)
    """

    category = "mcp_session_sim"

    # Inter-message pacing — varied to look like a real assistant
    # rather than a tight machine loop.
    PACING_MIN_SEC = 0.2
    PACING_MAX_SEC = 0.8

    def __init__(
        self,
        name: str,
        url: str,
        *,
        user_agent: str = "mcp-python-sdk/1.2.0",
        auth_header: str | None = None,
        auth_value: str | None = None,
    ) -> None:
        self.name = name
        self.url = url
        self._ua = user_agent
        self._auth_header = auth_header
        self._auth_value = auth_value

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        from . import mcp as _mcp  # late import to avoid load-order cycle

        # Pin one protocol version + one session id for the whole
        # session so every POST is mutually consistent — that's how
        # real clients behave once initialize returns.
        protocol_version = _mcp.pick_protocol_version()
        session_id = _mcp.synthetic_session_id()

        def hdrs(extra: dict[str, str] | None = None) -> dict[str, str]:
            h = _mcp.headers_for(
                "streamable",
                protocol_version=protocol_version,
                session_id=session_id,
            )
            h["User-Agent"] = self._ua
            if self._auth_header and self._auth_value:
                h[self._auth_header] = self._auth_value
            if extra:
                h.update(extra)
            return h

        steps = [
            ("initialize",                _mcp.build_initialize_request(protocol_version=protocol_version)),
            ("notifications/initialized", _mcp.build_initialized_notification()),
            ("tools/list",                _mcp.build_tools_list_request()),
            ("tools/call",                _mcp.random_tool_call_body()),
            ("resources/read",            _mcp.build_resources_read_request("file:///etc/hosts")),
        ]

        last_status: int | None = None
        ok_steps = 0
        try:
            for i, (label, body) in enumerate(steps):
                r = await client.post(self.url, headers=hdrs(), json=body)
                last_status = r.status_code
                if r.status_code < 500:
                    ok_steps += 1
                # If the server returned a session id on initialize,
                # adopt it for the rest of the conversation. This is
                # what real clients do — the server's session id wins
                # over whatever the client guessed.
                if label == "initialize":
                    server_sid = r.headers.get("Mcp-Session-Id")
                    if server_sid:
                        session_id = server_sid
                if i < len(steps) - 1:
                    await asyncio.sleep(
                        random.uniform(self.PACING_MIN_SEC, self.PACING_MAX_SEC)
                    )

            # Spec says clients SHOULD DELETE the session URL to
            # terminate the session cleanly. Many servers accept it,
            # many ignore it — either way it's part of the wire shape.
            try:
                await client.request("DELETE", self.url, headers=hdrs())
            except httpx.HTTPError:
                pass
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="POST",
                url=self.url,
                status_code=last_status,
                ok=False,
                error=(f"{type(e).__name__}: {e}; completed {ok_steps}/"
                       f"{len(steps)} steps before failure"),
            )

        return ProviderResult(
            name=self.name,
            category=self.category,
            method="POST",
            url=self.url,
            status_code=last_status,
            ok=ok_steps >= 1,
            response_snippet=(f"session-sim: {ok_steps}/{len(steps)} steps, "
                              f"version={protocol_version}, session={session_id[:8]}…"),
        )


# ---------------------------------------------------------------------------
# SSEStreamProbe — legacy HTTP+SSE long-poll
# ---------------------------------------------------------------------------
#
# The legacy MCP transport (HTTP+SSE) splits across two requests:
#   POST /messages  ← JSON-RPC requests (already covered by ApiProbe)
#   GET  /sse       ← long-poll event-stream the server pushes to
#
# Every synthetic MCP probe to date has been on the POST side. Add a
# probe that does the GET half: open the SSE channel, hold it for
# a few seconds, read whatever events the server pushes, close. SASE
# classifiers that key on "long-poll over HTTPS with Accept:
# text/event-stream" will see the right shape.

class SSEStreamProbe(Provider):
    """Long-poll GET against an MCP server's SSE endpoint.

    Reads up to ``max_bytes`` or ``max_seconds``, whichever comes
    first. The connection is allowed to idle so the fabric sees a
    sustained long-poll, not a quick poll-and-drop.
    """

    category = "mcp_synthetic"

    def __init__(
        self,
        name: str,
        url: str,
        *,
        user_agent: str = "@modelcontextprotocol/sdk/1.0.4",
        max_bytes: int = 64 * 1024,
        max_seconds: float = 8.0,
    ) -> None:
        self.name = name
        self.url = url
        self._ua = user_agent
        self._max_bytes = max_bytes
        self._max_seconds = max_seconds

    async def execute(self, client: httpx.AsyncClient) -> ProviderResult:
        headers = {
            "User-Agent": self._ua,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        try:
            async with client.stream("GET", self.url, headers=headers) as r:
                bytes_read = 0
                started = asyncio.get_event_loop().time()
                try:
                    async for chunk in r.aiter_bytes():
                        bytes_read += len(chunk)
                        elapsed = asyncio.get_event_loop().time() - started
                        if bytes_read >= self._max_bytes:
                            break
                        if elapsed >= self._max_seconds:
                            break
                except (asyncio.TimeoutError, httpx.ReadTimeout):
                    pass
                return ProviderResult(
                    name=self.name,
                    category=self.category,
                    method="GET",
                    url=self.url,
                    status_code=r.status_code,
                    ok=r.status_code < 500,
                    response_snippet=f"sse long-poll: {bytes_read} bytes read",
                )
        except httpx.HTTPError as e:
            return ProviderResult(
                name=self.name,
                category=self.category,
                method="GET",
                url=self.url,
                status_code=None,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
