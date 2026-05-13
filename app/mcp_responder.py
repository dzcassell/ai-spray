"""Inbound MCP responder — synthetic MCP *server* hosted by hAIrspray.

The synthetic probes elsewhere in this codebase generate *outbound*
MCP-shaped traffic so SASE engines can fingerprint it. This module
flips the direction: hAIrspray *answers* MCP requests from real
clients pointed at it, so two further test cases become possible:

1. **Inbound classification.** Stand the responder up inside your
   estate, point a real Claude Code / Cursor / Goose client at
   ``https://hairspray.lab/mcp-responder/streamable``, and observe
   whether the SASE fabric classifies the inbound flow as MCP.
   Useful for shadow-MCP detection — many enterprises now treat
   anything that *looks* like MCP as something to log.

2. **DLP egress against a controllable peer.** A real MCP client
   wired up to the responder will send tool-call inputs through
   the fabric. Drop synthetic PII into the prompts you feed the
   client; the responder echoes whatever it received back into the
   hAIrspray event stream so you can compare what the fabric's DLP
   engine reported against what actually went on the wire.

Two transports are mounted, matching the two MCP wire conventions:

* ``POST /mcp-responder/streamable``  — current Streamable HTTP
  semantics; minted Mcp-Session-Id is returned as a response
  header on the initialize result.

* ``POST /mcp-responder/sse/messages`` plus ``GET /mcp-responder/sse``
  — legacy HTTP+SSE; the GET holds open an SSE channel that the
  responder pushes periodic notifications down. Real MCP servers
  push tool-result notifications this way; we push synthetic
  ping/progress notifications so the long-poll has measurable
  content for the fabric to inspect.

**No authentication.** This is a lab tool, not production. Bind the
container to a network segment where every client is trusted, or
front it with your own auth proxy. The README repeats this loudly.

The responder is opt-in via the ``MCP_RESPONDER_ENABLED`` env var
(default off) so installs that only want outbound traffic don't
accidentally start answering inbound MCP requests.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .state import AppState

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Canned content the responder returns
# ---------------------------------------------------------------------------
# Realistic-shaped but harmless. The tools advertised are the ones a
# SASE/DLP classifier most expects to see from a real MCP server, so
# a fabric that fingerprints on the tools/list response gets a
# matching signal.

CANNED_TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Return the input message unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the lab sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Retrieve the body of a URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "submit_form",
        "description": ("Submit a form. Common DLP test surface — pipe "
                        "synthetic PII through the field_value argument."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "field_label": {"type": "string"},
                "field_value": {"type": "string"},
            },
            "required": ["field_label", "field_value"],
        },
    },
]


CANNED_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "summarize",
        "description": "Summarize the provided text in two sentences.",
        "arguments": [
            {"name": "text", "description": "What to summarize.", "required": True},
        ],
    },
]


CANNED_RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "lab://sample.txt",
        "name": "sample.txt",
        "description": "A canned text resource served by the lab responder.",
        "mimeType": "text/plain",
    },
]


# Pin a single protocol version for our responses. We echo back
# whatever the client sent in initialize if it's one we recognize.
SUPPORTED_PROTOCOL_VERSIONS = frozenset({
    "2024-11-05", "2025-03-26", "2025-06-18",
})
DEFAULT_RESPONSE_PROTOCOL = "2025-06-18"


# ---------------------------------------------------------------------------
# In-memory session table
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    session_id: str
    protocol_version: str
    client_info: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    transport: str = "streamable"  # or "sse-legacy"


# Sessions live in memory for the lifetime of the responder; expired
# entries are pruned lazily on each incoming POST. No persistence —
# this is a stateless test peer.
_SESSIONS: dict[str, _Session] = {}
_SESSION_TTL_SEC = 600.0  # 10 minutes of idleness → expired


def _prune_expired() -> None:
    now = time.time()
    expired = [
        sid for sid, sess in _SESSIONS.items()
        if now - sess.last_seen_at > _SESSION_TTL_SEC
    ]
    for sid in expired:
        _SESSIONS.pop(sid, None)


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

def _success(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _handle_one(
    msg: dict[str, Any],
    session: _Session | None,
) -> tuple[dict[str, Any] | None, _Session | None]:
    """Process one JSON-RPC message.

    Returns ``(response, updated_session)``. ``response`` is None when
    the input was a notification (no id) — JSON-RPC says notifications
    have no reply. ``updated_session`` reflects any side effects (most
    importantly: minted on initialize).
    """
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    is_notification = "id" not in msg

    if method == "initialize":
        # Echo the protocol version if we support it; otherwise return
        # our default. Either way, mint a fresh session id so the
        # client has something to echo back on subsequent calls.
        requested = params.get("protocolVersion")
        proto = (
            requested if requested in SUPPORTED_PROTOCOL_VERSIONS
            else DEFAULT_RESPONSE_PROTOCOL
        )
        client_info = params.get("clientInfo") or {}
        new_sess = _Session(
            session_id=uuid.uuid4().hex,
            protocol_version=proto,
            client_info=client_info,
        )
        _SESSIONS[new_sess.session_id] = new_sess
        result = {
            "protocolVersion": proto,
            "capabilities": {
                "tools":     {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts":   {"listChanged": False},
                "logging":   {},
            },
            "serverInfo": {
                "name":    "hairspray-mcp-responder",
                "version": "1.0.0",
            },
            "instructions": ("Lab responder for SASE/DLP testing. Every "
                             "tool result is canned; nothing is executed."),
        }
        return _success(req_id, result), new_sess

    if method == "ping":
        return (_success(req_id, {}), session) if not is_notification else (None, session)

    if method == "tools/list":
        return _success(req_id, {"tools": CANNED_TOOLS}), session

    if method == "tools/call":
        # Echo the call's input back as the tool result so the operator
        # can see exactly what the client sent — that's the DLP test.
        name = params.get("name", "?")
        args = params.get("arguments") or {}
        text = (f"Lab responder received tool call name={name!r} "
                f"arguments={json.dumps(args, ensure_ascii=False)[:500]}")
        result = {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }
        return _success(req_id, result), session

    if method == "resources/list":
        return _success(req_id, {"resources": CANNED_RESOURCES}), session

    if method == "resources/read":
        uri = params.get("uri", "")
        return _success(req_id, {
            "contents": [
                {
                    "uri":      uri,
                    "mimeType": "text/plain",
                    "text":     f"Lab responder canned content for {uri!r}.",
                },
            ],
        }), session

    if method == "prompts/list":
        return _success(req_id, {"prompts": CANNED_PROMPTS}), session

    if method == "prompts/get":
        name = params.get("name", "?")
        return _success(req_id, {
            "description": f"Canned lab prompt {name!r}.",
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text":  "Summarize the input text in two sentences.",
                },
            }],
        }), session

    if method == "logging/setLevel":
        return _success(req_id, {}), session

    # Notifications: no response, no error.
    if is_notification:
        return None, session

    return _error(req_id, -32601, f"Method not found: {method!r}"), session


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

def _publish_inbound(
    state: AppState,
    *,
    transport: str,
    method: str | None,
    session_id: str | None,
    status: int,
    body_preview: str,
) -> None:
    """Publish one inbound responder hit to the event stream."""
    try:
        state.publish_result({
            "target":   f"mcp-responder ({transport})",
            "category": "mcp_responder",
            "method":   "POST" if transport != "sse-stream" else "GET",
            "url":      f"local://mcp-responder/{transport}",
            "status":   status,
            "ok":       200 <= status < 400,
            "source":   "responder",
            "snippet":  (f"{method or '(no method)'}"
                         + (f" sid={session_id[:8]}…" if session_id else "")
                         + (f"  {body_preview}" if body_preview else "")),
        })
    except Exception:  # pragma: no cover - publish must never raise into handler
        log.warning("mcp_responder_publish_failed", method=method)


def _make_streamable_handler(state: AppState):
    """POST handler for the streamable transport."""

    async def streamable(request: Request) -> Response:
        _prune_expired()

        # Reject oversized bodies — even a lab tool shouldn't accept
        # megabyte JSON-RPC payloads. 1 MiB is generous for MCP.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > 1024 * 1024:
                    return JSONResponse(
                        {"error": "request too large"}, status_code=413,
                    )
            except ValueError:
                pass

        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            _publish_inbound(state, transport="streamable", method=None,
                             session_id=None, status=400,
                             body_preview="invalid JSON")
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        client_sid = request.headers.get("Mcp-Session-Id")
        session = _SESSIONS.get(client_sid) if client_sid else None

        # MCP allows JSON-RPC batched arrays in spec rev 2025-03-26.
        # Handle both forms uniformly.
        if isinstance(payload, list):
            messages = payload
            batch = True
        elif isinstance(payload, dict):
            messages = [payload]
            batch = False
        else:
            return JSONResponse(
                {"error": "expected object or array"}, status_code=400,
            )

        responses: list[dict[str, Any]] = []
        minted_session_id: str | None = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            resp, session = _handle_one(msg, session)
            method_name = msg.get("method")
            _publish_inbound(
                state, transport="streamable",
                method=method_name,
                session_id=client_sid or (session.session_id if session else None),
                status=200,
                body_preview=(
                    f" args={json.dumps(msg.get('params') or {}, ensure_ascii=False)[:120]}"
                    if method_name == "tools/call" else ""
                ),
            )
            if session and not minted_session_id and method_name == "initialize":
                minted_session_id = session.session_id
            if resp is not None:
                responses.append(resp)
            if session:
                session.last_seen_at = time.time()

        headers = {}
        if minted_session_id:
            headers["Mcp-Session-Id"] = minted_session_id

        if not responses:
            # All inputs were notifications — spec says return 202.
            return Response(status_code=202, headers=headers)

        return JSONResponse(
            responses if batch else responses[0],
            headers=headers,
        )

    return streamable


def _make_legacy_message_handler(state: AppState):
    """POST handler for the legacy SSE transport's /messages endpoint."""

    async def legacy_post(request: Request) -> Response:
        _prune_expired()

        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "invalid JSON"}, status_code=400,
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": "expected single JSON object on the legacy "
                          "endpoint — batches are streamable-only"},
                status_code=400,
            )

        # Legacy session id is conventionally a query param, not a header.
        sid = request.query_params.get("sessionId")
        session = _SESSIONS.get(sid) if sid else None

        resp, session = _handle_one(payload, session)
        method_name = payload.get("method")
        _publish_inbound(
            state, transport="sse-legacy",
            method=method_name,
            session_id=sid or (session.session_id if session else None),
            status=200,
            body_preview="",
        )
        if session:
            session.last_seen_at = time.time()

        if resp is None:
            return Response(status_code=202)
        return JSONResponse(resp)

    return legacy_post


def _make_legacy_sse_stream(state: AppState):
    """GET handler — long-poll SSE channel for the legacy transport.

    Real MCP servers push tool-call results and notifications down
    this channel. We push synthetic ping/progress notifications at
    1-2s intervals so the long-poll has measurable content the
    fabric can inspect, then idle quietly until the client closes.
    """

    async def legacy_sse(request: Request) -> Response:
        async def gen():
            yield b": hairspray-mcp-responder legacy sse channel\n\n"
            _publish_inbound(
                state, transport="sse-stream", method="GET /sse",
                session_id=None, status=200, body_preview="opened",
            )
            ping_count = 0
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    ping_count += 1
                    notification = {
                        "jsonrpc": "2.0",
                        "method":  "notifications/progress",
                        "params":  {
                            "progressToken": "responder-heartbeat",
                            "progress":  ping_count,
                            "total":     0,
                        },
                    }
                    yield (
                        f"event: message\ndata: {json.dumps(notification)}\n\n"
                    ).encode("utf-8")
                    # Pace the heartbeat. Long enough that the
                    # connection counts as a real long-poll, short
                    # enough that classifiers see periodic content.
                    await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection":        "keep-alive",
            },
        )

    return legacy_sse


def _make_delete_handler(state: AppState):
    """DELETE handler — terminate a session.

    Spec says clients SHOULD send DELETE to clean up when done. We
    drop the session entry; subsequent calls with the same id will
    look stateless.
    """

    async def delete_session(request: Request) -> Response:
        sid = request.headers.get("Mcp-Session-Id") or \
              request.query_params.get("sessionId")
        if sid and sid in _SESSIONS:
            _SESSIONS.pop(sid, None)
            _publish_inbound(
                state, transport="streamable", method="DELETE",
                session_id=sid, status=200, body_preview="session terminated",
            )
            return Response(status_code=204)
        return Response(status_code=404)

    return delete_session


# ---------------------------------------------------------------------------
# Public registration
# ---------------------------------------------------------------------------

def responder_enabled() -> bool:
    return os.environ.get(
        "MCP_RESPONDER_ENABLED", "false",
    ).strip().lower() in ("1", "true", "yes", "on", "y")


def build_routes(state: AppState) -> list[Route]:
    """Return the Route list to mount on the main Starlette app.

    No-op when MCP_RESPONDER_ENABLED is falsy — saves the operator
    from accidentally hosting an open MCP responder.
    """
    if not responder_enabled():
        return []
    return [
        Route(
            "/mcp-responder/streamable",
            _make_streamable_handler(state),
            methods=["POST"],
        ),
        Route(
            "/mcp-responder/streamable",
            _make_delete_handler(state),
            methods=["DELETE"],
        ),
        Route(
            "/mcp-responder/sse",
            _make_legacy_sse_stream(state),
            methods=["GET"],
        ),
        Route(
            "/mcp-responder/sse/messages",
            _make_legacy_message_handler(state),
            methods=["POST"],
        ),
    ]
