"""Model Context Protocol (MCP) traffic generation.

Three slices land in this module:

* **Slice A (synthetic)**: well-formed MCP JSON-RPC payloads sent to
  reflectors and known public MCP server hostnames *without* auth.
  Tests SASE/NGFW classification of MCP-shaped traffic by payload
  shape and by destination, before any real handshake completes.

* **Slice B-static (real, token-auth)**: lands in the next commit.
  A few hosted MCP servers accept simple token auth (GitHub PAT,
  Notion integration token, Linear API key); those will plug into
  the existing key-store pattern.

* **Slice C (DLP)**: the Profile Tests endpoint will gain a
  ``payload_shape="mcp"`` mode that wraps synthetic PII inside an
  MCP ``tools/call`` envelope. Reuses ``build_initialize_request``
  and ``build_tools_call_request`` from this module.

## On the wire

MCP rides three transports:

1. **stdio**         — local pipes; never crosses the network. Out
                       of scope for SASE/NGFW testing entirely.
2. **HTTP+SSE**      — legacy: client POSTs JSON-RPC to ``/messages``,
                       reads server-sent events from a long-poll GET
                       ``/sse``. Officially deprecated in spec rev
                       2024-11 but still widely deployed.
3. **Streamable HTTP** — current spec: single POST endpoint (``/mcp``
                       by convention), server upgrades the response
                       to SSE for streaming results back.

We generate traffic for transports 2 and 3. The wire shape is
identical at the message level (JSON-RPC 2.0); the difference is
the URL and the ``Accept`` header.
"""
from __future__ import annotations

import random
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelope helpers
# ---------------------------------------------------------------------------

# MCP protocol revisions seen in real-world traffic. We rotate across
# the set per probe so SASE classifiers see the variety they'd see in
# a heterogeneous client population:
#
#   * 2024-11-05 — first version that pinned Streamable HTTP semantics;
#                  still widely deployed by older clients.
#   * 2025-03-26 — added JSON-RPC batching and Mcp-Session-Id; legacy
#                  HTTP+SSE was officially deprecated here.
#   * 2025-06-18 — current normative version. Adds elicitation,
#                  resource indicators (RFC 8707) for OAuth, removes
#                  JSON-RPC batching.
#
# DEFAULT_PROTOCOL_VERSION is kept for callers that need a single
# concrete value (e.g. the MCP-Protocol-Version header on the very
# first POST, when no negotiation has happened yet).
MCP_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)
DEFAULT_PROTOCOL_VERSION: str = MCP_PROTOCOL_VERSIONS[-1]
# Back-compat alias — call sites that already imported the singular
# name continue to work. New code should prefer pick_protocol_version()
# or pass a version explicitly.
MCP_PROTOCOL_VERSION = DEFAULT_PROTOCOL_VERSION


def pick_protocol_version() -> str:
    """Randomly choose a protocol version for the next probe."""
    return random.choice(MCP_PROTOCOL_VERSIONS)


def _msg_id() -> str:
    """Fresh JSON-RPC message ID per call."""
    return uuid.uuid4().hex[:12]


def build_initialize_request(
    client_name: str = "hairspray-test-client",
    client_version: str = "1.0.0",
    protocol_version: str | None = None,
) -> dict[str, Any]:
    """The required first message of any MCP session.

    ``protocol_version`` defaults to a random pick from
    MCP_PROTOCOL_VERSIONS so successive probes rotate through the
    versions in real-world deployment.
    """
    if protocol_version is None:
        protocol_version = pick_protocol_version()
    return {
        "jsonrpc": "2.0",
        "id": _msg_id(),
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
            },
            "clientInfo": {
                "name": client_name,
                "version": client_version,
            },
        },
    }


def build_tools_list_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": _msg_id(),
        "method": "tools/list",
        "params": {},
    }


def build_tools_call_request(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """tools/call — the workhorse MCP method.

    For DLP testing (slice C), ``arguments`` is the field where
    synthetic PII gets embedded; SASE engines that parse MCP
    properly should inspect it.
    """
    return {
        "jsonrpc": "2.0",
        "id": _msg_id(),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def build_resources_read_request(uri: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": _msg_id(),
        "method": "resources/read",
        "params": {"uri": uri},
    }


def build_prompts_get_request(name: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": _msg_id(),
        "method": "prompts/get",
        "params": {"name": name, "arguments": {}},
    }


def build_initialized_notification() -> dict[str, Any]:
    """Sent by the client after receiving the initialize response.

    Notifications have no ``id`` field — that's what distinguishes them
    from requests in JSON-RPC 2.0.
    """
    return {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }


# ---------------------------------------------------------------------------
# Headers per transport
# ---------------------------------------------------------------------------

# Streamable HTTP requires the client to advertise it can read both
# JSON and SSE responses — the server picks at runtime.
STREAMABLE_HTTP_ACCEPT = "application/json, text/event-stream"

# Legacy HTTP+SSE: client POST is plain JSON; the SSE channel is the
# separate GET /sse stream.
LEGACY_SSE_ACCEPT = "application/json"


def headers_for(
    transport: str,
    protocol_version: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """Headers the SASE classifier sees on the outbound POST.

    ``transport`` is "streamable" or "sse-legacy". ``protocol_version``
    is echoed back in the MCP-Protocol-Version header (defaults to a
    random pick so successive probes vary). ``session_id``, if given,
    is set as Mcp-Session-Id; real clients echo whatever the server
    returned on the initialize response on every subsequent POST.
    """
    if protocol_version is None:
        protocol_version = pick_protocol_version()
    if transport == "streamable":
        h = {
            "Content-Type": "application/json",
            "Accept": STREAMABLE_HTTP_ACCEPT,
            # MCP-Protocol-Version is mandated post-initialize since
            # spec rev 2025-03-26. Several inspecting fabrics now look
            # for it as a strong MCP signal.
            "MCP-Protocol-Version": protocol_version,
        }
        if session_id:
            h["Mcp-Session-Id"] = session_id
        return h
    # Legacy
    return {
        "Content-Type": "application/json",
        "Accept": LEGACY_SSE_ACCEPT,
    }


# ---------------------------------------------------------------------------
# Tool-call argument profiles (for slice A and slice C realism)
# ---------------------------------------------------------------------------

# Realistic argument shapes for common MCP tool genres. SASE DLP
# engines that parse MCP properly should inspect these argument
# objects; engines that treat the body as opaque JSON won't.

TOOL_CALL_PROFILES: list[dict[str, Any]] = [
    {
        "genre": "filesystem",
        "tool": "read_file",
        "args": lambda: {
            "path": random.choice([
                "/etc/passwd",
                "/home/user/.aws/credentials",
                "/var/log/auth.log",
                "C:\\Users\\Administrator\\AppData\\Roaming\\Microsoft\\Credentials",
                "~/Documents/quarterly-report.xlsx",
            ]),
        },
    },
    {
        "genre": "shell",
        "tool": "shell.execute",
        "args": lambda: {
            "command": random.choice([
                "ls -la /home",
                "cat /etc/hosts",
                "whoami && id",
                "ps -ef | grep ssh",
                "git log --oneline -20",
            ]),
        },
    },
    {
        "genre": "browser",
        "tool": "browser.navigate",
        "args": lambda: {
            "url": random.choice([
                "https://internal.example.com/admin",
                "https://intranet.corp.invalid/files",
                "https://github.com/dzcassell/hAIrspray",
                "https://salesforce.com/lightning/r/Account",
            ]),
        },
    },
    {
        "genre": "database",
        "tool": "db.query",
        "args": lambda: {
            "connection": "postgres://app@db.internal:5432/prod",
            "sql": random.choice([
                "SELECT * FROM users LIMIT 100",
                "SELECT email, ssn FROM customers WHERE id = 1234",
                "UPDATE accounts SET balance = balance - 500 WHERE id = 1",
            ]),
        },
    },
    {
        "genre": "generic-echo",
        "tool": "echo",
        "args": lambda: {
            "message": random.choice([
                "ping",
                "diagnostic test from hAIrspray",
                "MCP traffic generation probe",
            ]),
        },
    },
]


def random_tool_call_body() -> dict[str, Any]:
    """Pick a tool-call profile at random and build the JSON-RPC body."""
    profile = random.choice(TOOL_CALL_PROFILES)
    return build_tools_call_request(
        tool_name=profile["tool"],
        arguments=profile["args"](),
    )


# ---------------------------------------------------------------------------
# Public MCP server registry (slice A — unauthenticated traffic)
# ---------------------------------------------------------------------------

# Real public MCP server hostnames as of April 2026. These are hit
# without auth in slice A — the *initial POST* is what generates
# classifiable traffic, and the server's 401/403 response is fine
# (we want SASE to see the destination and the payload shape, not
# necessarily a successful handshake). Slice B-static, in the next
# commit, adds auth for the three providers that accept simple
# token auth.

PUBLIC_MCP_SERVERS: list[dict[str, Any]] = [
    # GitHub — accepts PAT auth (slice B-static); shown here for
    # the unauth path so SASE sees the destination either way.
    {
        "name": "GitHub MCP",
        "host": "api.githubcopilot.com",
        "streamable_url": "https://api.githubcopilot.com/mcp",
        "sse_url": None,  # GitHub only ships streamable HTTP
    },
    {
        "name": "Cloudflare Workers MCP",
        "host": "bindings.mcp.cloudflare.com",
        "streamable_url": "https://bindings.mcp.cloudflare.com/sse",
        "sse_url": "https://bindings.mcp.cloudflare.com/sse",
    },
    {
        "name": "Cloudflare Docs MCP",
        "host": "docs.mcp.cloudflare.com",
        "streamable_url": "https://docs.mcp.cloudflare.com/sse",
        "sse_url": "https://docs.mcp.cloudflare.com/sse",
    },
    {
        "name": "Notion MCP",
        "host": "mcp.notion.com",
        "streamable_url": "https://mcp.notion.com/mcp",
        "sse_url": "https://mcp.notion.com/sse",
    },
    {
        "name": "Linear MCP",
        "host": "mcp.linear.app",
        "streamable_url": "https://mcp.linear.app/mcp",
        "sse_url": "https://mcp.linear.app/sse",
    },
    {
        "name": "Atlassian MCP",
        "host": "mcp.atlassian.com",
        "streamable_url": "https://mcp.atlassian.com/v1/sse",
        "sse_url": "https://mcp.atlassian.com/v1/sse",
    },
    {
        "name": "Asana MCP",
        "host": "mcp.asana.com",
        "streamable_url": "https://mcp.asana.com/sse",
        "sse_url": "https://mcp.asana.com/sse",
    },
    {
        "name": "Sentry MCP",
        "host": "mcp.sentry.dev",
        "streamable_url": "https://mcp.sentry.dev/mcp",
        "sse_url": "https://mcp.sentry.dev/sse",
    },
    {
        "name": "Stripe MCP",
        "host": "mcp.stripe.com",
        "streamable_url": "https://mcp.stripe.com/v1",
        "sse_url": None,
    },
    {
        "name": "PayPal MCP",
        "host": "mcp.paypal.com",
        "streamable_url": "https://mcp.paypal.com/mcp",
        "sse_url": None,
    },
    {
        "name": "Square MCP",
        "host": "mcp.squareup.com",
        "streamable_url": "https://mcp.squareup.com/sse",
        "sse_url": "https://mcp.squareup.com/sse",
    },
    {
        "name": "Block (Goose) MCP",
        "host": "mcp.block.xyz",
        "streamable_url": "https://mcp.block.xyz/sse",
        "sse_url": "https://mcp.block.xyz/sse",
    },
]


# Reflectors for slice A — payload-shape testing without any real
# MCP destination. The reflector itself doesn't pretend to be MCP;
# the test is whether SASE classifies based on the body alone.
REFLECTOR_TARGETS: list[dict[str, str]] = [
    {
        "name": "Reflector (httpbin)",
        "url": "https://httpbin.org/post",
    },
    {
        "name": "Reflector (postman-echo)",
        "url": "https://postman-echo.com/post",
    },
]


# ---------------------------------------------------------------------------
# Keyed MCP servers (slice B-static)
# ---------------------------------------------------------------------------
#
# Three public MCP servers accept simple token auth — no OAuth, no
# refresh tokens, no callback URL registration. Each entry uses the
# same shape as KEYED_PROVIDERS in prompt.py so the existing key-store
# machinery handles save/refresh/delete unchanged. The Config → Keys
# panel renders them automatically alongside the AI providers.
#
# OAuth-only MCP servers (Atlassian, Asana, Sentry, Cloudflare's
# hosted MCPs, Stripe's MCP) are deliberately out of scope here —
# they require a callback handler and refresh-token rotation that
# would be a substantial separate feature. They appear unauth in
# slice A's mcp_synthetic probes, so SASE still sees the
# destination + payload combination.

MCP_KEYED_SERVERS: list[dict[str, Any]] = [
    {
        "provider":   "github_mcp",
        "label":      "GitHub MCP",
        "signup_url": "https://github.com/settings/tokens/new?scopes=repo,read:user",
        # GitHub MCP uses the same Authorization: Bearer <PAT> as the
        # GitHub Models endpoint. Standard PAT with at least repo and
        # read:user scopes works.
        "url":            "https://api.githubcopilot.com/mcp",
        "transport":      "streamable",
        "auth_header":    "Authorization",
        "auth_format":    "Bearer {key}",
        "user_agent":     "github-mcp-client/1.0.0",
        "scope_hint":     "PAT with repo + read:user scopes",
    },
    {
        "provider":   "notion_mcp",
        "label":      "Notion MCP",
        "signup_url": "https://www.notion.so/profile/integrations",
        # Notion MCP accepts the integration token (secret_xxx) as a
        # Bearer credential.
        "url":            "https://mcp.notion.com/mcp",
        "transport":      "streamable",
        "auth_header":    "Authorization",
        "auth_format":    "Bearer {key}",
        "user_agent":     "notion-mcp-client/1.0.0",
        "scope_hint":     "Internal integration token, any workspace",
    },
    {
        "provider":   "linear_mcp",
        "label":      "Linear MCP",
        "signup_url": "https://linear.app/settings/api",
        # Linear personal API key uses Authorization: <key> directly,
        # NO 'Bearer' prefix. This is a Linear-specific quirk that
        # tripped me up while testing.
        "url":            "https://mcp.linear.app/mcp",
        "transport":      "streamable",
        "auth_header":    "Authorization",
        "auth_format":    "{key}",
        "user_agent":     "linear-mcp-client/1.0.0",
        "scope_hint":     "Personal API key with read+write access",
    },
]


def mcp_keyed_entry(provider: str) -> dict[str, Any] | None:
    """Look up an MCP_KEYED_SERVERS entry by its provider slug."""
    for s in MCP_KEYED_SERVERS:
        if s["provider"] == provider:
            return s
    return None


def build_authed_probe_body() -> dict[str, Any]:
    """The body sent to authed MCP servers in slice B-static.

    Same as slice A — initialize is the universal first message, most
    classifiable. Once we have a real session token we *could* go
    further (tools/list to enumerate the server's actual tool surface),
    but for SASE-classification testing the initialize is the
    important wire shape.
    """
    return build_initialize_request()


def headers_for_keyed(
    server: dict[str, Any],
    api_key: str,
    protocol_version: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """Headers for an authed MCP server probe — transport headers
    plus the server-specific auth header."""
    h = dict(headers_for(
        server["transport"],
        protocol_version=protocol_version,
        session_id=session_id,
    ))
    h["User-Agent"] = server["user_agent"]
    h[server["auth_header"]] = server["auth_format"].format(key=api_key)
    return h


# ---------------------------------------------------------------------------
# DLP test payload wrapping (slice C)
# ---------------------------------------------------------------------------
#
# The Profile Tests tab will pass payload_shape="mcp" to wrap synthetic
# PII inside an MCP tools/call envelope instead of a chat completion.
# The DLP question being tested: does your DLP engine *parse* MCP and
# inspect the tools/call.params.arguments, or treat the whole body as
# opaque JSON?

def wrap_pii_as_mcp_tool_call(
    pii_value: str,
    pii_label: str,
    prompt_text: str,
) -> dict[str, Any]:
    """Build an MCP tools/call body that embeds PII in the arguments.

    The shape is realistic — a generic 'submit_form' tool with named
    fields, exactly the kind of thing a real MCP-using assistant might
    call when helping a user with a form. The PII goes into the
    arguments dict where a properly-implemented DLP MCP parser
    *should* see it.
    """
    return build_tools_call_request(
        tool_name="submit_form",
        arguments={
            "field_label": pii_label,
            "field_value": pii_value,
            "context": prompt_text,
            "submitted_by": "hairspray-profile-test",
        },
    )
