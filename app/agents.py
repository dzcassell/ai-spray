"""Agents — coder-prompt random-sprinkle traffic against AI assistant CLIs.

Powers the Agents tab in the UI. Two sources of traffic:

* **Anthropic** — calls the `/v1/messages` endpoint directly with a
  `claude-cli` User-Agent. If the `claude` binary is on the container
  PATH at startup, we *prefer* the subprocess invocation over a raw
  HTTP call (real CLI, real wire shape, including its specific
  multipart-message-batching behavior). Otherwise we fall back to
  raw API. Either way the wire signature is what an enterprise SASE
  needs to classify as Claude Code.

* **Cursor** — uses Cursor's User API Key (issued via the Cursor
  Integrations Dashboard for the headless CLI) against
  `api2.cursor.sh`. The Cursor binary is *not* installed in the
  container — it auto-updates aggressively on every launch which
  generates noisy classifiable traffic of its own and breaks
  predictable container behavior. We hit the API directly with the
  CLI's documented User-Agent.

The "fire" loop runs on the server, not the browser, so it survives
page reloads and tab closes. State lives in AgentLoopState; the
single asyncio.Task is started on /api/agents/start and cancelled
on /api/agents/stop. Random gap between fires is configurable in
the UI (defaults: 60-120s). On each fire we pick a random enabled
prompt × a random enabled provider, fire it, record the result in
a ring buffer, and sleep until the next fire.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import shutil
import subprocess  # nosec B404 — used only for the well-known `claude` binary
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Predefined coder prompt matrix
# ---------------------------------------------------------------------------
# Ten prompts spanning five genres. Each prompt has a slug for the UI to
# reference, a genre (for filtering / display grouping), and the prompt
# text itself. Prompts are chosen to:
#   * be small enough that the wire signature is the prompt body, not
#     a 4KB context dump
#   * not require file-system access or shell execution (so neither the
#     real Claude CLI nor a Cursor agent tries to mutate anything)
#   * exercise different traffic shapes — short Q&A, code-gen with
#     longer responses, code-review with longer prompts, etc.

PROMPTS: list[dict[str, str]] = [
    # ---- Pure Q&A (short prompt, short response) ----
    {
        "slug":  "qa-btree",
        "genre": "qa",
        "label": "Q&A: B-tree vs B+ tree",
        "text":  "In two short paragraphs, explain how a B-tree differs "
                 "from a B+ tree, and when you'd choose one over the other.",
    },
    {
        "slug":  "qa-cap",
        "genre": "qa",
        "label": "Q&A: CAP theorem",
        "text":  "Briefly: what does the CAP theorem actually say, and "
                 "what's the most common misunderstanding of it?",
    },

    # ---- Code generation (short prompt, longer response) ----
    {
        "slug":  "gen-palindrome",
        "genre": "codegen",
        "label": "Code: palindrome detector in Python",
        "text":  "Write a Python function `is_palindrome(s: str) -> bool` "
                 "that ignores case and non-alphanumeric characters. "
                 "Include three short doctests.",
    },
    {
        "slug":  "gen-binsearch-rust",
        "genre": "codegen",
        "label": "Code: binary search in Rust",
        "text":  "Write an idiomatic Rust function for binary search over "
                 "a `&[i32]`. Use the standard signature returning a "
                 "`Result<usize, usize>` matching slice::binary_search. "
                 "No external crates.",
    },

    # ---- Code review (long prompt with embedded code, terse response) ----
    {
        "slug":  "review-py-bug",
        "genre": "review",
        "label": "Review: subtle Python bug",
        "text":  ("Review this snippet for bugs. Be specific.\n\n"
                  "```python\n"
                  "def merge_dicts(a, b):\n"
                  "    out = a\n"
                  "    for k, v in b.items():\n"
                  "        out[k] = v\n"
                  "    return out\n"
                  "\n"
                  "user_defaults = {'theme': 'dark', 'lang': 'en'}\n"
                  "alice = merge_dicts(user_defaults, {'lang': 'fr'})\n"
                  "bob   = merge_dicts(user_defaults, {'lang': 'es'})\n"
                  "print(user_defaults)\n"
                  "```\n"),
    },
    {
        "slug":  "review-sql-injection",
        "genre": "review",
        "label": "Review: SQL injection risk",
        "text":  ("Is this code safe? Why or why not?\n\n"
                  "```python\n"
                  "import sqlite3\n"
                  "def find_user(name):\n"
                  "    con = sqlite3.connect('app.db')\n"
                  "    cur = con.cursor()\n"
                  "    cur.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
                  "    return cur.fetchall()\n"
                  "```\n"),
    },

    # ---- Refactoring (long prompt, long response) ----
    {
        "slug":  "refactor-fizzbuzz",
        "genre": "refactor",
        "label": "Refactor: clarify a tangled function",
        "text":  ("Refactor for clarity. Keep the same behavior.\n\n"
                  "```python\n"
                  "def f(n):\n"
                  "    r = []\n"
                  "    for i in range(1, n+1):\n"
                  "        x = ''\n"
                  "        if i % 3 == 0: x += 'Fizz'\n"
                  "        if i % 5 == 0: x += 'Buzz'\n"
                  "        r.append(x or str(i))\n"
                  "    return r\n"
                  "```\n"),
    },

    # ---- Debugging (long prompt with stack trace, focused response) ----
    {
        "slug":  "debug-asyncio",
        "genre": "debug",
        "label": "Debug: asyncio RuntimeError",
        "text":  ("What's likely going wrong here, and how do I fix it?\n\n"
                  "```\n"
                  "RuntimeError: This event loop is already running\n"
                  "  File \"app.py\", line 42, in handle\n"
                  "    result = asyncio.run(do_thing())\n"
                  "  File \".../asyncio/runners.py\", line 33, in run\n"
                  "    raise RuntimeError(\n"
                  "```\n\n"
                  "Context: this is inside a request handler in a "
                  "FastAPI app."),
    },

    # ---- Architecture (moderate prompt, very long response) ----
    {
        "slug":  "arch-rate-limiter",
        "genre": "architecture",
        "label": "Architecture: distributed rate limiter",
        "text":  "Sketch the design of a rate limiter that works across "
                 "a horizontally-scaled web tier (~50 nodes). Cover: "
                 "what backing store, what algorithm, how you handle "
                 "the 'thundering herd' near reset boundaries, and "
                 "what the failure mode is if the backing store is "
                 "briefly unavailable.",
    },
    {
        "slug":  "arch-feature-flags",
        "genre": "architecture",
        "label": "Architecture: feature-flag system",
        "text":  "What are the must-have properties of a feature-flag "
                 "system for a 200-engineer org? Cover propagation "
                 "latency, percentage rollouts, kill-switch semantics, "
                 "and how you avoid the flag store becoming a single "
                 "point of failure.",
    },
]

PROMPT_BY_SLUG: dict[str, dict[str, str]] = {p["slug"]: p for p in PROMPTS}
PROMPT_GENRES: tuple[str, ...] = (
    "qa", "codegen", "review", "refactor", "debug", "architecture",
)


# ---------------------------------------------------------------------------
# Agentic prompts — designed to *force* tool use
# ---------------------------------------------------------------------------
# Each task is short enough to cap at ~6 turns but structured so the model
# cannot complete it from memory — it must Read, Write, run Bash, or
# WebFetch. The traffic this produces is exactly the mixed bag a SASE
# fabric sees in real assistant use: model API calls plus file I/O,
# shell, and outbound HTTPS fetches all signed by the CLI's wire UA.
#
# All tasks run in /tmp/agent-sandbox (tmpfs, wiped per container start)
# so they cannot affect the rest of the container.

AGENTIC_PROMPTS: list[dict[str, str]] = [
    {
        "slug":  "agentic-fizzbuzz",
        "genre": "agentic",
        "label": "Agentic: build & run fizzbuzz variants",
        "text":  ("In the current directory, create three Python files "
                  "fizz_v1.py, fizz_v2.py, fizz_v3.py each printing "
                  "FizzBuzz for n=1..15 at increasing levels of "
                  "conciseness (terse to one-liner). Run each with "
                  "`python3` and report which produced the shortest "
                  "source while still producing the correct output."),
    },
    {
        "slug":  "agentic-httpbin",
        "genre": "agentic",
        "label": "Agentic: fetch httpbin and summarise",
        "text":  ("Use WebFetch to retrieve https://httpbin.org/json. "
                  "List the top-level keys in the response, and explain "
                  "in one sentence what each represents."),
    },
    {
        "slug":  "agentic-readdir",
        "genre": "agentic",
        "label": "Agentic: inspect cwd",
        "text":  ("List the files in the current working directory. "
                  "Pick the largest one, read its first ten lines, and "
                  "summarise what kind of file it is."),
    },
    {
        "slug":  "agentic-sha256",
        "genre": "agentic",
        "label": "Agentic: write a hashing tool",
        "text":  ("Write a Python script `hash_it.py` that reads a "
                  "string from argv and prints its SHA-256 hex digest. "
                  "Run it on the string 'sasetest' and confirm the "
                  "output looks like a 64-character hex string."),
    },
    {
        "slug":  "agentic-wiki-metar",
        "genre": "agentic",
        "label": "Agentic: research METAR format",
        "text":  ("Use WebFetch to read "
                  "https://en.wikipedia.org/wiki/METAR. Write a file "
                  "`metar_notes.md` (~10 lines) explaining how METAR "
                  "weather reports are structured, citing the page."),
    },
    {
        "slug":  "agentic-csv",
        "genre": "agentic",
        "label": "Agentic: synth a CSV and analyse it",
        "text":  ("Create a CSV file `sample.csv` with five rows of "
                  "made-up sales data (date, region, amount in USD). "
                  "Then write and run a Python script `analyse.py` "
                  "that reads the CSV and prints the region with the "
                  "highest total."),
    },
]

AGENTIC_PROMPT_BY_SLUG: dict[str, dict[str, str]] = {
    p["slug"]: p for p in AGENTIC_PROMPTS
}

# Tools we let the CLI use inside the sandbox. Deliberately allowlist
# specific Bash command prefixes rather than blanket-granting Bash — a
# sandbox plus a tight allowlist is the layered defence the lab needs.
AGENTIC_ALLOWED_TOOLS = ",".join([
    "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)",
    "Bash(wc:*)", "Bash(python3:*)", "Bash(sha256sum:*)",
    "Bash(echo:*)", "Bash(grep:*)", "Bash(find:*)",
    "Read", "Write", "Edit", "WebFetch",
])

# Working directory for every agentic fire. The compose tmpfs mount
# wipes it on every container restart; per-fire wiping happens in
# _fire_anthropic_via_cli_agentic before each invocation.
AGENT_SANDBOX_DIR = "/tmp/agent-sandbox"

# Providers we can sprinkle. Each entry knows how to fire one prompt at
# the matching service. Ordered: anthropic first (because the real CLI
# may be present), cursor second.
PROVIDERS: tuple[str, ...] = ("anthropic", "cursor")


# ---------------------------------------------------------------------------
# Loop state
# ---------------------------------------------------------------------------

@dataclass
class AgentToolCall:
    """One Read/Write/Bash/WebFetch call observed inside an agentic fire.

    Captured from Claude Code's stream-json output so the UI can render
    a per-fire tool-call timeline. ``input_summary`` is a short
    human-readable description of the call's arguments — the full
    input dict can be massive (code blobs, web responses) and isn't
    useful in the timeline view.
    """
    tool: str          # 'Bash', 'Read', 'Write', 'WebFetch', etc.
    input_summary: str
    elapsed_ms: int    # offset from the fire's started_at


@dataclass
class AgentFireResult:
    """One completed agent fire — what gets shown in the Agents tab."""
    provider: str             # "anthropic" | "cursor"
    prompt_slug: str
    prompt_label: str
    started_at: float         # epoch seconds
    elapsed_ms: int
    ok: bool
    response_text: str        # the model's response, or error message if !ok
    response_chars: int       # length of response_text
    error: str | None = None  # short error class name on failure
    # Agentic-only fields. Populated by _fire_anthropic_via_cli_agentic
    # when the loop runs in agentic mode; left at defaults for chat
    # fires so the UI can branch on `mode`.
    mode: str = "chat"                                     # "chat" | "agentic"
    num_turns: int = 0
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclass
class AgentLoopState:
    """Server-side state for the random-sprinkle loop.

    Lives in AppState so it survives page reloads. One asyncio.Task at
    most; cancelled on stop. fire_history is a deque so it auto-trims
    to the last N results.
    """
    running: bool = False
    task: asyncio.Task | None = None
    started_at: float | None = None  # epoch seconds when current run began
    total_fired: int = 0
    enabled_prompts: set[str] = field(
        default_factory=lambda: set(p["slug"] for p in PROMPTS)
    )
    enabled_providers: set[str] = field(
        default_factory=lambda: set(PROVIDERS)
    )
    min_gap_sec: int = 60
    max_gap_sec: int = 120
    fire_history: deque[AgentFireResult] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    # Agentic mode: when True, Anthropic fires use the Claude Code CLI
    # with --max-turns + --allowed-tools so the model exercises real
    # file/shell/web tools instead of one-shot completions. Requires
    # the `claude` binary on PATH (Dockerfile.agentic) and an
    # ANTHROPIC_API_KEY in the KeyStore. Falls back gracefully to
    # chat mode if either is missing.
    agentic_mode: bool = False
    daily_token_budget: int = field(
        default_factory=lambda: int(
            os.environ.get("AGENTIC_DAILY_TOKEN_BUDGET", "500000")
        )
    )
    daily_tokens_used: int = 0
    daily_tokens_reset_at: float = 0.0  # epoch; loop rolls over at UTC midnight
    agentic_min_gap_sec: int = field(
        default_factory=lambda: int(
            os.environ.get("AGENTIC_MIN_GAP_SEC", "300")
        )
    )

    def reset_daily_budget_if_due(self) -> None:
        """Roll the daily token counter over at UTC midnight."""
        now = time.time()
        if now >= self.daily_tokens_reset_at:
            self.daily_tokens_used = 0
            # Next midnight UTC.
            now_dt = datetime.now(timezone.utc)
            tomorrow = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            self.daily_tokens_reset_at = tomorrow.timestamp() + 86400.0

    def budget_remaining(self) -> int:
        return max(0, self.daily_token_budget - self.daily_tokens_used)

    def status(self) -> dict[str, Any]:
        """JSON-safe snapshot for /api/agents/status."""
        self.reset_daily_budget_if_due()
        return {
            "running":           self.running,
            "started_at":        self.started_at,
            "total_fired":       self.total_fired,
            "enabled_prompts":   sorted(self.enabled_prompts),
            "enabled_providers": sorted(self.enabled_providers),
            "min_gap_sec":       self.min_gap_sec,
            "max_gap_sec":       self.max_gap_sec,
            "agentic_mode":      self.agentic_mode,
            "agentic_available": claude_cli_available(),
            "daily_token_budget":   self.daily_token_budget,
            "daily_tokens_used":    self.daily_tokens_used,
            "daily_tokens_reset_at": self.daily_tokens_reset_at,
            "agentic_min_gap_sec":  self.agentic_min_gap_sec,
            "history": [
                dataclasses.asdict(r) for r in self.fire_history
            ],
        }


# ---------------------------------------------------------------------------
# Real-CLI detection
# ---------------------------------------------------------------------------

def claude_cli_available() -> bool:
    """Did the operator install the `claude` binary in the container?

    We don't try to install it ourselves — the operator either bakes
    it into a derived image or doesn't. Detection is just `shutil.which`.
    """
    return shutil.which("claude") is not None


# ---------------------------------------------------------------------------
# Anthropic dispatcher (real CLI when available, API fallback otherwise)
# ---------------------------------------------------------------------------

# Anthropic API base. If a future user has ANTHROPIC_BASE_URL set in
# the environment we honor it (e.g. for routing through their own
# proxy or a Bedrock-style gateway), but the default is the Anthropic
# public endpoint.
ANTHROPIC_API_BASE = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://api.anthropic.com",
).rstrip("/")
ANTHROPIC_DEFAULT_MODEL = os.environ.get(
    "AGENTS_ANTHROPIC_MODEL", "claude-sonnet-4-5",
)

# User-Agent matching the official `claude` CLI (Anthropic Claude Code).
# Real claude-code/v1.x. Updated April 2026; SASE classifiers key off
# this UA prefix.
ANTHROPIC_CLI_USER_AGENT = "claude-cli/1.0 (sasetest hAIrspray)"


async def _fire_anthropic_via_cli(
    prompt: dict[str, str],
    api_key: str,
    timeout: float,
) -> tuple[bool, str, str | None]:
    """Run `claude -p '<prompt>'` and capture stdout. Returns
    (ok, response_text, error_class_name)."""
    # Subprocess in a thread so we don't block the event loop.
    def _run() -> tuple[int, str, str]:
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = api_key
        proc = subprocess.run(  # nosec B603 — well-known binary, fixed args
            ["claude", "-p", prompt["text"], "--model",
             ANTHROPIC_DEFAULT_MODEL],
            capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr

    try:
        rc, stdout, stderr = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired"
    except FileNotFoundError:
        # Race with config — `claude` was on PATH at startup but isn't
        # now. Fall back to API.
        return False, "", "FileNotFoundError"
    if rc != 0:
        return False, stderr.strip()[:1000] or "non-zero exit", \
               f"ExitCode{rc}"
    return True, stdout.strip(), None


def _summarise_tool_input(tool: str, inp: dict[str, Any]) -> str:
    """Short, log-safe description of a tool_use's input."""
    if tool == "Bash":
        cmd = inp.get("command", "")
        return cmd[:120] + ("…" if len(cmd) > 120 else "")
    if tool in ("Read", "Edit"):
        path = inp.get("file_path") or inp.get("path", "")
        return str(path)[:120]
    if tool == "Write":
        path = inp.get("file_path") or inp.get("path", "")
        n = len(inp.get("content", "")) if isinstance(inp.get("content"), str) else 0
        return f"{path} ({n} chars)"[:120]
    if tool == "WebFetch":
        url = inp.get("url", "")
        return str(url)[:120]
    # Unknown tool — show a JSON-ish hint without dumping the whole dict.
    try:
        return json.dumps(inp)[:120]
    except (TypeError, ValueError):
        return "(unrepresentable input)"


def _parse_stream_json_lines(
    raw: str, started: float,
) -> tuple[list[AgentToolCall], int, int, int, float, str]:
    """Parse Claude Code's --output-format stream-json output.

    Returns ``(tool_calls, num_turns, input_tokens, output_tokens,
    total_cost_usd, final_result_text)``. Unknown event types are
    skipped silently — the format evolves and we'd rather tolerate
    new keys than crash an unrelated fire.
    """
    tool_calls: list[AgentToolCall] = []
    num_turns = 0
    in_tok = 0
    out_tok = 0
    cost_usd = 0.0
    result_text = ""

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")

        if et == "assistant":
            num_turns += 1
            msg = ev.get("message") or {}
            for blk in msg.get("content") or []:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "tool_use":
                    tool_calls.append(AgentToolCall(
                        tool=str(blk.get("name", "?")),
                        input_summary=_summarise_tool_input(
                            str(blk.get("name", "?")),
                            blk.get("input") or {},
                        ),
                        elapsed_ms=int((time.time() - started) * 1000),
                    ))

        elif et == "result":
            # Final summary event. Contains the full text reply,
            # usage tokens, and (sometimes) the cost in USD.
            if isinstance(ev.get("result"), str):
                result_text = ev["result"]
            usage = ev.get("usage") or {}
            try:
                in_tok = int(usage.get("input_tokens", 0))
                out_tok = int(usage.get("output_tokens", 0))
                in_tok += int(usage.get("cache_creation_input_tokens", 0))
                in_tok += int(usage.get("cache_read_input_tokens", 0))
            except (TypeError, ValueError):
                pass
            try:
                cost_usd = float(ev.get("total_cost_usd", 0.0))
            except (TypeError, ValueError):
                pass

    return tool_calls, num_turns, in_tok, out_tok, cost_usd, result_text


async def _fire_anthropic_via_cli_agentic(
    prompt: dict[str, str],
    api_key: str,
    timeout: float,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    """Run `claude -p '<prompt>' --max-turns N --allowed-tools ...`
    inside the sandbox dir and parse the stream-json output.

    Returns ``(ok, response_text, error_class_name, telemetry)`` where
    telemetry carries ``tool_calls``, ``num_turns``, ``input_tokens``,
    ``output_tokens``, and ``total_cost_usd`` for the caller to attach
    to the AgentFireResult.
    """
    started = time.time()

    # Wipe the sandbox dir each fire so accumulated state from past
    # fires never bleeds into a new run. The mkdir is defensive in
    # case tmpfs hasn't been mounted (e.g. running outside compose).
    try:
        os.makedirs(AGENT_SANDBOX_DIR, exist_ok=True)
        for name in os.listdir(AGENT_SANDBOX_DIR):
            path = os.path.join(AGENT_SANDBOX_DIR, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
            except OSError:
                pass
    except OSError as e:
        log.warning("agentic_sandbox_setup_failed", error=str(e))
        return False, "", f"SandboxSetup:{type(e).__name__}", {}

    def _run() -> tuple[int, str, str]:
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = api_key
        # Belt-and-braces: even if the image env didn't carry these,
        # set them here so the CLI behaves deterministically.
        env.setdefault("DISABLE_AUTOUPDATER", "1")
        proc = subprocess.run(  # nosec B603 — well-known binary, fixed args
            [
                "claude", "-p", prompt["text"],
                "--model", ANTHROPIC_DEFAULT_MODEL,
                "--max-turns", "6",
                "--allowed-tools", AGENTIC_ALLOWED_TOOLS,
                "--permission-mode", "acceptEdits",
                "--output-format", "stream-json",
                "--verbose",
            ],
            capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
            cwd=AGENT_SANDBOX_DIR,
        )
        return proc.returncode, proc.stdout, proc.stderr

    try:
        rc, stdout, stderr = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired", {}
    except FileNotFoundError:
        return False, "", "FileNotFoundError", {}

    tool_calls, num_turns, in_tok, out_tok, cost, result_text = (
        _parse_stream_json_lines(stdout, started)
    )
    telemetry = {
        "tool_calls":    tool_calls,
        "num_turns":     num_turns,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "total_cost_usd": cost,
    }

    if rc != 0:
        err_msg = stderr.strip()[:1000] or "non-zero exit"
        return False, err_msg, f"ExitCode{rc}", telemetry
    return True, result_text.strip(), None, telemetry


async def _fire_anthropic_via_api(
    client: httpx.AsyncClient,
    prompt: dict[str, str],
    api_key: str,
    timeout: float,
) -> tuple[bool, str, str | None]:
    """POST /v1/messages with x-api-key auth and a CLI-style UA."""
    headers = {
        "x-api-key":          api_key,
        "anthropic-version":  "2023-06-01",
        "Content-Type":       "application/json",
        "User-Agent":         ANTHROPIC_CLI_USER_AGENT,
    }
    body = {
        "model":      ANTHROPIC_DEFAULT_MODEL,
        "max_tokens": 1024,
        "messages":   [{"role": "user", "content": prompt["text"]}],
    }
    url = f"{ANTHROPIC_API_BASE}/v1/messages"
    try:
        r = await client.post(
            url, headers=headers, json=body, timeout=timeout,
        )
    except httpx.HTTPError as e:
        return False, "", type(e).__name__
    if r.status_code >= 400:
        return False, r.text[:1000], f"HTTP{r.status_code}"
    try:
        data = r.json()
        # Anthropic /v1/messages returns content blocks. Concatenate the
        # text blocks. Tool-use blocks are ignored — these prompts don't
        # ask for tools.
        parts = [
            blk.get("text", "")
            for blk in data.get("content", [])
            if blk.get("type") == "text"
        ]
        return True, "".join(parts).strip(), None
    except (ValueError, KeyError) as e:
        return False, "", type(e).__name__


async def fire_anthropic(
    client: httpx.AsyncClient,
    prompt: dict[str, str],
    api_key: str,
    *,
    agentic: bool = False,
) -> AgentFireResult:
    """Fire one prompt at Anthropic.

    Three execution paths in priority order:
      1. agentic mode + CLI available → multi-turn run with tools,
         stream-json parsed for the timeline + token usage.
      2. CLI available (chat mode) → one-shot `claude -p ...` subprocess.
      3. No CLI → raw POST /v1/messages with a claude-cli UA.
    """
    started = time.time()
    timeout = 180.0 if agentic else 60.0  # agentic runs take much longer

    telemetry: dict[str, Any] = {}
    if agentic and claude_cli_available():
        ok, text, err, telemetry = await _fire_anthropic_via_cli_agentic(
            prompt, api_key, timeout,
        )
        mode = "agentic"
    elif claude_cli_available():
        ok, text, err = await _fire_anthropic_via_cli(
            prompt, api_key, timeout,
        )
        mode = "chat"
    else:
        ok, text, err = await _fire_anthropic_via_api(
            client, prompt, api_key, timeout,
        )
        mode = "chat"

    elapsed_ms = int((time.time() - started) * 1000)
    return AgentFireResult(
        provider="anthropic",
        prompt_slug=prompt["slug"],
        prompt_label=prompt["label"],
        started_at=started,
        elapsed_ms=elapsed_ms,
        ok=ok,
        response_text=text,
        response_chars=len(text),
        error=err,
        mode=mode,
        num_turns=int(telemetry.get("num_turns", 0)),
        tool_calls=list(telemetry.get("tool_calls", [])),
        input_tokens=int(telemetry.get("input_tokens", 0)),
        output_tokens=int(telemetry.get("output_tokens", 0)),
        total_cost_usd=float(telemetry.get("total_cost_usd", 0.0)),
    )


# ---------------------------------------------------------------------------
# Cursor dispatcher (API only — binary deliberately not installed)
# ---------------------------------------------------------------------------

# Cursor's headless API surface. Cursor does NOT expose a public
# chat-completions endpoint that accepts user API keys for inference —
# their BYOK only works inside the IDE, where the user's
# OpenAI/Anthropic key is routed through Cursor's backend.
#
# What they DO expose at api.cursor.com/v0/* is the **Background
# Agents API**, which takes a User API Key and is what `cursor-agent`
# uses for its remote/headless mode. That's what hAIrspray fires
# against. From a SASE classification standpoint this is the wire
# shape that actually identifies Cursor traffic — which is exactly
# what we want to test.
#
# Two endpoints get exercised per fire:
#   GET  /v0/me            — lightweight key-validation hit
#   POST /v0/agents        — launches a Background Agent with the
#                            coder prompt as its initial instruction
# Both go out on api.cursor.com, both with Bearer auth, both with a
# cursor-agent UA. The agent itself doesn't actually run repo work
# (we don't supply a source.repository, so creation 4xx's), but
# the *POST hits the wire* — which is what matters for classification.
CURSOR_API_BASE = os.environ.get(
    "CURSOR_BASE_URL", "https://api.cursor.com",
).rstrip("/")
CURSOR_CLI_USER_AGENT = "cursor-agent/0.4 (sasetest hAIrspray)"


async def fire_cursor(
    client: httpx.AsyncClient,
    prompt: dict[str, str],
    api_key: str,
) -> AgentFireResult:
    """Fire one prompt at Cursor's Background Agents API.

    No chat-completions endpoint exists on the public Cursor API
    surface for user keys — their docs only expose /v0/me and
    /v0/agents. We POST a Background Agent creation request with
    the coder prompt as the agent's initial instruction. The
    request will likely 4xx (no repo supplied, free-tier limits)
    but the wire shape is what SASE classifies.
    """
    started = time.time()
    timeout = 30.0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "User-Agent":    CURSOR_CLI_USER_AGENT,
    }
    body = {
        # Background Agents API shape per Cursor docs (April 2026).
        # The 'prompt' field carries the human's instruction. Other
        # fields (source.repository, model) are normally required for
        # a real run; omitting them produces a 400 — which is fine,
        # the POST went out the wire either way and that's what the
        # classifier sees.
        "prompt": {
            "text": prompt["text"],
        },
        "model": "auto",
    }
    url = f"{CURSOR_API_BASE}/v0/agents"

    try:
        r = await client.post(
            url, headers=headers, json=body, timeout=timeout,
        )
    except httpx.HTTPError as e:
        elapsed_ms = int((time.time() - started) * 1000)
        return AgentFireResult(
            provider="cursor", prompt_slug=prompt["slug"],
            prompt_label=prompt["label"], started_at=started,
            elapsed_ms=elapsed_ms, ok=False, response_text="",
            response_chars=0, error=type(e).__name__,
        )

    elapsed_ms = int((time.time() - started) * 1000)

    # 200/201 with a JSON body containing an agent ID is success.
    # Anything in 4xx is "fabric saw a real Cursor request" — log it
    # as ok=False with the response body as the result so the operator
    # can see what came back, but it's still useful traffic.
    response_text = r.text[:1500] if r.text else ""
    if r.status_code < 400:
        return AgentFireResult(
            provider="cursor", prompt_slug=prompt["slug"],
            prompt_label=prompt["label"], started_at=started,
            elapsed_ms=elapsed_ms, ok=True,
            response_text=response_text, response_chars=len(response_text),
            error=None,
        )
    return AgentFireResult(
        provider="cursor", prompt_slug=prompt["slug"],
        prompt_label=prompt["label"], started_at=started,
        elapsed_ms=elapsed_ms, ok=False,
        response_text=response_text, response_chars=len(response_text),
        error=f"HTTP{r.status_code}",
    )


# ---------------------------------------------------------------------------
# The random-sprinkle loop itself
# ---------------------------------------------------------------------------

async def run_loop(
    state: AgentLoopState,
    client: httpx.AsyncClient,
    key_provider: Any,
) -> None:
    """Long-running task: fire random (provider, prompt) pairs at random
    intervals until cancelled. Blocks on httpx I/O; cancellation is the
    only way out.

    `key_provider` is the KeyStore. We look up keys per-fire so that if
    the operator pastes/changes a key mid-loop, the next fire uses the
    new value without restart.
    """
    state.running = True
    state.started_at = time.time()
    state.total_fired = 0

    log.info("agent_loop_started",
             min_gap=state.min_gap_sec, max_gap=state.max_gap_sec,
             enabled_prompts=len(state.enabled_prompts),
             enabled_providers=sorted(state.enabled_providers))

    try:
        while True:
            # Snapshot enabled sets — the user can toggle these from the
            # UI mid-loop and we want the next fire to reflect the change.
            providers = [
                p for p in PROVIDERS
                if p in state.enabled_providers
            ]
            chat_prompts = [
                PROMPT_BY_SLUG[s] for s in state.enabled_prompts
                if s in PROMPT_BY_SLUG
            ]

            if not providers or not chat_prompts:
                # Nothing to fire — sleep briefly and check again. Don't
                # spin tightly.
                await asyncio.sleep(5.0)
                continue

            provider = random.choice(providers)
            # Agentic mode runs the agentic library on Anthropic fires;
            # other provider/mode combinations use the chat prompt set.
            # We intentionally don't gate agentic prompts behind the
            # enabled_prompts toggle — the whole agentic library is
            # tiny (six tasks) and the agentic toggle itself is the
            # opt-in.
            if (provider == "anthropic"
                    and state.agentic_mode
                    and claude_cli_available()
                    and state.budget_remaining() > 0):
                prompt = random.choice(AGENTIC_PROMPTS)
            else:
                prompt = random.choice(chat_prompts)

            # Look up the key. If missing, record an error result and
            # continue — the loop should not silently stop because one
            # provider's key was removed.
            try:
                if provider == "anthropic":
                    key = await key_provider.get("anthropic")
                else:
                    key = await key_provider.get("cursor")
            except Exception as e:  # noqa: BLE001
                log.warning("agent_key_lookup_failed",
                            provider=provider, error=str(e))
                key = None

            # Per-fire agentic-mode decision: only for Anthropic, only
            # if the toggle is on, only if the CLI is on PATH, and only
            # if we have budget remaining. Falls back to chat mode in
            # every other case rather than skipping the fire entirely.
            state.reset_daily_budget_if_due()
            fire_agentic = (
                provider == "anthropic"
                and state.agentic_mode
                and claude_cli_available()
                and state.budget_remaining() > 0
            )
            if (provider == "anthropic" and state.agentic_mode
                    and not fire_agentic):
                # Note why we downgraded so the operator can spot
                # mismatched expectations in the log.
                log.info(
                    "agentic_downgraded_to_chat",
                    cli_available=claude_cli_available(),
                    budget_remaining=state.budget_remaining(),
                )

            if not key:
                state.fire_history.append(AgentFireResult(
                    provider=provider, prompt_slug=prompt["slug"],
                    prompt_label=prompt["label"], started_at=time.time(),
                    elapsed_ms=0, ok=False, response_text="",
                    response_chars=0,
                    error=f"no API key saved for {provider}",
                ))
                state.total_fired += 1
            else:
                if provider == "anthropic":
                    result = await fire_anthropic(
                        client, prompt, key, agentic=fire_agentic,
                    )
                else:
                    result = await fire_cursor(client, prompt, key)
                state.fire_history.append(result)
                state.total_fired += 1
                # Charge tokens against the daily budget on every
                # agentic fire (even failed ones — Anthropic charges
                # for partial responses).
                if result.mode == "agentic":
                    state.daily_tokens_used += (
                        result.input_tokens + result.output_tokens
                    )
                log.info("agent_fire",
                         provider=provider, prompt_slug=prompt["slug"],
                         mode=result.mode,
                         ok=result.ok, elapsed_ms=result.elapsed_ms,
                         response_chars=result.response_chars,
                         num_turns=result.num_turns,
                         tool_calls=len(result.tool_calls),
                         tokens=result.input_tokens + result.output_tokens,
                         error=result.error)

            # Sleep until the next fire. Random gap in [min, max] —
            # the agentic floor overrides when agentic mode is on,
            # because agentic runs are ~6x the cost of one-shots.
            lo = state.min_gap_sec
            hi = max(state.max_gap_sec, lo)
            if state.agentic_mode:
                lo = max(lo, state.agentic_min_gap_sec)
                hi = max(hi, lo)
            gap = random.uniform(lo, hi)
            await asyncio.sleep(gap)

    except asyncio.CancelledError:
        log.info("agent_loop_cancelled", total_fired=state.total_fired)
        raise
    finally:
        state.running = False
        state.task = None
