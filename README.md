# hAIrspray

**Generate traffic at public AI applications from a single box so your
SASE / NGFW / SWG shows you whether it can actually identify, classify,
and control them.**

![hAIrspray Prompt tab — 160 AI text and image models in one grid, one prompt, real streamed responses side by side.](docs/screenshot-prompt.jpg)

<sub>Prompt & Fire — one prompt, 160 model/provider pairs, real responses streamed back inline. 12 keyless providers (Pollinations, DuckDuckGo AI Chat), 18 keyed free-tier providers (Gemini, Groq, Mistral, Cohere, OpenRouter, HuggingFace, Together, Cerebras, SambaNova, Hyperbolic, DeepSeek, xAI, AI21, Fireworks, NVIDIA NIM, GitHub Models, Anthropic, Cursor). Live dynamic-discovery populates the grid with whatever models each key actually unlocks.</sub>

Vendors claim AI-app visibility. hAIrspray lets you find out. Point it
at your corporate egress, enable the probes relevant to your test, and
watch your security fabric's application-analytics dashboard light up —
or fail to.

> _Created by Damon Cassell · Cato Networks · No Warranty · You Accept
> All Risk._

---

## What this is for

Modern SASE platforms and next-generation firewalls market AI-app
inspection as a first-class feature: policy rules that target
"OpenAI ChatGPT" or "Generative AI" as a category, DLP over GenAI
prompt bodies, block lists for specific model providers, and so on.

The only way to know whether a given vendor's app-ID actually sees
what it claims to see is to generate realistic traffic at the real
services and look at what the fabric reports. hAIrspray is that
traffic generator. It runs as a container behind your SASE fabric (or
through your NGFW egress) and fires outbound requests at **181 curated
AI endpoints** so you can:

- Verify your vendor's app signatures fire for each AI service you
  care about (not just OpenAI).
- Measure classification latency — how fast does the first hit become
  a matched app event?
- Test policy rules: block categories, rate-limit models, inspect
  prompt bodies with DLP, force TLS inspection on AI gateways.
- Compare vendors side-by-side on the same probe catalog.
- Produce reproducible sample traffic for customer demos or MSP lab
  setups.

## What actually goes out the wire

- **181 probes across 7 categories**: LLM APIs (OpenAI, Anthropic,
  Gemini, Groq, Mistral, Cohere, and ~16 more), chatbot UIs (ChatGPT,
  Claude.ai, Gemini Web, Copilot, Character.AI, and ~27 more), media
  generation (Midjourney, Leonardo, Runway, Suno, ElevenLabs, and ~44
  more), aggregators (HuggingFace, Replicate, OpenRouter, Cursor,
  Lovable, and ~50 more), MCP synthetic traffic (JSON-RPC payloads
  to reflectors and known public MCP server hostnames — both legacy
  HTTP+SSE and current Streamable HTTP transports), authed MCP traffic
  (real handshakes against GitHub MCP, Notion MCP, and Linear MCP
  when you've saved a token in Config → Keys), and a small "real
  response" set.
- **Realistic request shapes.** LLM-API probes use SDK-style
  User-Agents (`OpenAI/Python 1.51.0`, `anthropic-python/0.39.0`) and
  the right endpoints, bodies, and auth headers. Chatbot-UI probes
  use browser User-Agents and the URLs a browser would fetch. No key
  required — 401/403 is fine, the flow is the point.
- **Real AI responses** for the "Prompt & Fire" flow. Twelve keyless
  model/provider pairs (Pollinations text + image, DuckDuckGo AI Chat)
  return real completions. Eighteen additional providers unlock real
  responses if you paste free-tier API keys (Gemini, Groq, Mistral,
  Cohere, OpenRouter, HuggingFace, Together, Cerebras, SambaNova,
  Hyperbolic, DeepSeek, xAI, AI21, Fireworks, NVIDIA NIM, GitHub
  Models, Anthropic, Cursor).
- **File uploads** in Prompt & Fire. Attach a `.txt` / `.md` / `.csv`
  / `.json` / `.pdf` / `.docx` / `.xlsx` / source-code file (or any
  supported text format) and the extracted content rides along in
  the prompt body. The realistic data-loss vector — "user uploaded
  a quarterly-results spreadsheet to ChatGPT" — becomes a one-click
  test against your DLP fabric. Server-side extraction via pypdf,
  python-docx, openpyxl. 10 MB upload cap, ~20K char extraction cap.
- **Configurable pacing**: 30–180s random gaps between probes by
  default, plus optional burst mode (3–7 back-to-back requests at
  1–4s intervals) for session-like patterns.
- **TLS / HTTP/2** on every endpoint that supports it (`httpx[http2]`).

## Quick start

On a Linux Docker host sitting behind the SASE fabric or NGFW you want
to test:

```bash
cd /opt
sudo git clone https://github.com/dzcassell/hAIrspray.git
sudo chown -R "$USER": hAIrspray
cd hAIrspray

cp .env.example .env
# Edit .env if you want to narrow categories, tune pacing, or bind
# the UI to loopback only.

docker compose up -d --build
docker compose logs -f
```

Then open `http://<host>:8090/` — the web UI is the primary interface.
You can also run it as a systemd service with the unit file in
`systemd/hairspray.service`.

## Running on macOS

hAIrspray runs on current macOS (Apple Silicon, Docker Desktop)
without any code changes. The container itself is Linux, executing
inside Docker Desktop's lightweight VM — macOS is only hosting the
process and forwarding port 8090 to your browser.

### Quick start on macOS

```bash
# Install Docker Desktop if you haven't already:
#   https://www.docker.com/products/docker-desktop/
# Open the app so the whale icon appears in your menu bar. Docker
# Desktop must be running before any `docker compose` command works.

git clone https://github.com/dzcassell/hAIrspray.git ~/Code/hAIrspray
cd ~/Code/hAIrspray

cp .env.example .env
# Optional edits: narrow categories, bind the UI to loopback
# (HEALTH_BIND=127.0.0.1), change the host port, etc.

docker compose up -d --build
docker compose logs -f
```

Open <http://localhost:8090> in Safari or your browser of choice.

### Autostart options

On Linux, autostart is handled by `systemd/hairspray.service`. macOS
has no systemd; you have two options:

**Option 1 (simplest) — Docker Desktop's built-in autostart.** Open
Docker Desktop → Settings → General → check "Start Docker Desktop
when you log in." Because `docker-compose.yml` declares
`restart: unless-stopped`, the hAIrspray container resumes
automatically whenever Docker Desktop starts. For most lab use, this
is all you need.

**Option 2 — launchd agent.** If you want hAIrspray to come up
deterministically on every login, independent of Docker Desktop's
own autostart setting, install the launchd agent shipped in
`macos/`:

```bash
cd ~/Code/hAIrspray
./macos/install.sh
```

What the installer does:

1. Renders `macos/hairspray.plist.template` with your absolute
   paths (launchd doesn't expand `~` or `$HOME` in plist values)
   and writes the result to
   `~/Library/LaunchAgents/ai.hairspray.autostart.plist`.
2. Makes `macos/hairspray-start.sh` executable — this is the
   wrapper launchd actually invokes. It polls `docker info` for up
   to 3 minutes, waiting for Docker Desktop to finish initializing
   its Linux VM (on a cold boot this can take 30–90s), then runs
   `docker compose up -d --build`.
3. Loads the agent with `launchctl load`, so it runs immediately
   and on every login thereafter.

Logs go to `~/Library/Logs/hAIrspray-launchd.log`. To check status:

```bash
launchctl list | grep ai.hairspray.autostart
```

To remove the agent later:

```bash
launchctl unload ~/Library/LaunchAgents/ai.hairspray.autostart.plist
rm ~/Library/LaunchAgents/ai.hairspray.autostart.plist
```

The launchd agent runs **per-user** (it's in `~/Library/` not
`/Library/`), so it won't try to start hAIrspray before the user
logs in. Docker Desktop is also per-user, so this matches the
expected lifecycle.

### Things that are different on macOS

- **No systemd.** The `systemd/hairspray.service` file is Linux-only;
  ignore it on Mac.
- **Paths.** The Linux quick-start uses `/opt`; on Mac, `~/Code/hAIrspray`
  is more idiomatic, and you don't need `sudo` to clone into your
  home directory.
- **Named volumes, not bind mounts.** hAIrspray only uses named Docker
  volumes, so you don't hit Docker Desktop's bind-mount file-sharing
  performance quirks. Saved API keys live in the `ai-spray-keys`
  volume inside Docker Desktop's VM and persist across container
  rebuilds.
- **Resource limits.** Docker Desktop caps CPU/RAM from the Mac's
  pool; the defaults (~4 CPUs, 8 GB RAM) are more than enough for
  hAIrspray, which uses well under 200 MB at full tilt.
- **No `tini` install needed.** The container already bundles tini;
  macOS doesn't care what runs as PID 1 inside the Linux VM.

## The UI

Five tabs:

- **Prompt & Fire** — send a real prompt to every keyless (and keyed,
  if you saved keys) AI provider at once. Responses stream back
  inline. This is the fastest way to show a SASE demo audience that
  AI DLP either is or isn't inspecting the reply body. **Attach a
  file** (text, source code, PDF, DOCX, XLSX) via the 📎 button or
  by dragging onto the textarea — content is extracted server-side
  (pypdf for PDFs, python-docx for Word, openpyxl for spreadsheets)
  and prepended to the prompt as a labeled block, so DLP sees the
  full document content on the wire. Caps: 10 MB upload, ~20K
  characters of extracted text per send.
- **App Probes** — the full 181-entry probe catalog. Filter by name,
  URL, or category; enable/disable individual probes; fire a single
  probe, fire an entire category, or fire every enabled probe once
  (concurrency-capped) via **⚡ Fire All**.
- **Profile Tests** — synthetic-PII DLP testing. Pick a model + locale
  + payload shape, tick categories of PII to test (Address, Phone,
  SSN, IBAN, Credit Card, Passport, MRN, etc. — 17 categories), fire
  individually or in bulk, and see which payloads were blocked,
  partially redacted, or echoed verbatim. The **Payload** dropdown
  switches between standard chat-completion bodies and MCP
  `tools/call` envelopes — the latter tests whether your DLP engine
  actually parses MCP payloads or treats them as opaque JSON.
- **Agents** — random-sprinkle coder prompts at Anthropic Claude and
  Cursor every 60–120 seconds (configurable). Pick a prompt matrix
  (10 prompts spanning Q&A / code-gen / review / refactor / debug /
  architecture genres), pick which providers participate, click
  Fire, and the loop runs server-side until you click Stop. Realistic
  agentic-AI traffic that increasingly enterprise SASE/NGFW vendors
  claim to identify. If the `claude` CLI binary is on the container's
  PATH at startup, Anthropic fires use the real subprocess (matching
  wire shape down to the SDK's specific batching behavior); otherwise
  it falls back to direct `/v1/messages` API calls with a
  `claude-cli` User-Agent. Cursor always goes via API on
  `api2.cursor.sh` with a `cursor-agent` UA — installing the Cursor
  binary in a long-running container is operationally fragile (it
  auto-updates on every launch).
- **Config** — scheduler knobs (min/max interval, burst probability,
  burst size, burst gap, category toggles) and the persistent key
  store. Eighteen keyed AI providers (Groq, Mistral, Gemini, Cohere,
  OpenRouter, HuggingFace, Together, Cerebras, SambaNova, Hyperbolic,
  DeepSeek, xAI, AI21, Fireworks, NVIDIA NIM, GitHub Models, Anthropic,
  Cursor) plus three keyed MCP servers (GitHub MCP, Notion MCP, Linear
  MCP).
  **Dynamic model discovery**: when you save an AI key, hAIrspray
  calls the provider's `/v1/models` endpoint with it and caches
  whatever catalog that key unlocks, so the Prompt grid always
  renders models your key actually has access to. A **↻** button per
  row re-runs discovery when a vendor churns their catalog.
- **Monitor** — live stats (totals, per-category bars, OK vs error
  counts) and an SSE-streamed event log with search, category
  filter, status filter, and NDJSON export.

## A note on obtaining free API keys

The easiest way to stand up accounts across the sixteen keyed
providers is with a Gmail address. Most of them accept **"Sign in
with Google"** for signup, so a single Google account gets you keys
across most of the catalog in roughly a minute each — no need to
create a dozen-plus separate accounts, verify email on each, set a
new password each, and so on.

Providers that accept Google OAuth for signup (at time of writing):

- Google Gemini (obviously — it's theirs)
- Groq, Mistral, Cohere
- OpenRouter, Together, Fireworks, Cerebras
- HuggingFace, Hyperbolic, SambaNova, AI21
- NVIDIA NIM (via the NVIDIA Developer Program, which itself
  accepts Google)

The two-and-a-half exceptions worth knowing about:

- **GitHub Models** uses your GitHub account — fair enough, since
  the whole point of the service is that you're already on GitHub.
  Generate a personal access token with the `models:read` scope
  and paste that instead.
- **DeepSeek** typically requires email (and sometimes phone)
  signup rather than Google OAuth. It's a Chinese cloud provider;
  the console is mostly English-localized but the signup flow is
  more conventional.
- **xAI** is the half — Google OAuth is sometimes available,
  sometimes not, depending on the current state of their auth
  integrations. X/Twitter auth is the reliably-present alternative.

The Config → Keys panel in hAIrspray puts a **🔑 add key** link next
to every provider that jumps you directly to that vendor's key-
generation page, so you don't need to hunt through each vendor's
documentation to find where they hid it. Expected total time to
populate the whole panel from scratch on a fresh Gmail account:
roughly 20–30 minutes, most of which is clicking through sign-up
flows rather than anything hAIrspray-specific.

## Optional: agentic variant (Claude Code CLI + aider baked in)

The lean image fires Anthropic prompts via the API with a
`claude-cli` User-Agent. If you want the *real* Claude Code binary
in the container — same wire shape Anthropic's CLI emits in its
day-job, plus actual multi-turn tool use (Read/Write/Bash/WebFetch)
producing the heterogeneous traffic real agentic clients generate —
use the prebuilt `Dockerfile.agentic` variant:

```bash
docker compose -f docker-compose.yml -f docker-compose.agentic.yml up -d --build
```

This builds `hairspray:agentic` (~250 MB; +100 MB over the lean
image) with Claude Code installed via the official `claude.ai/install.sh`
script, `aider-chat` via pip, `DISABLE_AUTOUPDATER=1` so the wire
signature stays stable, and a 64 MB tmpfs at `/tmp/agent-sandbox`
as the agent's working directory (wiped per container restart).

In the Agents tab, the **Agentic / Tool use** toggle becomes
operable. With it on:

* Anthropic fires use `claude -p '<prompt>' --max-turns 6
  --allowed-tools '<allowlist>' --permission-mode acceptEdits
  --output-format stream-json` against the sandbox.
* The stream-json output is parsed for tool-use events and final
  usage. Each fire's card in the UI shows a tool-call timeline
  (Bash/Read/Write/WebFetch with arg summaries and elapsed-ms),
  turn count, in/out tokens, and the USD cost when Claude Code
  reports it.
* A daily token budget (`AGENTIC_DAILY_TOKEN_BUDGET`, default 500k)
  is enforced — fires that would exceed the cap downgrade to chat
  mode rather than blocking the loop. Counter resets at UTC midnight.
* The agentic min-gap floor (`AGENTIC_MIN_GAP_SEC`, default 300s)
  overrides chat-mode pacing because multi-turn runs are ~6× the
  one-shot spend.

The Cursor binary is deliberately **not** installed. `cursor-agent`
auto-updates on every launch, which generates noisy traffic and
breaks predictable container behavior. If you want to test that
specifically, install Cursor on a dev workstation behind the same
fabric and let it update there.

## Optional: inbound MCP responder

Set `MCP_RESPONDER_ENABLED=true` in `.env` and hAIrspray will *also*
answer inbound MCP requests at:

* `POST /mcp-responder/streamable`   — Streamable HTTP transport
* `POST /mcp-responder/sse/messages` — legacy HTTP+SSE write half
* `GET  /mcp-responder/sse`          — legacy HTTP+SSE read half
* `DELETE /mcp-responder/streamable` — session termination

Two test cases this unlocks:

1. **Inbound classification.** Point a real Claude Code / Cursor /
   Goose client at `https://<host>/mcp-responder/streamable` and
   observe whether the SASE fabric flags the inbound connection as
   MCP. Useful for shadow-MCP detection rules.

2. **DLP egress against a controllable peer.** Wire a real MCP
   client to the responder, send `tools/call` invocations with
   synthetic PII in the arguments. The responder echoes the
   received arguments back as the tool result, so you have a
   ground-truth view of what crossed the wire — compare against
   what your fabric's DLP engine reported.

Every inbound message publishes into the existing event ring
buffer with `category=mcp_responder`, so they appear alongside
outbound probes in the live log. The header shows an **MCP RESPONDER**
chip while it's live.

**Security**: the responder has **no authentication**. Run it
inside a trusted network segment, or front it with your own auth
proxy. Default is off.

## Typical SASE/NGFW test workflow

1. **Deploy behind the fabric** you want to test. For SASE sockets,
   that's usually a LAN host whose default route is the socket. For
   NGFWs, any host on the inside interface.
2. **Disable everything** in the App Probes tab, then enable only the
   category you're testing (e.g. `llm_api` if you're validating LLM
   API signatures).
3. **Click ⚡ Fire All.** hAIrspray will hit every enabled probe
   once with bounded concurrency (default 10).
4. **Open your vendor's app-analytics dashboard.** For Cato, that's
   _Monitoring → Events → Application Analytics_ filtered by the
   source IP of your hAIrspray host. You should see one matched
   event per probe.
5. **Compare against the probe list.** Any miss is a gap in your
   vendor's signatures or your license/policy tier.
6. **Optional**: save a free-tier API key for Groq or Google Gemini
   in Config, then use Prompt & Fire to submit a prompt. The
   response body is what DLP will see — good for testing GenAI DLP
   rules.

## Architecture

Python 3.12, asyncio, `httpx[http2]` for egress, Starlette + uvicorn
for the web UI/API, structlog for JSON logs. Single-process, single-
container. State lives in memory except for saved API keys, which are
persisted to a Docker-managed named volume at `/data/keys.json`
(mode 0600, schema-versioned JSON). See `app/keys.py`.

Event flow: the scheduler picks a probe, runs it through an
`httpx.AsyncClient`, and publishes a `ProviderResult` into shared
state. The state publishes to an internal ring buffer plus any
connected SSE subscribers, driving the Live Log and stats in the UI.
Fire All and Prompt & Fire use the same publish path, so every
request — scheduled or manual — appears in the same Monitor stream
and in your SASE's logs at the same layer.

## Security caveats

- **No authentication on the web UI.** Bind it to loopback
  (`HEALTH_BIND=127.0.0.1` in `.env`) or keep the host behind your
  SASE fabric. _Do not expose to the public internet._
- **API keys are stored in plaintext** inside a Docker volume. File
  mode is 0600 but anyone with Docker socket access on the host
  (i.e. membership in the `docker` group) can read them. This is a
  lab tool; do not store keys here that would cost real money if
  leaked.
- **This tool generates actual requests at real services.** If you
  leave it running indefinitely at high concurrency it is fully
  capable of tripping rate limiters and getting your source IP
  temporarily banned by some providers. Default pacing is
  deliberately slow.
- **TLS verification: three modes in priority order.** Most SASE
  deployments need the first one.
  1. **Custom CA bundle (preferred).** Drop your SASE/NGFW re-sign
     root CA into `./certs/` as a `.crt`, `.pem`, or `.cer` file
     (see [`certs/README.md`](certs/README.md)). At container boot,
     it's concatenated with certifi's Mozilla store and httpx is
     pointed at the combined bundle. You get **full TLS verification
     on every flow** — including flows re-signed by your fabric, but
     also catching anything the fabric didn't sign. This is how
     enterprise-grade tools are meant to live inside an inspecting
     fabric.
  2. **Stock verification.** If no extras are in `./certs/` and
     `TLS_VERIFY=true`, the container verifies against the built-in
     Mozilla bundle. Correct mode when not behind any inspecting
     fabric.
  3. **Verification bypassed.** If no extras are in `./certs/` and
     `TLS_VERIFY=false` (default), TLS verification is disabled
     entirely. Compatibility mode; the container cannot detect MitM
     on its own in this mode, so only use it where you fully trust
     the network path.

  The boot log always states which mode won. Grep for `tls_custom_ca`,
  `tls_system_verify`, or `tls_verify_disabled` in
  `docker compose logs hairspray`.

## Documentation

Code map:

- `app/registry.py` — the 181-probe catalog
- `app/prompt.py` — the 12 keyless + 18 keyed prompt-capable providers
- `app/discovery.py` — per-provider `/v1/models` fetchers that power
  dynamic model discovery (three request shapes: openai-compatible,
  gemini, cohere)
- `app/keys.py` — persistent key store (Docker volume, mode-0600
  JSON, v2 schema with per-provider model cache)
- `app/extract.py` — file-attachment extraction for the Prompt
  panel: inline-text formats (.txt, .md, .csv, .json, source code,
  …) plus PDF/DOCX/XLSX via pypdf/python-docx/openpyxl, capped at
  ~20K extracted characters per upload
- `app/pii.py` — synthetic-PII generators (17 categories, locale-
  aware) for the Profile Tests tab; checksum-valid for cards
  (Luhn), IBANs (mod-97), and VINs (ISO 3779)
- `app/mcp.py` — MCP JSON-RPC envelope builders, transport headers,
  public-server registry; underpins the synthetic + authed MCP
  probes and the slice-C MCP-payload-shape DLP toggle
- `app/agents.py` — Agents tab: predefined coder-prompt matrix, the
  Anthropic + Cursor dispatchers, and the random-sprinkle loop. If
  the `claude` CLI binary is present at startup, Anthropic fires
  use the real subprocess; otherwise direct API call with a
  `claude-cli` User-Agent. Cursor is always API-only via
  `api2.cursor.sh`.
- `app/config.py` — all `.env` knobs and their defaults
- `app/web.py` — REST + SSE API surface (`/healthz`, `/metrics`,
  `/api/status`, `/api/config`, `/api/targets/…`, `/api/fire-all`,
  `/api/scheduler/…`, `/api/events/stream`, `/api/prompt/…`,
  `/api/keys/…`)
- `app/main.py` — entry point; the `_resolve_tls_verify()` function
  at the top picks TLS mode at boot (custom bundle / system / bypass)

Deployment:

- `.env.example` — every tunable with inline notes
- `Dockerfile` — multi-stage build against `python:3.12-slim`
  (multi-arch: works on linux/amd64 and linux/arm64)
- `docker-compose.yml` — named `ai-spray-keys` volume for key
  persistence; bind-mount of `./certs/` into
  `/etc/ssl/hairspray-extra-ca:ro` for SASE CA trust
- `certs/README.md` — how to install your SASE/NGFW re-sign CA so
  TLS verification works inside an inspecting fabric (also
  accessible in-app via the **CERT HELP** button in the header)
- `systemd/hairspray.service` — systemd unit for Linux autostart
- `macos/` — launchd agent + installer for macOS autostart
  independent of Docker Desktop's own start-at-login setting

## License

MIT — see [`LICENSE`](LICENSE). You accept all risk of using this
tool. Nothing about it is warrantied.
