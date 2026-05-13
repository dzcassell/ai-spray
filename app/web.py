"""Starlette HTTP layer.

Serves:
* ``GET /``                          — single-page UI (HTML with inline CSS/JS)
* ``GET /healthz``                   — Docker HEALTHCHECK target (plain text)
* ``GET /metrics``                   — JSON snapshot, kept for scrapers
* ``GET /api/status``                — full dashboard snapshot
* ``GET /api/config``                — current live-tunable config
* ``PATCH|POST /api/config``         — apply a partial config update
* ``GET /api/targets``               — list all 58 providers with state
* ``POST /api/targets/{n}/toggle``   — flip enabled state
* ``POST /api/targets/{n}/fire``     — fire a one-shot request (async)
* ``POST /api/scheduler/pause``      — pause the scheduler loop
* ``POST /api/scheduler/resume``     — resume it
* ``GET /api/events``                — recent events (JSON array)
* ``GET /api/events/stream``         — live SSE stream of events
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from .prompt import (
    KEYED_PROVIDERS,
    PROMPT_TARGETS,
    run_prompt_target,
    targets_catalogue,
    validate_target_id,
)
from .keys import KeyStore
from . import discovery
from . import extract
from . import mcp
from . import pii
from .state import AppState

log = structlog.get_logger()

UI_DIR = Path(__file__).parent / "ui"


# ---------------------------------------------------------------------------
# Core: run a provider and publish the result through AppState.
# Used by both the scheduler loop and the manual "fire" endpoint.
# ---------------------------------------------------------------------------

async def run_and_publish(
    provider,
    client: httpx.AsyncClient,
    state: AppState,
    source: str = "scheduler",
) -> None:
    try:
        result = await provider.execute(client)
    except Exception as e:  # noqa: BLE001 — scheduler must never die
        log.exception("provider_crashed", target=provider.name, error=str(e))
        state.publish_result({
            "target": provider.name,
            "category": provider.category,
            "method": "?",
            "url": AppState._provider_url(provider),
            "status": None,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "snippet": None,
            "source": source,
        })
        return

    state.publish_result({
        "target": result.name,
        "category": result.category,
        "method": result.method,
        "url": result.url,
        "status": result.status_code,
        "ok": result.ok,
        "error": result.error,
        "snippet": result.response_snippet,
        "source": source,
    })
    log.info(
        "traffic",
        target=result.name,
        category=result.category,
        method=result.method,
        url=result.url,
        status=result.status_code,
        ok=result.ok,
        error=result.error,
        source=source,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _config_to_dict(cfg) -> dict:
    return {
        "min_interval": cfg.min_interval,
        "max_interval": cfg.max_interval,
        "burst_probability": cfg.burst_probability,
        "burst_min_size": cfg.burst_min_size,
        "burst_max_size": cfg.burst_max_size,
        "burst_gap_min": cfg.burst_gap_min,
        "burst_gap_max": cfg.burst_gap_max,
        "categories": sorted(cfg.categories),
        "enable_real_responses": cfg.enable_real_responses,
        "http_timeout": cfg.http_timeout,
        "log_level": cfg.log_level,
    }


def create_app(
    state: AppState,
    client: httpx.AsyncClient,
    key_store: KeyStore | None = None,
) -> Starlette:
    # Background tasks spawned by fire-and-forget endpoints. Holding a
    # strong reference prevents the event loop from GC'ing them mid-run,
    # and the done-callback ensures the set drains automatically.
    bg_tasks: set[asyncio.Task] = set()

    def _spawn(coro, *, name: str | None = None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        task.add_done_callback(_log_task_exception)
        return task

    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.warning(
                "background_task_failed",
                task=task.get_name(),
                error=f"{type(exc).__name__}: {exc}",
            )


    # Shared key store. If the caller provided one (so the same
    # instance is also passed to build_registry for MCPAuthedProbe),
    # use that — a single KeyStore avoids cache coherence issues.
    # Otherwise lazily construct one; lazily loads on first access;
    # file lives in the Docker volume mounted at /data (configurable
    # via AI_SPRAY_KEYS_PATH).
    if key_store is None:
        key_store = KeyStore()

    # ------------------ UI ------------------

    async def index(request: Request) -> Response:
        try:
            html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            return PlainTextResponse("UI assets missing", status_code=500)
        return HTMLResponse(html)

    # ------------------ Legacy / compat ------------------

    async def healthz(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def metrics(request: Request) -> Response:
        return JSONResponse(state.stats_snapshot())

    # ------------------ Status (everything at once) ------------------

    async def status(request: Request) -> Response:
        return JSONResponse({
            **state.stats_snapshot(),
            "config": _config_to_dict(state.config),
        })

    # ------------------ Config ------------------

    async def config_handler(request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse(_config_to_dict(state.config))

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected object"}, status_code=400)

        applied = await state.update_config(body)
        # Normalize for JSON (set → sorted list).
        return JSONResponse({
            "applied": {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in applied.items()
            },
            "config": _config_to_dict(state.config),
        })

    # ------------------ Targets ------------------

    async def list_targets(request: Request) -> Response:
        return JSONResponse(state.targets_snapshot())

    async def toggle_target(request: Request) -> Response:
        name = request.path_params["name"]
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass

        if "enabled" in body:
            new_state = bool(body["enabled"])
        else:
            new_state = not state.is_enabled(name)

        if not state.set_enabled(name, new_state):
            return JSONResponse({"error": "unknown target"}, status_code=404)
        return JSONResponse({"name": name, "enabled": state.is_enabled(name)})

    async def fire_target(request: Request) -> Response:
        name = request.path_params["name"]
        provider = state.get_provider(name)
        if provider is None:
            return JSONResponse({"error": "unknown target"}, status_code=404)
        # Fire-and-forget. Result will appear in the SSE stream + stats.
        _spawn(
            run_and_publish(provider, client, state, source="manual"),
            name=f"fire_target:{name}",
        )
        return JSONResponse({"name": name, "fired": True}, status_code=202)

    # ------------------ Fire all ------------------
    #
    # Runs one request per selected target using a bounded concurrency
    # pool (default 10). Ten parallel flights finishes a full 161-target
    # spray in ~30 s while staying well under any reasonable Cloudflare
    # WAF threshold that might briefly affect co-tenant containers on
    # the same outbound IP. Only one fire-all job runs at a time; a
    # second call while one is in progress returns 409.
    #
    # Selection rules (any combination may be supplied in the POST body):
    #   * "scope": "enabled" (default) | "all"
    #     - "enabled" honors the per-target toggles and the channel
    #        checkboxes (same set the scheduler picks from).
    #     - "all" ignores all toggles and fires every known target.
    #   * "category": "llm_api" etc. — restrict to one channel.
    #   * "names": explicit list of provider names; overrides everything
    #     else.
    #   * "concurrency": max in-flight requests (default 10, hard cap 32).
    #   * "gap_min_sec" / "gap_max_sec": minimum launch spacing between
    #     new requests starting (default 0.1 → 0.3). Keeps the initial
    #     burst from arriving as a single TCP fan-out blip.

    async def _fire_all_runner(
        targets: list,
        concurrency: int,
        gap_min: float,
        gap_max: float,
        source: str,
    ) -> None:
        import random
        rng = random.Random()
        total = len(targets)
        sem = asyncio.Semaphore(concurrency)
        launched = 0
        completed = 0

        state.begin_fire_all(
            total=total, source=source, concurrency=concurrency,
        )
        log.info(
            "fire_all_started",
            total=total, concurrency=concurrency, source=source,
        )

        async def _one(provider) -> None:
            nonlocal completed
            async with sem:
                if state.fire_all_cancel_requested():
                    return
                state.update_fire_all_current(provider.name)
                await run_and_publish(provider, client, state, source=source)
                completed += 1
                state.update_fire_all_progress(done=completed)

        tasks: list[asyncio.Task] = []
        try:
            for provider in targets:
                if state.fire_all_cancel_requested():
                    log.info(
                        "fire_all_cancelled_before_launch",
                        launched=launched, total=total,
                    )
                    break
                tasks.append(asyncio.create_task(_one(provider)))
                launched += 1
                # Small random stagger so the initial fan-out isn't a
                # single tight burst at t=0.
                if launched < total:
                    gap = rng.uniform(gap_min, gap_max)
                    try:
                        await asyncio.wait_for(
                            state.fire_all_cancel_event().wait(), timeout=gap
                        )
                    except asyncio.TimeoutError:
                        pass

            # Wait for all in-flight work to settle, even if we broke
            # out of the launch loop due to cancel.
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            state.end_fire_all()
            log.info(
                "fire_all_finished",
                completed=completed, launched=launched, total=total,
            )

    async def fire_all(request: Request) -> Response:
        if state.fire_all_running():
            return JSONResponse(
                {"error": "fire-all already in progress",
                 "progress": state.fire_all_snapshot()},
                status_code=409,
            )

        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass

        scope = str(body.get("scope", "enabled")).lower()
        category = body.get("category")
        names = body.get("names")
        concurrency = int(body.get("concurrency", 10))
        concurrency = max(1, min(32, concurrency))
        gap_min = float(body.get("gap_min_sec", 0.1))
        gap_max = float(body.get("gap_max_sec", 0.3))
        if gap_min < 0: gap_min = 0.0
        if gap_max < gap_min: gap_max = gap_min

        # Build the target list.
        if names and isinstance(names, list):
            providers = [
                p for n in names
                if (p := state.get_provider(str(n))) is not None
            ]
        elif scope == "all":
            providers = list(state.all_providers())
        else:
            providers = state.eligible_providers()

        if category:
            providers = [p for p in providers if p.category == str(category)]

        if not providers:
            return JSONResponse(
                {"error": "no targets matched selection"},
                status_code=400,
            )

        # Shuffle so repeated fire-all runs produce different orderings
        # and don't always hammer the same vendor first.
        import random as _r
        _r.shuffle(providers)

        _spawn(
            _fire_all_runner(
                providers, concurrency, gap_min, gap_max, source="fire-all",
            ),
            name="fire_all_runner",
        )
        # With concurrency=C and per-request time ≈T_req, total ≈
        # (T_req * N) / C + launch_gap * N; we don't know T_req so we
        # return a rough wall-clock floor of launch_gap * N.
        return JSONResponse(
            {
                "started": True,
                "total": len(providers),
                "concurrency": concurrency,
                "launch_gap_sec": [gap_min, gap_max],
            },
            status_code=202,
        )

    async def fire_all_status(request: Request) -> Response:
        return JSONResponse(state.fire_all_snapshot())

    async def fire_all_cancel(request: Request) -> Response:
        if not state.fire_all_running():
            return JSONResponse({"running": False})
        state.request_fire_all_cancel()
        return JSONResponse({"cancelling": True})

    # ------------------ Scheduler control ------------------

    async def pause_scheduler(request: Request) -> Response:
        state.pause()
        log.info("scheduler_paused_via_api")
        return JSONResponse({"running": state.is_running()})

    async def resume_scheduler(request: Request) -> Response:
        state.resume()
        log.info("scheduler_resumed_via_api")
        return JSONResponse({"running": state.is_running()})

    # ------------------ Events ------------------

    async def recent_events(request: Request) -> Response:
        try:
            limit = int(request.query_params.get("limit", "200"))
        except ValueError:
            limit = 200
        return JSONResponse(state.recent_events(limit=limit))

    async def event_stream(request: Request) -> Response:
        q = state.subscribe()

        async def gen():
            try:
                # Catch the client up with recent history.
                for ev in state.recent_events(limit=50):
                    yield f"data: {json.dumps(ev)}\n\n".encode("utf-8")
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(ev)}\n\n".encode("utf-8")
                    except asyncio.TimeoutError:
                        # Keep-alive comment frame so proxies don't idle-close.
                        yield b": ping\n\n"
            finally:
                state.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------ Prompt mode ------------------
    #
    # GET /api/prompt/targets — list the 13 keyless prompt-capable
    #   model targets with stable ids + display labels.
    #
    # POST /api/prompt/stream — take a JSON body:
    #     {
    #       "prompt":    "<user text>",
    #       "target_ids": ["pollinations-text::openai", ...]   # optional
    #     }
    #   Returns an SSE stream of one event per target as responses
    #   arrive. Each event payload matches PromptResult.to_dict().
    #   Max concurrency 10 (same bound used by fire-all, for the same
    #   shared-IP WAF reasons).
    #
    #   Responses are ALSO published to the normal traffic event log
    #   with source="prompt" so prompt runs show up in /api/events,
    #   the Live Log pane, and per-target stats the same as any
    #   scheduler traffic.

    async def prompt_targets(request: Request) -> Response:
        # Inject per-provider key presence so the UI can enable/disable
        # keyed-provider checkboxes accordingly. Also pull the cached
        # per-provider model catalogs (from dynamic discovery) so the
        # UI renders whatever the provider's /models endpoint returned,
        # not the hard-coded KEYED_PROVIDERS defaults.
        summary = await key_store.summary()
        presence = {
            prov: bool(v.get("present"))
            for prov, v in summary["providers"].items()
        }
        models_override = await key_store.all_cached_models()
        return JSONResponse({
            "targets": targets_catalogue(presence, models_override),
        })

    # ------------------ Keys (persistent key store) ------------------
    #
    # GET /api/keys
    #   Returns a summary: which providers have keys, with masked
    #   previews (never full values), plus the model_count discovered
    #   for each (zero if discovery hasn't run or failed — the UI
    #   falls back to hard-coded defaults).
    #
    # POST /api/keys/{provider}
    #   Body: {"key": "<raw-api-key>"}
    #   Saves the key for this provider, then immediately runs
    #   catalog discovery (GET /v1/models or equivalent). The response
    #   includes the updated summary so the UI can show the model
    #   count without a separate fetch.
    #
    # POST /api/keys/{provider}/refresh
    #   Re-runs catalog discovery for an existing key. Used by the
    #   ↻ Refresh button in the UI when the user suspects a
    #   provider's catalog has changed since the last save.
    #
    # DELETE /api/keys/{provider}
    #   Removes the key AND the cached catalog for this provider.

    def _provider_entry(provider: str) -> dict | None:
        for p in KEYED_PROVIDERS:
            if p["provider"] == provider:
                return p
        # Slice B-static: MCP keyed servers share the key store with
        # the AI providers but use a different entry shape (no shape/
        # base_url/extra/models — just url/transport/auth_*). The
        # discovery path skips them; the keys_summary endpoint
        # includes them with model_count omitted.
        for s in mcp.MCP_KEYED_SERVERS:
            if s["provider"] == provider:
                return {**s, "_mcp": True}
        return None

    async def _run_discovery_for(provider: str) -> tuple[bool, int, str | None]:
        """Fetch the provider's model catalog with the current stored
        key and cache the result.

        Returns (success, model_count, error_message).
        * success=True with model_count=N on a normal fetch.
        * success=False with error_message when discovery failed;
          the existing cache (if any) is left in place so the UI can
          continue to show whatever was last good.
        """
        entry = _provider_entry(provider)
        if entry is None:
            return False, 0, f"unknown provider: {provider}"
        # MCP keyed servers don't expose a /v1/models endpoint —
        # there's no catalog to discover, just one fixed initialize
        # endpoint. Treat refresh as a no-op so the Refresh button
        # still works (returns success, 0 models, no error).
        if entry.get("_mcp"):
            return True, 0, None
        api_key = await key_store.get(provider)
        if not api_key:
            return False, 0, "no key stored for this provider"

        extra = entry.get("extra") or {}
        base_url = extra.get("base_url")
        discovery_url = extra.get("discovery_url")
        try:
            models = await discovery.discover_models(
                client, entry["shape"], base_url, api_key, timeout=10.0,
                discovery_url=discovery_url,
            )
        except Exception as e:
            log.warning("discovery_unexpected_exception",
                        provider=provider, err=str(e))
            return False, 0, f"discovery raised: {type(e).__name__}"

        if models is None:
            # Discovery failed in a known way (non-200, parse error,
            # transport error). The discovery module already logged.
            return False, 0, ("discovery endpoint did not return a "
                              "usable model list")
        try:
            await key_store.set_models(provider, models)
        except OSError as e:
            log.error("set_models_io_fail", provider=provider, err=str(e))
            return False, 0, f"could not persist catalog: {e}"
        return True, len(models), None

    async def keys_summary(request: Request) -> Response:
        summary = await key_store.summary()
        mcp_summary = await key_store.mcp_summary()
        providers = [
            {
                "provider":    p["provider"],
                "label":       p["label"],
                "signup_url":  p["signup_url"],
                "kind":        "ai",
                "present":     bool(summary["providers"]
                                    .get(p["provider"], {})
                                    .get("present")),
                "preview":     (summary["providers"]
                                .get(p["provider"], {})
                                .get("preview")),
                "model_count": (summary["providers"]
                                .get(p["provider"], {})
                                .get("model_count", 0)),
                "fetched_at":  (summary["providers"]
                                .get(p["provider"], {})
                                .get("fetched_at")),
            }
            for p in KEYED_PROVIDERS
        ]
        # Slice B-static: append MCP keyed servers. They share the key
        # store *file* but are persisted in a parallel mcp_keys bucket
        # (different in-memory dict, no model-discovery semantics). The
        # presence/preview comes from mcp_summary(), not summary().
        for s in mcp.MCP_KEYED_SERVERS:
            mcp_entry = mcp_summary["providers"].get(s["provider"], {})
            providers.append({
                "provider":   s["provider"],
                "label":      s["label"],
                "signup_url": s["signup_url"],
                "kind":       "mcp",
                "url":        s["url"],
                "scope_hint": s["scope_hint"],
                "present":    bool(mcp_entry.get("present")),
                "preview":    mcp_entry.get("preview"),
            })
        return JSONResponse({"providers": providers, "path": summary["path"]})

    async def keys_set(request: Request) -> Response:
        provider = request.path_params["provider"]
        is_mcp = mcp.mcp_keyed_entry(provider) is not None
        if not is_mcp and _provider_entry(provider) is None:
            return JSONResponse(
                {"error": f"unknown provider: {provider}"},
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        key = body.get("key") if isinstance(body, dict) else None
        if not isinstance(key, str) or not key.strip():
            return JSONResponse(
                {"error": "key must be a non-empty string"},
                status_code=400,
            )

        # MCP servers: token-only, no model catalog to discover.
        if is_mcp:
            try:
                await key_store.mcp_set(provider, key)
            except OSError as e:
                log.error("key_store_write_failed", error=str(e))
                return JSONResponse(
                    {"error": f"could not save key: {e}"},
                    status_code=500,
                )
            log.info("mcp_key_set", provider=provider)
            summary_resp = await keys_summary(request)
            return summary_resp

        # AI providers: existing flow with discovery.
        try:
            await key_store.set(provider, key)
        except OSError as e:
            log.error("key_store_write_failed", error=str(e))
            return JSONResponse(
                {"error": f"could not save key: {e}"},
                status_code=500,
            )

        # Fire discovery synchronously. A 10s timeout is baked into
        # discover_models, so worst case the POST returns in ~10s.
        # If discovery fails, the key is still saved — the UI will
        # fall back to the hard-coded defaults for this provider.
        discovered_ok, count, err = await _run_discovery_for(provider)
        log.info("keys_set_discovery",
                 provider=provider, ok=discovered_ok,
                 model_count=count, err=err)

        # Return the updated summary plus a discovery outcome hint so
        # the UI can show a toast if needed.
        summary_resp = await keys_summary(request)
        summary_body = json.loads(summary_resp.body)
        summary_body["discovery"] = {
            "provider":    provider,
            "ok":          discovered_ok,
            "model_count": count,
            "error":       err,
        }
        return JSONResponse(summary_body)

    async def keys_refresh(request: Request) -> Response:
        """Re-run catalog discovery for an existing key. The ↻
        Refresh button in the UI hits this."""
        provider = request.path_params["provider"]
        if _provider_entry(provider) is None:
            return JSONResponse(
                {"error": f"unknown provider: {provider}"},
                status_code=400,
            )
        if not await key_store.has(provider):
            return JSONResponse(
                {"error": f"no key stored for {provider}"},
                status_code=400,
            )

        discovered_ok, count, err = await _run_discovery_for(provider)
        log.info("keys_refresh_discovery",
                 provider=provider, ok=discovered_ok,
                 model_count=count, err=err)

        summary_resp = await keys_summary(request)
        summary_body = json.loads(summary_resp.body)
        summary_body["discovery"] = {
            "provider":    provider,
            "ok":          discovered_ok,
            "model_count": count,
            "error":       err,
        }
        return JSONResponse(summary_body)

    async def keys_delete(request: Request) -> Response:
        provider = request.path_params["provider"]
        is_mcp = mcp.mcp_keyed_entry(provider) is not None
        if not is_mcp and _provider_entry(provider) is None:
            return JSONResponse(
                {"error": f"unknown provider: {provider}"},
                status_code=400,
            )
        try:
            if is_mcp:
                await key_store.mcp_delete(provider)
            else:
                await key_store.delete(provider)
        except OSError as e:
            log.error("key_store_write_failed", error=str(e))
            return JSONResponse(
                {"error": f"could not delete key: {e}"},
                status_code=500,
            )
        return await keys_summary(request)

    async def prompt_extract(request: Request) -> Response:
        """Accept a multipart upload, extract text, return it inline.

        The client uses this when the user attaches a file in the
        Prompt panel: upload here, get back the extracted text, then
        include the text in the next /api/prompt/stream POST as the
        attachment field. No server-side cache or TTL — the extracted
        text is held client-side until the user fires the prompt.

        Caps:
        * 10 MB upload (enforced both browser-side and here).
        * Extracted text capped at extract.MAX_EXTRACT_CHARS (~20K)
          inside the extractor itself.
        """
        # Starlette uses python-multipart for multipart/form-data
        # parsing; the import is implicit. Reading the form pulls the
        # file into memory — fine for our 10 MB cap, would be
        # insufficient for anything genuinely large but that's exactly
        # the scope we ruled out.
        try:
            form = await request.form()
        except Exception as e:  # noqa: BLE001 — multipart raises a zoo
            log.warning("prompt_extract_form_parse_failed", error=str(e))
            return JSONResponse(
                {"error": f"could not parse form upload: {e}"},
                status_code=400,
            )

        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse(
                {"error": "missing 'file' field in form upload"},
                status_code=400,
            )

        filename = getattr(upload, "filename", "") or ""
        if not filename:
            return JSONResponse(
                {"error": "uploaded file has no filename"},
                status_code=400,
            )

        # Read with a hard 10 MB ceiling. Reading one extra byte and
        # checking is the cleanest way to refuse oversize without
        # buffering the whole stream first.
        MAX_UPLOAD = 10 * 1024 * 1024
        content = await upload.read(MAX_UPLOAD + 1)
        if len(content) > MAX_UPLOAD:
            return JSONResponse(
                {"error": (f"file too large: {len(content):,} bytes "
                           f"exceeds {MAX_UPLOAD:,} byte cap")},
                status_code=413,
            )

        try:
            result = extract.extract(filename, content)
        except ValueError as e:
            # Unsupported extension — clean 422 with the message.
            return JSONResponse({"error": str(e)}, status_code=422)
        except Exception as e:  # noqa: BLE001 — pypdf/docx/openpyxl
            # Library blew up (corrupt file, encrypted, etc.). Tell
            # the operator what happened — they can pick a different
            # file. Don't 500 because the upload itself was valid.
            log.warning(
                "prompt_extract_library_failed",
                filename=filename, error=str(e),
            )
            return JSONResponse(
                {"error": f"could not extract from {filename}: "
                          f"{type(e).__name__}: {e}"},
                status_code=422,
            )

        log.info(
            "prompt_extract_ok",
            filename=filename,
            source_kind=result.source_kind,
            char_count=result.char_count,
            truncated=result.truncated,
        )

        return JSONResponse({
            "filename":    filename,
            "source_kind": result.source_kind,
            "text":        result.text,
            "char_count":  result.char_count,
            "truncated":   result.truncated,
            "summary":     result.summary,
            "size_bytes":  len(content),
        })

    async def prompt_stream(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected object"}, status_code=400)

        prompt_text = body.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return JSONResponse(
                {"error": "prompt must be a non-empty string"},
                status_code=400,
            )
        if len(prompt_text) > 4000:
            return JSONResponse(
                {"error": "prompt too long (max 4000 chars)"},
                status_code=400,
            )

        # Optional attachment block (file-upload feature). The client
        # extracts the file via /api/prompt/extract, then includes the
        # extracted text here as a labeled block. We prepend it to the
        # prompt with a clear "Attached file" header so the model
        # treats the document as context. Cap on the attachment side
        # (server-enforced in /extract) is ~20K chars; combined with
        # the 4K prompt limit we stay well under any provider's
        # context window.
        attachment = body.get("attachment")
        attachment_meta: dict[str, Any] | None = None
        if attachment is not None:
            if not isinstance(attachment, dict):
                return JSONResponse(
                    {"error": "attachment must be an object"},
                    status_code=400,
                )
            att_text = attachment.get("text")
            att_name = attachment.get("filename")
            att_kind = attachment.get("source_kind", "text")
            att_summary = attachment.get("summary", "")
            if not isinstance(att_text, str) or not att_text.strip():
                return JSONResponse(
                    {"error": "attachment.text must be a non-empty string"},
                    status_code=400,
                )
            if not isinstance(att_name, str) or not att_name.strip():
                return JSONResponse(
                    {"error": "attachment.filename must be a non-empty string"},
                    status_code=400,
                )
            # Hard ceiling — defensive even though /extract caps to
            # ~20K. Anyone hand-crafting a request can't blow past it.
            if len(att_text) > 25_000:
                return JSONResponse(
                    {"error": "attachment.text exceeds 25,000 chars"},
                    status_code=400,
                )
            attachment_meta = {
                "filename":    att_name,
                "source_kind": att_kind,
                "summary":     att_summary,
            }
            prompt_text = (
                f"Attached file: {att_name} ({att_summary})\n\n"
                f"```\n{att_text}\n```\n\n"
                f"User question:\n{prompt_text}"
            )

        requested_ids = body.get("target_ids")
        if requested_ids is None:
            # Default to every keyless target.
            selected = list(PROMPT_TARGETS)
        else:
            if not isinstance(requested_ids, list):
                return JSONResponse(
                    {"error": "target_ids must be a list"},
                    status_code=400,
                )
            selected = []
            for tid in requested_ids:
                entry = validate_target_id(str(tid))
                if entry is None:
                    return JSONResponse(
                        {"error": f"unknown target_id: {tid}"},
                        status_code=400,
                    )
                selected.append(entry)
            if not selected:
                return JSONResponse(
                    {"error": "no targets selected"},
                    status_code=400,
                )

        log.info(
            "prompt_run_started",
            prompt_chars=len(prompt_text),
            target_count=len(selected),
            attachment=(attachment_meta["filename"]
                        if attachment_meta else None),
            attachment_kind=(attachment_meta["source_kind"]
                             if attachment_meta else None),
        )

        out_q: asyncio.Queue = asyncio.Queue()

        async def _one(entry: dict[str, str]) -> None:
            # For keyed providers, fetch the stored API key. Missing keys
            # are handled gracefully by run_prompt_target (returns a
            # clear error PromptResult). No key lookup for keyless.
            api_key: str | None = None
            if entry.get("keyed"):
                api_key = await key_store.get(entry["provider"])

            result = await run_prompt_target(
                client, prompt_text, entry["provider"], entry["model"],
                api_key=api_key,
            )
            # Also publish into the main event log so prompt runs
            # surface in the same place as scheduled traffic.
            state.publish_result({
                "target": result.label,
                "category": "real_response",
                "method": "POST" if result.provider == "duckduckgo"
                          else "GET",
                "url": result.url or "",
                "status": result.status,
                "ok": result.ok,
                "error": (result.body if result.kind == "error"
                          else None),
                "snippet": (
                    (result.body[:160] + "…")
                    if result.kind == "text" and result.body
                       and len(result.body) > 160
                    else (result.body if result.kind == "text"
                          else (f"image {result.content_type or ''}"
                                if result.kind == "image" else None))
                ),
                "source": "prompt",
            })
            await out_q.put(result.to_dict())

        # Group requests by upstream host. Pollinations and DDG both
        # aggressively rate-limit concurrent requests from a single IP
        # (text.pollinations.ai was returning 429s on 6-way fan-outs),
        # so we serialize within each host and only parallelize across
        # hosts. This costs a handful of seconds but produces real
        # responses instead of a wall of rate-limit errors.
        #
        # Keyed providers use the 'host' field from KEYED_PROVIDERS so
        # e.g. all Groq models go through one queue, all Mistral models
        # through another, etc.
        _keyed_host_lookup = {p["provider"]: p["host"] for p in KEYED_PROVIDERS}

        def _host_key(entry: dict[str, str]) -> str:
            p = entry["provider"]
            if p == "pollinations-text":  return "text.pollinations.ai"
            if p == "pollinations-image": return "image.pollinations.ai"
            if p == "duckduckgo":         return "duckduckgo.com"
            if p in _keyed_host_lookup:   return _keyed_host_lookup[p]
            return p  # fallback

        by_host: dict[str, list[dict[str, str]]] = {}
        for e in selected:
            by_host.setdefault(_host_key(e), []).append(e)

        async def _host_queue(entries: list[dict[str, str]]) -> None:
            # Small jitter between requests to the same host helps us
            # stay under sliding-window rate limits.
            import random as _r
            for i, e in enumerate(entries):
                await _one(e)
                if i < len(entries) - 1:
                    await asyncio.sleep(_r.uniform(0.4, 0.9))

        # Hoisted so gen() can cancel inner host_tasks on client
        # disconnect — otherwise cancelling orch leaves the gather()
        # to complete naturally and live HTTP requests outlive the
        # response.
        host_tasks: list[asyncio.Task] = []

        async def _orchestrate() -> None:
            host_tasks.extend(
                asyncio.create_task(_host_queue(entries))
                for entries in by_host.values()
            )
            try:
                await asyncio.gather(*host_tasks, return_exceptions=True)
            finally:
                await out_q.put(None)  # sentinel — end of stream

        async def gen():
            # Emit a header event so the client knows how many cards to
            # reserve and the total count to drive the progress bar.
            header = {
                "kind": "start",
                "total": len(selected),
                "target_ids": [
                    f"{e['provider']}::{e['model']}" for e in selected
                ],
            }
            yield f"data: {json.dumps(header)}\n\n".encode("utf-8")

            orch = asyncio.create_task(_orchestrate())
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(
                            out_q.get(), timeout=15.0,
                        )
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
                        continue
                    if item is None:
                        break
                    yield f"data: {json.dumps(item)}\n\n".encode("utf-8")
                # Final end event.
                yield b"data: {\"kind\":\"end\"}\n\n"
            finally:
                for t in host_tasks:
                    t.cancel()
                orch.cancel()
                try:
                    await asyncio.gather(*host_tasks, orch, return_exceptions=True)
                except Exception:
                    pass

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------ Profile Tests (DLP) ------------------

    # GET /api/profile-tests/catalog — categories + locales + prompt
    # types, used by the UI to build the tab.
    async def profile_tests_catalog(request: Request) -> Response:
        return JSONResponse({
            "categories": pii.CATEGORIES,
            "locales": pii.LOCALES,
            "prompt_types": pii.PROMPT_TYPES,
            # Sample one value per category at each locale so the UI
            # can show a preview of what kind of PII the user is about
            # to send. Uses seed=0 so previews are stable — actual
            # fires use fresh non-seeded values.
            "previews": {
                cat: {
                    loc: pii.generate(cat, loc, seed=0)
                    for loc in pii.LOCALES
                }
                for cat in pii.CATEGORIES
            },
        })

    # POST /api/profile-tests/fire — SSE stream of DLP test results.
    # Body: {
    #   target_id: "provider::model",   # one model, like Prompt & Fire
    #   categories: [...],              # subset of pii.CATEGORIES
    #   prompt_types: [...],            # subset of pii.PROMPT_TYPES
    #   locale: "US" | "UK" | "EU"      # default "US"
    # }
    # Emits:
    #   {kind:"start", total:N}
    #   {kind:"result", category, prompt_type, locale,
    #                   sent_value, prompt, response, diff, ok,
    #                   latency_ms, error?}
    #   {kind:"end"}
    async def profile_tests_fire(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected object"}, status_code=400)

        target_id = body.get("target_id")
        if not isinstance(target_id, str):
            return JSONResponse(
                {"error": "target_id must be a string"}, status_code=400,
            )
        entry = validate_target_id(target_id)
        if entry is None:
            return JSONResponse(
                {"error": f"unknown target_id: {target_id}"}, status_code=400,
            )

        categories = body.get("categories") or list(pii.CATEGORIES)
        if not isinstance(categories, list) or not all(
            isinstance(c, str) for c in categories
        ):
            return JSONResponse(
                {"error": "categories must be a list of strings"},
                status_code=400,
            )
        for c in categories:
            if c not in pii.CATEGORIES:
                return JSONResponse(
                    {"error": f"unknown category: {c}"}, status_code=400,
                )

        prompt_types = body.get("prompt_types") or list(pii.PROMPT_TYPES)
        if not isinstance(prompt_types, list) or not all(
            isinstance(p, str) for p in prompt_types
        ):
            return JSONResponse(
                {"error": "prompt_types must be a list of strings"},
                status_code=400,
            )
        for p in prompt_types:
            if p not in pii.PROMPT_TYPES:
                return JSONResponse(
                    {"error": f"unknown prompt_type: {p}"}, status_code=400,
                )

        locale = body.get("locale", "US")
        if locale not in pii.LOCALES:
            return JSONResponse(
                {"error": f"unknown locale: {locale}"}, status_code=400,
            )

        # Slice C: payload_shape selects how synthetic PII is wrapped
        # for the outbound request. "chat" (default) sends a regular
        # chat-completion prompt — what every existing Profile Test
        # run does. "mcp" wraps the same PII inside an MCP tools/call
        # envelope so SASE DLP engines that parse MCP get exercised.
        # The actual destination is still the user-selected AI model,
        # so this tests *payload-shape* DLP coverage rather than full
        # MCP-server-aware DLP. (For testing against a real MCP
        # server, the future B-oauth slice will add that path.)
        payload_shape = body.get("payload_shape", "chat")
        if payload_shape not in ("chat", "mcp"):
            return JSONResponse(
                {"error": "payload_shape must be 'chat' or 'mcp'"},
                status_code=400,
            )

        # Build the full test matrix up front.
        matrix: list[tuple[str, str, dict[str, Any], str]] = []
        for cat in categories:
            for ptype in prompt_types:
                generated = pii.generate(cat, locale)
                prompt_text = pii.build_prompt(generated, ptype)
                if payload_shape == "mcp":
                    # Wrap the generated PII inside an MCP tools/call
                    # envelope. The model receives the JSON-RPC body
                    # as raw text — most chat completion APIs will
                    # echo or summarize it; what we care about is
                    # that DLP en route saw an MCP-shaped payload.
                    import json as _json
                    mcp_envelope = mcp.wrap_pii_as_mcp_tool_call(
                        pii_value=generated["value"],
                        pii_label=generated["label"],
                        prompt_text=prompt_text,
                    )
                    prompt_text = (
                        "Below is an MCP JSON-RPC tool call. Please "
                        "process the request as instructed:\n\n"
                        + _json.dumps(mcp_envelope, indent=2)
                    )
                matrix.append((cat, ptype, generated, prompt_text))

        log.info(
            "profile_test_run_started",
            target=target_id,
            total=len(matrix),
            locale=locale,
            categories=len(categories),
            prompt_types=len(prompt_types),
        )

        # Fetch API key once for the chosen target (keyed providers).
        api_key: str | None = None
        if entry.get("keyed"):
            api_key = await key_store.get(entry["provider"])
            # We don't hard-fail on missing key here — run_prompt_target
            # will surface a clear "no key" error per-fire which is
            # actually useful in the UI (user sees exactly which rows
            # couldn't run instead of a wall of nothing).

        async def gen():
            start = {"kind": "start", "total": len(matrix), "locale": locale}
            yield f"data: {json.dumps(start)}\n\n".encode("utf-8")

            import random as _r
            for i, (cat, ptype, generated, prompt_text) in enumerate(matrix):
                if await request.is_disconnected():
                    break

                result = await run_prompt_target(
                    client, prompt_text,
                    entry["provider"], entry["model"],
                    api_key=api_key,
                )

                # For image responses, the body is a URL not a text
                # string, so DLP diff doesn't meaningfully apply.
                response_text: str | None = None
                if result.kind == "text":
                    response_text = result.body
                elif result.kind == "image":
                    response_text = f"[image: {result.url or ''}]"

                diff = pii.dlp_diff(generated["value"], response_text)

                # Publish an abbreviated event into the main log so
                # profile tests appear in the Monitor tab alongside
                # scheduled traffic.
                state.publish_result({
                    "target": f"ProfileTest · {generated['label']} · "
                              f"{ptype} · {result.label}",
                    "category": "real_response",
                    "method": "POST",
                    "url": result.url or "",
                    "status": result.status,
                    "ok": result.ok,
                    "error": (result.body if result.kind == "error"
                              else None),
                    "snippet": f"[DLP diff: {diff}]",
                    "source": "profile-test",
                })

                event = {
                    "kind": "result",
                    "index": i,
                    "category": cat,
                    "prompt_type": ptype,
                    "locale": locale,
                    "label": generated["label"],
                    "sent_value": generated["value"],
                    "prompt": prompt_text,
                    "response": response_text,
                    "diff": diff,
                    "ok": result.ok,
                    "status": result.status,
                    "latency_ms": result.latency_ms,
                    "error": (result.body if result.kind == "error"
                              else None),
                }
                yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

                # Jitter between fires to the same model to stay under
                # per-minute rate limits. Slightly more aggressive than
                # the prompt-stream host-serialization since all these
                # requests hit the same endpoint on the same provider.
                if i < len(matrix) - 1:
                    await asyncio.sleep(_r.uniform(0.5, 1.1))

            yield b"data: {\"kind\":\"end\"}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------ Agents (random-sprinkle coder prompts) ----------

    # GET /api/agents/catalog — what prompts exist; UI populates the
    # checkbox matrix from this. Static (no per-user state).
    async def agents_catalog(request: Request) -> Response:
        from . import agents
        return JSONResponse({
            "prompts":   agents.PROMPTS,
            "providers": list(agents.PROVIDERS),
            "genres":    list(agents.PROMPT_GENRES),
            "claude_cli_available": agents.claude_cli_available(),
        })

    # GET /api/agents/status — current loop state (running flag,
    # total fired, recent fire history). UI polls this every few
    # seconds so it doesn't need a separate SSE channel.
    async def agents_status(request: Request) -> Response:
        return JSONResponse(state.agent_loop.status())

    # POST /api/agents/start — start the random-sprinkle loop.
    # Body: {min_gap_sec?, max_gap_sec?, prompts?, providers?}
    # All four optional; missing values use the existing AgentLoopState
    # defaults / current values. If the loop is already running, this
    # updates the gap range and enabled sets in-place rather than
    # restarting.
    async def agents_start(request: Request) -> Response:
        from . import agents
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}

        loop_state = state.agent_loop

        # Validate + apply optional knobs.
        min_gap = body.get("min_gap_sec")
        max_gap = body.get("max_gap_sec")
        if min_gap is not None:
            if not isinstance(min_gap, int) or min_gap < 5:
                return JSONResponse(
                    {"error": "min_gap_sec must be an int >= 5"},
                    status_code=400,
                )
            loop_state.min_gap_sec = min_gap
        if max_gap is not None:
            if not isinstance(max_gap, int) or max_gap < loop_state.min_gap_sec:
                return JSONResponse(
                    {"error": (f"max_gap_sec must be an int >= "
                               f"min_gap_sec ({loop_state.min_gap_sec})")},
                    status_code=400,
                )
            loop_state.max_gap_sec = max_gap

        prompts = body.get("prompts")
        if prompts is not None:
            if (not isinstance(prompts, list)
                    or not all(isinstance(p, str) for p in prompts)):
                return JSONResponse(
                    {"error": "prompts must be a list of slug strings"},
                    status_code=400,
                )
            valid = set(agents.PROMPT_BY_SLUG.keys())
            unknown = [p for p in prompts if p not in valid]
            if unknown:
                return JSONResponse(
                    {"error": f"unknown prompt slug(s): {unknown}"},
                    status_code=400,
                )
            loop_state.enabled_prompts = set(prompts)

        providers = body.get("providers")
        if providers is not None:
            if (not isinstance(providers, list)
                    or not all(isinstance(p, str) for p in providers)):
                return JSONResponse(
                    {"error": "providers must be a list of strings"},
                    status_code=400,
                )
            valid = set(agents.PROVIDERS)
            unknown = [p for p in providers if p not in valid]
            if unknown:
                return JSONResponse(
                    {"error": f"unknown provider(s): {unknown}; "
                              f"valid: {sorted(valid)}"},
                    status_code=400,
                )
            loop_state.enabled_providers = set(providers)

        # If already running, the in-place updates above are enough —
        # the loop reads enabled_* on every iteration. Don't kick off
        # a second task.
        if loop_state.running and loop_state.task and not loop_state.task.done():
            return JSONResponse({
                "ok": True,
                "already_running": True,
                "status": loop_state.status(),
            })

        # Spin up the loop. The task captures `state.agent_loop`,
        # `client`, and the key store (for per-fire key lookup).
        async def _runner() -> None:
            try:
                await agents.run_loop(loop_state, client, key_store)
            except asyncio.CancelledError:
                # Expected on stop — let it propagate so the task
                # registers as cancelled.
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("agent_loop_crashed", error=str(e))

        loop_state.task = asyncio.create_task(_runner())
        return JSONResponse({
            "ok": True,
            "already_running": False,
            "status": loop_state.status(),
        })

    # POST /api/agents/stop — cancel the current loop. Idempotent —
    # safe to call when the loop isn't running.
    async def agents_stop(request: Request) -> Response:
        loop_state = state.agent_loop
        if loop_state.task is not None and not loop_state.task.done():
            loop_state.task.cancel()
            # Wait briefly for the cancel to land so the next status
            # call shows running=false. Don't wait forever — if the
            # task is wedged, return anyway and let the operator try
            # again.
            try:
                await asyncio.wait_for(loop_state.task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        # Defensive: clear the running flag in case the task was
        # already done but not noticed.
        loop_state.running = False
        loop_state.task = None
        return JSONResponse({"ok": True, "status": loop_state.status()})

    routes = [
        Route("/", index),
        Route("/healthz", healthz),
        Route("/metrics", metrics),

        Route("/api/status", status),
        Route("/api/config", config_handler, methods=["GET", "PATCH", "POST"]),
        Route("/api/targets", list_targets),
        Route("/api/targets/{name:path}/toggle", toggle_target, methods=["POST"]),
        Route("/api/targets/{name:path}/fire", fire_target, methods=["POST"]),
        Route("/api/fire-all", fire_all, methods=["POST"]),
        Route("/api/fire-all/status", fire_all_status),
        Route("/api/fire-all/cancel", fire_all_cancel, methods=["POST"]),
        Route("/api/scheduler/pause", pause_scheduler, methods=["POST"]),
        Route("/api/scheduler/resume", resume_scheduler, methods=["POST"]),
        Route("/api/events", recent_events),
        Route("/api/events/stream", event_stream),
        Route("/api/prompt/targets", prompt_targets),
        Route("/api/prompt/extract", prompt_extract, methods=["POST"]),
        Route("/api/prompt/stream", prompt_stream, methods=["POST"]),
        Route("/api/profile-tests/catalog", profile_tests_catalog),
        Route("/api/profile-tests/fire", profile_tests_fire, methods=["POST"]),
        Route("/api/keys", keys_summary),
        Route("/api/keys/{provider}", keys_set, methods=["POST"]),
        Route("/api/keys/{provider}", keys_delete, methods=["DELETE"]),
        Route("/api/keys/{provider}/refresh", keys_refresh, methods=["POST"]),

        Route("/api/agents/catalog", agents_catalog),
        Route("/api/agents/status",  agents_status),
        Route("/api/agents/start",   agents_start, methods=["POST"]),
        Route("/api/agents/stop",    agents_stop,  methods=["POST"]),
    ]

    return Starlette(debug=False, routes=routes)
