"""Entry point: run the scheduler loop and the Starlette HTTP app side by side."""
from __future__ import annotations

import asyncio
import logging
import random
import signal
import sys

import httpx
import structlog
import uvicorn

from .config import Config
from .registry import build_registry
from .state import AppState
from .web import create_app, run_and_publish


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    lvl = getattr(logging, level, logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=lvl)
    # Chatty httpx/httpcore internals just clutter the journal.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Silence uvicorn's per-request access log; our 'traffic' events are
    # the interesting thing.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


# ---------------------------------------------------------------------------
# TLS verification strategy
# ---------------------------------------------------------------------------

EXTRA_CA_DIR = "/etc/ssl/hairspray-extra-ca"
COMBINED_BUNDLE_PATH = "/tmp/hairspray-ca-bundle.pem"


def _resolve_tls_verify(cfg: Config, log_) -> "bool | str":
    """Pick the right argument for httpx.AsyncClient(verify=...).

    Returns one of:
      * str path to a combined CA bundle (certifi + every mounted extra)
        — used when the operator has dropped a SASE re-sign CA into
        /etc/ssl/hairspray-extra-ca/. This gives verified HTTPS *and*
        compatibility with TLS-inspecting fabrics.
      * True — if no extras are mounted and TLS_VERIFY is truthy; httpx
        will use its built-in certifi bundle.
      * False — if no extras are mounted and TLS_VERIFY is falsy;
        verification is bypassed (compat mode for fabrics we cannot
        trust a CA for). Also silences urllib3's InsecureRequestWarning
        since we've explicitly opted out.

    Always emits a single boot-log line identifying which mode won, so
    operators can diagnose cert problems by reading one grep away.
    """
    import os
    # Inline imports keep module-load cheap on code paths that don't
    # need them (e.g. in test environments where certifi might not be
    # present).
    extra_dir = os.environ.get("HAIRSPRAY_EXTRA_CA_DIR", EXTRA_CA_DIR)
    extras: list[str] = []
    if os.path.isdir(extra_dir):
        extras = sorted(
            f for f in os.listdir(extra_dir)
            if f.lower().endswith((".crt", ".pem", ".cer"))
            and os.path.isfile(os.path.join(extra_dir, f))
        )

    if extras:
        try:
            import certifi
            with open(COMBINED_BUNDLE_PATH, "w", encoding="utf-8") as out:
                # System CAs first so their trust anchors take precedence
                # when the same subject exists in both.
                with open(certifi.where(), "r", encoding="utf-8") as f:
                    out.write(f.read())
                for name in extras:
                    out.write(f"\n# --- hAIrspray extra CA: {name} ---\n")
                    with open(os.path.join(extra_dir, name),
                              "r", encoding="utf-8") as f:
                        out.write(f.read())
                    out.write("\n")
            # Also export SSL_CERT_FILE / REQUESTS_CA_BUNDLE so any
            # subprocess or transitive library (requests, urllib3,
            # aiohttp) sees the same trust anchors as httpx does.
            os.environ["SSL_CERT_FILE"] = COMBINED_BUNDLE_PATH
            os.environ["REQUESTS_CA_BUNDLE"] = COMBINED_BUNDLE_PATH
            log_.info(
                "tls_custom_ca",
                mode="custom-bundle",
                extras=extras,
                bundle_path=COMBINED_BUNDLE_PATH,
                extra_dir=extra_dir,
                reason="Extra CA cert(s) found; verification ENABLED "
                       "against combined certifi + extras bundle. "
                       "Correct for SASE/NGFW re-sign deployments.",
            )
            return COMBINED_BUNDLE_PATH
        except Exception as e:  # pragma: no cover - defensive
            log_.error(
                "tls_custom_ca_failed",
                error=f"{type(e).__name__}: {e}",
                extras=extras,
                hint="Falling back to TLS_VERIFY setting.",
            )

    if cfg.tls_verify:
        log_.info(
            "tls_system_verify",
            mode="system",
            reason="TLS_VERIFY=true and no extra CAs mounted; using "
                   "certifi's built-in Mozilla trust store.",
        )
        return True

    # Verify off: silence the noisy urllib3 warning any transitive dep
    # might reach for.
    import warnings
    try:
        from urllib3.exceptions import InsecureRequestWarning
        warnings.simplefilter("ignore", InsecureRequestWarning)
    except Exception:
        pass
    log_.warning(
        "tls_verify_disabled",
        mode="bypass",
        reason="TLS_VERIFY=false and no extra CAs mounted; certificate "
               "validation SKIPPED. The preferred fix is to drop your "
               "SASE re-sign CA into ./certs/ and rebuild.",
    )
    return False


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

async def _scheduler_loop(
    state: AppState,
    client: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    rng = random.Random()
    log.info(
        "scheduler_started",
        target_count=len(state.all_providers()),
        categories=sorted(state.config.categories),
    )

    async def _sleep_or_stop(seconds: float) -> None:
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    while not stop.is_set():
        # Block while paused via the UI.
        await state.wait_for_resume()
        if stop.is_set():
            break

        cfg = state.config
        eligible = state.eligible_providers()

        if not eligible:
            # No enabled targets in the currently-enabled categories;
            # idle briefly and re-check.
            await _sleep_or_stop(5.0)
            continue

        if rng.random() < cfg.burst_probability:
            size = rng.randint(cfg.burst_min_size, cfg.burst_max_size)
            log.info("burst_start", size=size)
            for _ in range(size):
                if stop.is_set() or not state.is_running():
                    break
                picks = state.eligible_providers()
                if not picks:
                    break
                await run_and_publish(
                    rng.choice(picks), client, state, source="burst"
                )
                gap = rng.uniform(cfg.burst_gap_min, cfg.burst_gap_max)
                await _sleep_or_stop(gap)
            log.info("burst_end")
        else:
            await run_and_publish(
                rng.choice(eligible), client, state, source="scheduler"
            )

        # Inter-iteration pacing.
        delay = rng.uniform(cfg.min_interval, cfg.max_interval)
        await _sleep_or_stop(delay)

    log.info("scheduler_stopped")


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

async def _main_async() -> int:
    cfg = Config.from_env()
    _configure_logging(cfg.log_level)

    log.info(
        "booting",
        health_port=cfg.health_port,
        enable_real_responses=cfg.enable_real_responses,
        categories=sorted(cfg.categories),
        tls_verify=cfg.tls_verify,
    )

    # Build the full target catalogue; per-category filtering happens
    # at scheduling time so the UI can toggle categories live.
    # KeyStore is constructed once and shared between the registry
    # (so MCPAuthedProbe can fetch tokens lazily) and the web app
    # (where the /api/keys endpoints write to it). One instance, no
    # cache coherence headaches.
    from .keys import KeyStore
    key_store = KeyStore()
    # Load eagerly so a corrupt keys.json is caught at boot instead of
    # surfacing on the first write. Starlette's on_startup hook is not
    # reliable here because uvicorn runs with lifespan='off'.
    await key_store.load()
    log.info("key_store_loaded", count=len((await key_store.summary())["providers"]))
    providers = build_registry(key_store=key_store)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    limits = httpx.Limits(max_connections=32, max_keepalive_connections=8)
    timeout = httpx.Timeout(cfg.http_timeout, connect=min(10.0, cfg.http_timeout))

    # Resolve the TLS verification strategy. Three modes, in priority order:
    #
    #   1. CUSTOM CA BUNDLE — if any *.crt / *.pem files are mounted into
    #      /etc/ssl/hairspray-extra-ca/ (see docker-compose.yml volume
    #      mapping), concatenate them onto certifi's Mozilla bundle and
    #      point httpx at the combined file. This is the correct mode for
    #      SASE / NGFW deployments that decrypt-and-re-sign HTTPS: the
    #      extra CA is the fabric's re-sign root, so verification passes
    #      on every inspected flow while still catching anything the
    #      fabric didn't re-sign.
    #
    #   2. FULL SYSTEM VERIFY — if no extras are mounted and TLS_VERIFY
    #      is truthy, httpx uses the stock certifi bundle. The right
    #      mode when running outside any inspecting fabric.
    #
    #   3. VERIFY OFF — if no extras are mounted and TLS_VERIFY is
    #      falsy, TLS verification is bypassed entirely. Last-resort
    #      compatibility mode; use only inside a trusted network path
    #      since MitM attempts then go unnoticed.
    #
    # The boot log always states which mode is active so "why is TLS
    # still failing" debugging is a matter of reading the first few
    # lines of `docker compose logs hairspray`.
    verify_arg = _resolve_tls_verify(cfg, log)
    if isinstance(verify_arg, str):
        tls_mode = "custom-bundle"
    elif verify_arg is True:
        tls_mode = "system"
    else:
        tls_mode = "bypass"

    async with httpx.AsyncClient(
        http2=True,
        limits=limits,
        timeout=timeout,
        follow_redirects=False,
        verify=verify_arg,
    ) as client:

        state = AppState(
            initial_config=cfg, providers=providers, tls_mode=tls_mode,
        )
        app = create_app(state, client, key_store=key_store)

        uvi_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=cfg.health_port,
            log_level="warning",
            access_log=False,
            lifespan="off",
            loop="asyncio",
            timeout_graceful_shutdown=3,
        )
        server = uvicorn.Server(uvi_config)

        async def _watch_stop() -> None:
            await stop.wait()
            server.should_exit = True

        tasks = [
            asyncio.create_task(_scheduler_loop(state, client, stop),
                                name="scheduler"),
            asyncio.create_task(server.serve(), name="http"),
            asyncio.create_task(_watch_stop(), name="shutdown-watcher"),
        ]

        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        stop.set()
        server.should_exit = True

        for t in pending:
            t.cancel()
        for t in (*done, *pending):
            try:
                await t
            except (asyncio.CancelledError, Exception) as e:
                if not isinstance(e, asyncio.CancelledError):
                    log.warning("task_exited", task=t.get_name(), error=str(e))

    log.info("shutdown_complete")
    return 0


def main() -> None:
    try:
        code = asyncio.run(_main_async())
    except KeyboardInterrupt:
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
