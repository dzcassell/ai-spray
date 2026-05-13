"""Shared mutable run-state.

The scheduler and the HTTP API both hold a reference to a single
``AppState`` instance. It owns:

* the current ``Config`` (mutable via ``update_config``)
* per-target enabled flags
* a paused/running event for the scheduler loop
* aggregate counters (for ``/metrics`` and the dashboard)
* a ring buffer of recent events and a fan-out to SSE subscribers
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .config import VALID_CATEGORIES, Config

if TYPE_CHECKING:
    from .providers import Provider


# Fields that can be changed at runtime via PATCH /api/config.
# Anything not in this set requires a container restart.
LIVE_TUNABLE_FIELDS: frozenset[str] = frozenset({
    "min_interval", "max_interval",
    "burst_probability",
    "burst_min_size", "burst_max_size",
    "burst_gap_min", "burst_gap_max",
    "categories",
    "enable_real_responses",
})


class AppState:
    def __init__(
        self,
        initial_config: Config,
        providers: list["Provider"],
    ) -> None:
        self._config = initial_config
        # Dict preserves insertion order so the UI table stays stable.
        self._providers: dict[str, "Provider"] = {p.name: p for p in providers}
        self._enabled: dict[str, bool] = {p.name: True for p in providers}

        self._running = asyncio.Event()
        self._running.set()

        # Stats
        self.started_at: float = time.time()
        self.last_tick_at: float | None = None
        self.total_requests: int = 0
        self.total_ok: int = 0
        self.total_errors: int = 0
        self.per_category: dict[str, int] = {}
        self.per_target_count: dict[str, int] = {}
        self.per_target_last_status: dict[str, Any] = {}

        # Event fan-out
        self._buffer: deque = deque(maxlen=1000)
        self._subs: set[asyncio.Queue] = set()

        self._cfg_lock = asyncio.Lock()

        # Fire-all tracking
        self._fa_running: bool = False
        self._fa_source: str = ""
        self._fa_total: int = 0
        self._fa_done: int = 0
        self._fa_current: str | None = None
        self._fa_concurrency: int = 1
        self._fa_started_at: float | None = None
        self._fa_cancel: asyncio.Event = asyncio.Event()

        # Agent random-sprinkle loop state. Lives on AppState (not a
        # local in the endpoint) so the loop survives page reloads.
        # Late-imported to avoid a top-level circular dep with agents.py
        # which imports nothing from state.py but is imported by web.py
        # which imports state.
        from .agents import AgentLoopState
        self.agent_loop: AgentLoopState = AgentLoopState()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    async def update_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial config update. Returns the set of changes
        actually applied (invalid fields / values are silently dropped)."""
        applied: dict[str, Any] = {}

        for key, value in changes.items():
            if key not in LIVE_TUNABLE_FIELDS:
                continue

            if key == "categories":
                if not isinstance(value, (list, tuple, set)):
                    continue
                cats = {str(c).strip() for c in value}
                cats &= VALID_CATEGORIES
                applied[key] = cats

            elif key == "burst_probability":
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= fv <= 1.0:
                    applied[key] = fv

            elif key in ("burst_min_size", "burst_max_size"):
                try:
                    iv = int(value)
                except (TypeError, ValueError):
                    continue
                if iv >= 1:
                    applied[key] = iv

            elif key in ("min_interval", "max_interval",
                         "burst_gap_min", "burst_gap_max"):
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if fv >= 0:
                    applied[key] = fv

            elif key == "enable_real_responses":
                applied[key] = bool(value)

        # Coerce sanity: ensure min <= max for the two ranges.
        new_min = applied.get("min_interval", self._config.min_interval)
        new_max = applied.get("max_interval", self._config.max_interval)
        if new_min > new_max:
            applied["min_interval"], applied["max_interval"] = new_max, new_min

        new_bmin = applied.get("burst_min_size", self._config.burst_min_size)
        new_bmax = applied.get("burst_max_size", self._config.burst_max_size)
        if new_bmin > new_bmax:
            applied["burst_min_size"], applied["burst_max_size"] = new_bmax, new_bmin

        if applied:
            async with self._cfg_lock:
                self._config = replace(self._config, **applied)

        return applied

    # ------------------------------------------------------------------
    # Scheduler pause / resume
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._running.is_set()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    async def wait_for_resume(self) -> None:
        await self._running.wait()

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def all_providers(self) -> list["Provider"]:
        return list(self._providers.values())

    def get_provider(self, name: str) -> "Provider | None":
        return self._providers.get(name)

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, True)

    def set_enabled(self, name: str, enabled: bool) -> bool:
        if name not in self._providers:
            return False
        self._enabled[name] = bool(enabled)
        return True

    def eligible_providers(self) -> list["Provider"]:
        """Providers that (a) are individually enabled AND
        (b) belong to a currently-enabled category."""
        cats = self._config.categories
        return [
            p for p in self._providers.values()
            if self._enabled.get(p.name, True) and p.category in cats
        ]

    @staticmethod
    def _provider_url(p: "Provider") -> str:
        return (
            getattr(p, "url", None)
            or getattr(p, "CHAT_URL", None)
            or ""
        )

    @staticmethod
    def _provider_method(p: "Provider") -> str:
        return getattr(p, "method", "GET")

    def targets_snapshot(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "category": p.category,
                "enabled": self._enabled.get(p.name, True),
                "count": self.per_target_count.get(p.name, 0),
                "last_status": self.per_target_last_status.get(p.name),
                "url": self._provider_url(p),
                "method": self._provider_method(p),
            }
            for p in self._providers.values()
        ]

    # ------------------------------------------------------------------
    # Events: fan-out + ring buffer
    # ------------------------------------------------------------------

    def publish_result(self, event: dict) -> None:
        """Record one provider result and broadcast to SSE subscribers."""
        self.last_tick_at = time.time()
        self.total_requests += 1
        if event.get("ok"):
            self.total_ok += 1
        else:
            self.total_errors += 1

        cat = event.get("category", "unknown")
        name = event.get("target", "unknown")
        self.per_category[cat] = self.per_category.get(cat, 0) + 1
        # Prompt mode and profile-test mode publish compound labels
        # (e.g. "ProfileTest · CC# · echo · Pollinations-Text · openai")
        # that produce a new per_target_count key on every run. Bucket
        # anything that isn't a registered provider into "<other>" so
        # the counters dict stays bounded over long-running sessions.
        bucket = name if name in self._providers else "<other>"
        self.per_target_count[bucket] = self.per_target_count.get(bucket, 0) + 1
        # last_status is per-registered-provider; compound labels from
        # prompt/profile-test mode would just bloat this dict without
        # ever being read back via targets_snapshot.
        if name in self._providers:
            self.per_target_last_status[name] = event.get("status")

        stamped = {"ts": self.last_tick_at, **event}
        self._buffer.append(stamped)

        # Snapshot the subscriber set before iterating: subscribe() and
        # unsubscribe() are called from other coroutines and mutating
        # _subs mid-iteration would raise RuntimeError.
        dead: list[asyncio.Queue] = []
        for q in list(self._subs):
            try:
                q.put_nowait(stamped)
            except asyncio.QueueFull:
                # Slow consumer — drop silently; they'll catch up next event.
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            self._subs.discard(q)

    def recent_events(self, limit: int = 200) -> list[dict]:
        if limit <= 0:
            return []
        buf = list(self._buffer)
        return buf[-limit:] if len(buf) > limit else buf

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    # ------------------------------------------------------------------
    # Fire-all coordination
    # ------------------------------------------------------------------

    def fire_all_running(self) -> bool:
        return self._fa_running

    def fire_all_cancel_event(self) -> asyncio.Event:
        return self._fa_cancel

    def fire_all_cancel_requested(self) -> bool:
        return self._fa_cancel.is_set()

    def request_fire_all_cancel(self) -> None:
        self._fa_cancel.set()

    def begin_fire_all(
        self, total: int, source: str, concurrency: int = 1,
    ) -> None:
        self._fa_running = True
        self._fa_source = source
        self._fa_total = total
        self._fa_done = 0
        self._fa_current = None
        self._fa_concurrency = concurrency
        self._fa_started_at = time.time()
        self._fa_cancel = asyncio.Event()

    def update_fire_all_progress(self, done: int) -> None:
        self._fa_done = done

    def update_fire_all_current(self, current: str | None) -> None:
        # With concurrency>1 this is just "most recently launched".
        self._fa_current = current

    def end_fire_all(self) -> None:
        self._fa_running = False
        self._fa_current = None

    def fire_all_snapshot(self) -> dict:
        if not self._fa_running and self._fa_started_at is None:
            return {"running": False}
        elapsed = (
            time.time() - self._fa_started_at
            if self._fa_started_at is not None else 0.0
        )
        return {
            "running": self._fa_running,
            "source": self._fa_source,
            "total": self._fa_total,
            "done": self._fa_done,
            "current": self._fa_current,
            "concurrency": getattr(self, "_fa_concurrency", 1),
            "elapsed_seconds": round(elapsed, 1),
            "cancelling": self._fa_cancel.is_set(),
        }

    # ------------------------------------------------------------------
    # Snapshot for /metrics and /api/status
    # ------------------------------------------------------------------

    def stats_snapshot(self) -> dict:
        now = time.time()
        return {
            "uptime_seconds": round(now - self.started_at, 1),
            "last_tick_seconds_ago": (
                round(now - self.last_tick_at, 1)
                if self.last_tick_at is not None else None
            ),
            "scheduler_running": self.is_running(),
            "total_requests": self.total_requests,
            "total_ok": self.total_ok,
            "total_errors": self.total_errors,
            "per_category": dict(self.per_category),
            "per_target": dict(
                sorted(self.per_target_count.items(), key=lambda kv: -kv[1])
            ),
            "total_targets": len(self._providers),
            "enabled_targets": sum(
                1 for flag in self._enabled.values() if flag
            ),
            "subscribers": len(self._subs),
            "fire_all": self.fire_all_snapshot(),
        }
