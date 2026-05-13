"""Persistent API key store.

Stores API keys for keyed prompt providers at rest in a single JSON
file, mounted from a Docker volume so values survive container
restarts and image rebuilds.

## Scope and caveats

This is an experimental lab tool. The store is plaintext JSON with
file permissions 0600 — that keeps it away from other users inside
the container, but anyone with Docker socket access on the host
(i.e., anyone in the `docker` group) can read the volume. Do not
put keys here that would cost real money if leaked.

## File layout

::

    {
      "version": 1,
      "keys": {
        "google":     "AIzaSy...",
        "groq":       "gsk_...",
        "mistral":    "...",
        ...
      }
    }

Provider ids are stable slugs matching the ``PROMPT_TARGETS`` entries
in ``prompt.py``. The dict only contains providers the user has
supplied keys for; absence means "no key set". The ``version`` field
lets future schema migrations happen without data loss.

## Concurrency model

Writes are atomic via write-to-temp + os.replace so a crashed process
cannot leave a half-written file. Reads and writes are serialized
through an ``asyncio.Lock`` because both the HTTP layer and the
prompt runner can touch the store.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default location inside the container. Docker compose mounts a named
# volume at this path. Overrideable via the AI_SPRAY_KEYS_PATH env var
# for tests / local dev.
DEFAULT_KEYS_PATH = Path(os.environ.get(
    "AI_SPRAY_KEYS_PATH", "/data/keys.json"
))

_SCHEMA_VERSION = 2
_FILE_MODE = 0o600
_DIR_MODE = 0o700


class KeyStore:
    """Asynchronous, file-backed key store.

    Schema version 2 stores, per provider:
      * ``keys[provider]`` — the raw API key string
      * ``models[provider]`` — an object with ``models`` (a list of
        model IDs discovered from the provider's /models endpoint)
        and ``fetched_at`` (ISO-8601 timestamp). Missing or empty
        means discovery hasn't run (or failed) — the caller should
        fall back to the hard-coded defaults in KEYED_PROVIDERS.

    Schema v1 files (keys only, no models) are read transparently; the
    models dict simply starts empty and gets populated on first
    discovery (typically when the user re-saves a key or clicks the
    Refresh button in the UI).
    """

    def __init__(self, path: Path = DEFAULT_KEYS_PATH):
        self._path = path
        self._lock = asyncio.Lock()
        self._keys: dict[str, str] = {}
        self._models: dict[str, dict[str, Any]] = {}
        # Parallel bucket for MCP server tokens (slice B-static). Lives
        # in the same file under a different top-level key so it
        # doesn't collide with the AI-provider keys bucket. No model
        # discovery for MCP servers — what the user authed for is what
        # they get.
        self._mcp_keys: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Read the keys file from disk. Safe to call repeatedly; a
        successful load clears any in-memory state first."""
        async with self._lock:
            keys, models, mcp_keys = await asyncio.to_thread(self._load_sync)
            self._keys = keys
            self._models = models
            self._mcp_keys = mcp_keys
            self._loaded = True

    def _load_sync(self) -> tuple[
        dict[str, str], dict[str, dict[str, Any]], dict[str, str],
    ]:
        if not self._path.exists():
            log.info("no keys file at %s (fresh install)", self._path)
            return {}, {}, {}

        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as e:
            log.error("cannot read keys file %s: %s", self._path, e)
            return {}, {}, {}

        if not raw.strip():
            return {}, {}, {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error(
                "keys file %s is corrupt JSON (%s); leaving empty — "
                "fix or delete the file to recover",
                self._path, e,
            )
            return {}, {}, {}

        if not isinstance(data, dict):
            log.error("keys file root is not an object; ignoring")
            return {}, {}, {}

        # --- keys dict ---
        raw_keys = data.get("keys", {})
        clean_keys: dict[str, str] = {}
        if isinstance(raw_keys, dict):
            for prov, val in raw_keys.items():
                if (isinstance(prov, str)
                        and isinstance(val, str)
                        and val.strip()):
                    clean_keys[prov] = val.strip()

        # --- models dict (only present in schema v2+; tolerate absent) ---
        raw_models = data.get("models", {})
        clean_models: dict[str, dict[str, Any]] = {}
        if isinstance(raw_models, dict):
            for prov, entry in raw_models.items():
                if not isinstance(prov, str) or not isinstance(entry, dict):
                    continue
                models_list = entry.get("models")
                fetched_at  = entry.get("fetched_at")
                if not isinstance(models_list, list):
                    continue
                # Filter to strings only; drop anything weird.
                mids = [m for m in models_list if isinstance(m, str)]
                clean_models[prov] = {
                    "models":     mids,
                    "fetched_at": fetched_at if isinstance(fetched_at, str)
                                  else None,
                }

        # --- mcp_keys dict (slice B-static; absent in older files) ---
        raw_mcp_keys = data.get("mcp_keys", {})
        clean_mcp_keys: dict[str, str] = {}
        if isinstance(raw_mcp_keys, dict):
            for prov, val in raw_mcp_keys.items():
                if (isinstance(prov, str)
                        and isinstance(val, str)
                        and val.strip()):
                    clean_mcp_keys[prov] = val.strip()

        return clean_keys, clean_models, clean_mcp_keys

    async def _save_locked(self) -> None:
        """Persist the in-memory state. Caller must hold ``self._lock``."""
        await asyncio.to_thread(self._save_sync)

    def _save_sync(self) -> None:
        payload = json.dumps(
            {
                "version":  _SCHEMA_VERSION,
                "keys":     self._keys,
                "models":   self._models,
                "mcp_keys": self._mcp_keys,
            },
            indent=2, sort_keys=True,
        ) + "\n"

        parent = self._path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(parent, _DIR_MODE)
            except OSError:
                pass  # may fail on mounted volumes on some hosts
        except OSError as e:
            log.error("cannot create keys directory %s: %s", parent, e)
            raise

        # Atomic write: temp file in the same directory, flush + fsync,
        # then os.replace to swap it into place.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".keys-", suffix=".tmp", dir=parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_path, _FILE_MODE)
            os.replace(tmp_path, self._path)
        except OSError:
            # Best effort cleanup if the replace failed.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Read / write API
    # ------------------------------------------------------------------

    async def has(self, provider: str) -> bool:
        if not self._loaded:
            await self.load()
        return provider in self._keys

    async def get(self, provider: str) -> str | None:
        if not self._loaded:
            await self.load()
        return self._keys.get(provider)

    async def _ensure_loaded_locked(self) -> None:
        """Populate in-memory state from disk. Caller must hold the lock."""
        if not self._loaded:
            self._keys, self._models, self._mcp_keys = (
                await asyncio.to_thread(self._load_sync)
            )
            self._loaded = True

    async def set(self, provider: str, key: str) -> None:
        key = (key or "").strip()
        if not key:
            raise ValueError("key must be a non-empty string")
        async with self._lock:
            await self._ensure_loaded_locked()
            self._keys[provider] = key
            await self._save_locked()
        log.info("saved key for provider %s", provider)

    async def delete(self, provider: str) -> bool:
        async with self._lock:
            await self._ensure_loaded_locked()
            if provider not in self._keys:
                return False
            del self._keys[provider]
            # Also drop the cached models — they're provider-tied via
            # the key, so a cleared key invalidates the catalog.
            self._models.pop(provider, None)
            await self._save_locked()
        log.info("deleted key for provider %s", provider)
        return True

    # ------------------------------------------------------------------
    # Model catalog cache (populated by app/discovery.py)
    # ------------------------------------------------------------------

    async def set_models(
        self, provider: str, models: list[str],
    ) -> None:
        """Cache the discovered model catalog for a provider.

        Empty lists are stored as-is — that's a valid result meaning
        "discovery ran and found nothing". Callers that want to clear
        the entry entirely (e.g. a discovery failure that shouldn't
        overwrite a previously-good cache) should call
        ``clear_models`` instead.
        """
        async with self._lock:
            await self._ensure_loaded_locked()
            from datetime import datetime, timezone
            self._models[provider] = {
                "models":     [m for m in models if isinstance(m, str)],
                "fetched_at": datetime.now(timezone.utc)
                              .isoformat(timespec="seconds")
                              .replace("+00:00", "Z"),
            }
            await self._save_locked()
        log.info("cached %d models for provider %s",
                 len(models), provider)

    async def get_models(self, provider: str) -> list[str] | None:
        """Return the cached model catalog, or None if we have no
        successful discovery result for this provider."""
        if not self._loaded:
            await self.load()
        entry = self._models.get(provider)
        if not entry:
            return None
        return list(entry.get("models", []))

    async def clear_models(self, provider: str) -> None:
        """Drop the cached catalog for a provider without touching the
        key itself. Used when a refresh fails and we want to force the
        UI to fall back to hard-coded defaults rather than serve a
        stale cache indefinitely."""
        async with self._lock:
            await self._ensure_loaded_locked()
            self._models.pop(provider, None)
            await self._save_locked()

    # ------------------------------------------------------------------
    # MCP-server tokens (slice B-static)
    # ------------------------------------------------------------------
    # Parallel to the AI-provider get/set/delete above; same persistence
    # but a different in-memory dict so the two never collide. No
    # discovery semantics — what the user pastes is what we use.

    async def mcp_has(self, provider: str) -> bool:
        if not self._loaded:
            await self.load()
        return bool(self._mcp_keys.get(provider))

    async def mcp_get(self, provider: str) -> str | None:
        if not self._loaded:
            await self.load()
        return self._mcp_keys.get(provider)

    async def mcp_set(self, provider: str, key: str) -> None:
        # Mirror the validation that set() does — non-empty string,
        # non-empty provider slug — so a buggy caller can't quietly
        # write an empty token. Both fields are required.
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        key = (key or "").strip()
        if not key:
            raise ValueError("key must be a non-empty string")
        async with self._lock:
            await self._ensure_loaded_locked()
            self._mcp_keys[provider] = key
            await self._save_locked()
        log.info("saved mcp key for provider %s", provider)

    async def mcp_delete(self, provider: str) -> bool:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        async with self._lock:
            await self._ensure_loaded_locked()
            existed = provider in self._mcp_keys
            self._mcp_keys.pop(provider, None)
            if existed:
                await self._save_locked()
                log.info("deleted mcp key for provider %s", provider)
            return existed

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    async def summary(self) -> dict[str, Any]:
        """Non-sensitive summary: which providers have keys + how
        many models are cached. Never returns full key values."""
        if not self._loaded:
            await self.load()
        out: dict[str, dict[str, Any]] = {}
        for prov, val in self._keys.items():
            m_entry = self._models.get(prov) or {}
            out[prov] = {
                "present":      True,
                "preview":      _mask(val),
                "length":       len(val),
                "model_count":  len(m_entry.get("models") or []),
                "fetched_at":   m_entry.get("fetched_at"),
            }
        return {"providers": out, "path": str(self._path)}

    async def all_cached_models(self) -> dict[str, list[str]]:
        """Return ``{provider: [model_id, ...]}`` for every provider
        with a cached catalog. Used by the prompt-targets endpoint to
        override the hard-coded defaults on a per-provider basis."""
        if not self._loaded:
            await self.load()
        return {
            prov: list(entry.get("models", []))
            for prov, entry in self._models.items()
            if entry.get("models")
        }

    async def mcp_summary(self) -> dict[str, Any]:
        """Non-sensitive summary of stored MCP server tokens.

        Same shape as ``summary()`` but for the parallel mcp_keys
        bucket. No model-count / fetched-at fields — those are
        AI-provider concepts that don't apply to MCP servers.
        """
        if not self._loaded:
            await self.load()
        out: dict[str, dict[str, Any]] = {}
        for prov, val in self._mcp_keys.items():
            out[prov] = {
                "present": True,
                "preview": _mask(val),
                "length":  len(val),
            }
        return {"providers": out, "path": str(self._path)}


def _mask(v: str) -> str:
    if len(v) <= 10:
        return "•" * len(v)
    return f"{v[:4]}…{v[-4:]}"
