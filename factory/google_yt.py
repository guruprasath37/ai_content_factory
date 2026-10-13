"""Read-only YouTube Data API v3 access.

Uses YOUTUBE_API_KEY if set, otherwise the Google service-account key named by
GOOGLE_APPLICATION_CREDENTIALS (path relative to this folder is fine). Read-only scope only.
"""
from __future__ import annotations
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent   # project root
_tok: dict = {"creds": None}


def _creds_file() -> Path | None:
    p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not p: return None
    f = Path(p).expanduser()
    f = f if f.is_absolute() else ROOT / f
    return f if f.exists() else None


def available() -> bool:
    return bool(os.getenv("YOUTUBE_API_KEY", "").strip() or _creds_file())


def auth_mode() -> str:
    if os.getenv("YOUTUBE_API_KEY", "").strip(): return "api_key"
    return "service_account" if _creds_file() else "none"


def _bearer() -> str:
    import google.auth.transport.requests as gtr
    from google.oauth2 import service_account
    c = _tok["creds"]
    if c is None:
        c = _tok["creds"] = service_account.Credentials.from_service_account_file(str(_creds_file()), scopes=["https://www.googleapis.com/auth/youtube.readonly"])
    if not c.valid: c.refresh(gtr.Request())
    return c.token


def get(path: str, **params) -> dict:
    headers, key = {}, os.getenv("YOUTUBE_API_KEY", "").strip()
    if key: params["key"] = key
    elif _creds_file(): headers["Authorization"] = "Bearer " + _bearer()
    else: raise RuntimeError("No YouTube credentials: set YOUTUBE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in .env")
    r = requests.get("https://www.googleapis.com/youtube/v3/" + path, params=params, headers=headers, timeout=30)
    if r.status_code >= 400:
        try: msg = r.json()["error"]["message"]
        except Exception: msg = r.text[:200]
        raise RuntimeError(f"YouTube API {r.status_code}: {msg}")
    return r.json()
