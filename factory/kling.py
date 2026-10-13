"""Optional paid AI video clips from the Kling API (https://klingai.com), ~5 s per scene. Two modes:
  t2v  text-to-video (default): the scene description alone generates a native 9:16 clip - no start photo needed
  i2v  image-to-video: a matching Pixabay photo is animated
Anything that fails is simply missing from the result, so the caller falls back
to the next tier (local LTX -> stock footage -> drawn animation).

Cost is controlled by clip COUNTS, since the units a clip consumes depend on model/mode/duration (check your Kling console):
  KLING_MAX_CLIPS_PER_VIDEO (default 4)   KLING_DAILY_CLIP_LIMIT (default 12)

Env: KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_BASE_URL (default https://api.klingai.com; some accounts use
     https://api-singapore.klingai.com), KLING_MODEL (kling-v1-6), KLING_MODE (std|pro), KLING_DURATION (5|10),
     KLING_TIMEOUT_MIN (15)
"""
from __future__ import annotations
import base64, hashlib, hmac, json, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from . import aivideo

ROOT = Path(__file__).resolve().parent.parent
USAGE_FILE = ROOT / "content" / "kling_usage.json"
MOTION = ". Smooth cinematic camera movement and clear natural motion, high detail."
FAST = (" Sped-up fast motion, roughly 1.5x real speed: the action happens briskly and the whole transformation completes"
        " within the clip.")   # single-shot videos: the renderer slows the clip down to natural speed to fill the video
T2V_STYLE = ". Vertical 9:16 video, cinematic, realistic, natural lighting, smooth camera movement, high detail."
ENDPOINT = {"t2v": "/v1/videos/text2video", "i2v": "/v1/videos/image2video"}


def _cfg() -> dict:
    return {"ak": os.getenv("KLING_ACCESS_KEY", "").strip(), "sk": os.getenv("KLING_SECRET_KEY", "").strip(),
            "base": os.getenv("KLING_BASE_URL", "https://api.klingai.com").rstrip("/"), "model": os.getenv("KLING_MODEL", "kling-v1-6"),
            "mode": os.getenv("KLING_MODE", "std"), "duration": os.getenv("KLING_DURATION", "5"),
            "per_video": int(os.getenv("KLING_MAX_CLIPS_PER_VIDEO", "4")), "per_day": int(os.getenv("KLING_DAILY_CLIP_LIMIT", "12")),
            "timeout": int(os.getenv("KLING_TIMEOUT_MIN", "15")) * 60}


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_token(ak: str, sk: str, now: float | None = None) -> str:
    """JWT (HS256): issuer = access key, valid 30 minutes - the scheme Kling's API uses."""
    t = int(now or time.time())
    head = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps({"iss": ak, "exp": t + 1800, "nbf": t - 5}, separators=(",", ":")).encode())
    sig = hmac.new(sk.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{_b64url(sig)}"


def _usage() -> dict:
    try:
        d = json.loads(USAGE_FILE.read_text())
        return d if d.get("day") == str(date.today()) else {"day": str(date.today()), "clips": 0}
    except Exception:
        return {"day": str(date.today()), "clips": 0}


def _add_usage(n: int) -> None:
    u = _usage(); u["clips"] += n
    USAGE_FILE.parent.mkdir(exist_ok=True); USAGE_FILE.write_text(json.dumps(u))


def status() -> dict:
    c = _cfg(); u = _usage()
    return {"ready": bool(c["ak"] and c["sk"]), "model": c["model"], "quality": c["mode"], "duration_s": int(c["duration"]),
            "max_clips_per_video": c["per_video"], "daily_clip_limit": c["per_day"], "clips_today": u["clips"]}


def _call(c: dict, method: str, path: str, **kw) -> dict:
    r = requests.request(method, c["base"] + path, headers={"Authorization": "Bearer " + make_token(c["ak"], c["sk"]), "Content-Type": "application/json"}, timeout=60, **kw)
    try: d = r.json()
    except ValueError: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400 or d.get("code", 0) != 0:
        raise RuntimeError(f"HTTP {r.status_code} code={d.get('code')}: {d.get('message', r.text[:200])}")
    return d["data"]


def _one(c: dict, job: dict, log: Callable[[str], None]) -> Path | None:
    """Submit one clip, wait for it, download it. Returns the file or None."""
    try:
        path = ENDPOINT[job["mode"]]
        body = {"model_name": c["model"], "mode": c["mode"], "duration": job.get("seconds", c["duration"]), "prompt": job["prompt"][:2500],
                "negative_prompt": "blurry, distorted, watermark, text, low quality, deformed", "cfg_scale": 0.5}
        if job["mode"] == "i2v": body["image"] = base64.b64encode(Path(job["image"]).read_bytes()).decode()   # raw base64, no data: prefix
        else: body["aspect_ratio"] = "9:16"
        task = _call(c, "POST", path, data=json.dumps(body))["task_id"]
        log(f"[kling] scene {job['index']} submitted ({task})")
        deadline = time.time() + c["timeout"]
        while time.time() < deadline:
            time.sleep(float(os.getenv("KLING_POLL_SEC", "8")))
            d = _call(c, "GET", f"{path}/{task}")
            if d["task_status"] == "succeed":
                url = d["task_result"]["videos"][0]["url"]
                data = requests.get(url, timeout=120); data.raise_for_status()
                Path(job["out"]).write_bytes(data.content)
                log(f"[kling] scene {job['index']} done"); return Path(job["out"])
            if d["task_status"] == "failed":
                raise RuntimeError(d.get("task_status_msg", "task failed"))
        raise RuntimeError("timed out waiting for Kling")
    except Exception as e:
        log(f"[kling] scene {job['index']} failed: {str(e)[:200]}"); return None


def generate(job_id: str, scenes: list[dict], pixabay_key: str, log: Callable[[str], None] = print, mode: str = "t2v") -> dict[int, Path]:
    """Return {scene_index: mp4 path}. Never raises. Respects the per-video and daily clip caps."""
    c = _cfg(); mode = mode if mode in ENDPOINT else "t2v"
    if not (c["ak"] and c["sk"]): return {}
    aivideo.IMG_DIR.mkdir(parents=True, exist_ok=True); aivideo.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[int, Path] = {}; jobs, used = [], set()
    budget = max(0, min(c["per_video"], c["per_day"] - _usage()["clips"]))
    secs = "10" if len(scenes) == 1 else c["duration"]                        # a single-shot video gets the longest clip
    for i, sc in enumerate(scenes):
        out = aivideo.CLIP_DIR / f"{job_id}_kl_{mode}{i}.mp4"       # per mode, so switching modes never reuses the other mode's clip
        if out.exists(): done[i] = out; continue                           # cached from an earlier attempt - free
        if len(jobs) >= budget: continue
        q = sc.get("stock_query") or sc.get("visual") or ""
        desc = (sc.get("visual") or q).strip().rstrip(".")
        if mode == "t2v":
            jobs.append({"index": i, "mode": "t2v", "prompt": desc + T2V_STYLE + (FAST if len(scenes) == 1 else ""), "out": str(out), "seconds": secs}); continue
        try: photo = aivideo.pixabay_photo(q, pixabay_key, used, sc.get("visual", "")) if pixabay_key and q else None
        except Exception as e: log(f"[kling] no start photo for scene {i}: {e}"); photo = None
        if not photo: continue                                              # image-to-video needs a start image
        jobs.append({"index": i, "mode": "i2v", "image": str(photo), "prompt": desc + MOTION + (FAST if len(scenes) == 1 else ""), "out": str(out), "seconds": secs})
    if not jobs:
        if budget == 0: log("[kling] clip limit reached (per video or per day); skipping")
        return done
    _add_usage(len(jobs))                                                   # count at submit time: a submitted clip is billed even if we time out
    with ThreadPoolExecutor(max_workers=min(5, len(jobs))) as ex:           # Kling allows 5 concurrent tasks on the trial package
        for job, res in zip(jobs, ex.map(lambda j: _one(c, j, log), jobs)):
            if res: done[job["index"]] = res
    return done
