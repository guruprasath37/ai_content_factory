"""Optional AI video clips from Apiframe (https://apiframe.ai), one API in front of many video models. Two modes:
  t2v  text-to-video (default): the scene description alone generates a native 9:16 clip
  i2v  image-to-video: a matching Pixabay photo is uploaded to Apiframe and used as the first frame
Anything that fails is simply missing from the result, so the caller falls back
to the next tier (local LTX -> stock footage -> drawn animation). Failed jobs are refunded by Apiframe.

Credits per clip depend on the model (see https://apiframe.ai/docs/pricing). The default, seedance-1-pro-fast at 720p
and 5 s, costs 16 credits; single-shot videos (visual-first series) ask for one 10 s clip, 31 credits. Spend is capped by clip COUNTS:
  APIFRAME_MAX_CLIPS_PER_VIDEO (default 3)   APIFRAME_DAILY_CLIP_LIMIT (default 6)

Env: APIFRAME_API_KEY, APIFRAME_MODEL (seedance-1-pro-fast; any seedance-* or kling-* model id),
     APIFRAME_RESOLUTION (480p|720p|1080p, Seedance only), APIFRAME_DURATION (5|10), APIFRAME_TIMEOUT_MIN (15)
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from . import aivideo

ROOT = Path(__file__).resolve().parent.parent
USAGE_FILE = ROOT / "content" / "apiframe_usage.json"
BASE = "https://api.apiframe.ai/v2"
MOTION = ". Smooth cinematic camera movement and clear natural motion, high detail."
FAST = (" Sped-up fast motion, roughly 1.5x real speed: the action happens briskly and the whole transformation completes"
        " within the clip.")   # single-shot videos: the renderer slows the clip down to natural speed to fill the video
T2V_STYLE = ". Vertical 9:16 video, cinematic, realistic, natural lighting, smooth camera movement, high detail."
NEGATIVE = "blurry, distorted, watermark, text, low quality, deformed"
FAMILIES = ("seedance", "kling")   # model families whose parameter objects this client knows


def _cfg() -> dict:
    return {"key": os.getenv("APIFRAME_API_KEY", "").strip(), "model": os.getenv("APIFRAME_MODEL", "seedance-1-pro-fast").strip(),
            "resolution": os.getenv("APIFRAME_RESOLUTION", "720p"), "duration": int(os.getenv("APIFRAME_DURATION", "5")),
            "per_video": int(os.getenv("APIFRAME_MAX_CLIPS_PER_VIDEO", "3")), "per_day": int(os.getenv("APIFRAME_DAILY_CLIP_LIMIT", "6")),
            "timeout": int(os.getenv("APIFRAME_TIMEOUT_MIN", "15")) * 60}


def _family(model: str) -> str | None:
    return next((f for f in FAMILIES if model.startswith(f)), None)


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
    return {"ready": bool(c["key"] and _family(c["model"])), "configured": bool(c["key"]), "model": c["model"],
            "supported_model": bool(_family(c["model"])), "resolution": c["resolution"], "duration_s": c["duration"],
            "max_clips_per_video": c["per_video"], "daily_clip_limit": c["per_day"], "clips_today": u["clips"]}


def _check(r: requests.Response) -> dict:
    try: d = r.json()
    except ValueError: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        msg = d.get("message") or d.get("error") or r.text[:200]
        if r.status_code == 402: msg = "out of Apiframe credits (" + str(msg) + ")"
        raise RuntimeError(f"HTTP {r.status_code}: {msg}")
    return d


def _upload(c: dict, image: str) -> str:
    """Upload a local image; Apiframe keeps it 1-2 hours, long enough to start the job. Returns its CDN URL."""
    with open(image, "rb") as f:
        r = requests.post(BASE + "/uploads", headers={"X-API-Key": c["key"]}, files={"file": f}, timeout=120)
    return _check(r)["url"]


def _body(c: dict, job: dict, image_url: str | None) -> dict:
    fam = _family(c["model"]); secs = job.get("seconds", c["duration"])
    if fam == "seedance":
        params = {"duration": secs, "resolution": c["resolution"], "aspect_ratio": "9:16"}
    else:   # kling
        params = {"duration": secs, "aspect_ratio": "9:16", "negative_prompt": NEGATIVE}
    if image_url: params["start_image"] = image_url
    return {"prompt": job["prompt"][:4000], "model": c["model"], fam + "Params": params}


def _one(c: dict, job: dict, log: Callable[[str], None]) -> Path | None:
    """Submit one clip, wait for it, download it. Returns the file or None."""
    hdr = {"X-API-Key": c["key"], "Content-Type": "application/json"}
    try:
        image_url = _upload(c, job["image"]) if job["mode"] == "i2v" else None
        r = requests.post(BASE + "/videos/generate", headers=hdr, data=json.dumps(_body(c, job, image_url)), timeout=60)
        task = _check(r)["jobId"]
        log(f"[apiframe] scene {job['index']} submitted ({task})")
        deadline = time.time() + c["timeout"]
        while time.time() < deadline:
            time.sleep(float(os.getenv("APIFRAME_POLL_SEC", "8")))
            d = _check(requests.get(f"{BASE}/jobs/{task}", headers=hdr, timeout=60))
            if d.get("status") == "COMPLETED":
                url = (d.get("result") or {}).get("videoUrl")
                if not url: raise RuntimeError("completed without a videoUrl")
                data = requests.get(url, timeout=120); data.raise_for_status()
                Path(job["out"]).write_bytes(data.content)
                log(f"[apiframe] scene {job['index']} done ({d.get('creditCost', '?')} credits)"); return Path(job["out"])
            if d.get("status") == "FAILED":
                raise RuntimeError(d.get("error") or "job failed")
        raise RuntimeError("timed out waiting for Apiframe")
    except Exception as e:
        log(f"[apiframe] scene {job['index']} failed: {str(e)[:200]}"); return None


def generate(job_id: str, scenes: list[dict], pixabay_key: str, log: Callable[[str], None] = print, mode: str = "t2v") -> dict[int, Path]:
    """Return {scene_index: mp4 path}. Never raises. Respects the per-video and daily clip caps."""
    c = _cfg(); mode = mode if mode in ("t2v", "i2v") else "t2v"
    if not c["key"]: return {}
    if not _family(c["model"]):
        log(f"[apiframe] model {c['model']!r} is not supported here (use a seedance-* or kling-* model); skipping"); return {}
    aivideo.IMG_DIR.mkdir(parents=True, exist_ok=True); aivideo.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[int, Path] = {}; jobs, used = [], set()
    budget = max(0, min(c["per_video"], c["per_day"] - _usage()["clips"]))
    secs = 10 if len(scenes) == 1 else c["duration"]                          # a single-shot video gets the longest clip
    for i, sc in enumerate(scenes):
        out = aivideo.CLIP_DIR / f"{job_id}_af_{mode}{i}.mp4"       # per mode, so switching modes never reuses the other mode's clip
        if out.exists(): done[i] = out; continue                           # cached from an earlier attempt - free
        if len(jobs) >= budget: continue
        q = sc.get("stock_query") or sc.get("visual") or ""
        desc = (sc.get("visual") or q).strip().rstrip(".")
        if mode == "t2v":
            jobs.append({"index": i, "mode": "t2v", "prompt": desc + T2V_STYLE + (FAST if len(scenes) == 1 else ""), "out": str(out), "seconds": secs}); continue
        try: photo = aivideo.pixabay_photo(q, pixabay_key, used, sc.get("visual", "")) if pixabay_key and q else None
        except Exception as e: log(f"[apiframe] no start photo for scene {i}: {e}"); photo = None
        if not photo: continue                                              # image-to-video needs a start image
        jobs.append({"index": i, "mode": "i2v", "image": str(photo), "prompt": desc + MOTION + (FAST if len(scenes) == 1 else ""), "out": str(out), "seconds": secs})
    if not jobs:
        if budget == 0: log("[apiframe] clip limit reached (per video or per day); skipping")
        return done
    _add_usage(len(jobs))                                                   # count at submit time, like the Kling client
    with ThreadPoolExecutor(max_workers=min(3, len(jobs))) as ex:
        for job, res in zip(jobs, ex.map(lambda j: _one(c, j, log), jobs)):
            if res: done[job["index"]] = res
    return done
