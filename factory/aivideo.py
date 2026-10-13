"""Optional AI video clips (LTX-Video 2B, local). Each scene gets a short clip animated from a matching
Pixabay photo; anything that fails is simply missing from the result so the caller falls back to stock footage.

Env: LTX_MODEL_DIR (default models/LTX-Video), LTX_WIDTH/LTX_HEIGHT (480x864), LTX_FRAMES (49 = ~2s),
     LTX_STEPS (30), LTX_TIMEOUT_MIN (120)
"""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, time, zlib
from pathlib import Path
from typing import Callable

import requests

ROOT = Path(__file__).resolve().parent.parent   # project root
CONTENT = ROOT / "content"
IMG_DIR = ROOT / "assets" / "pixabay"
CLIP_DIR = ROOT / "assets" / "ai_clips"   # kept so a retried render does not regenerate (slow) clips
STYLE = ". The camera slowly pushes in. Clear natural movement: wind, flowing water, drifting clouds, swaying plants, moving light. Cinematic, natural lighting, sharp detail."
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, deformed"


def model_dir() -> Path:
    p = Path(os.getenv("LTX_MODEL_DIR", "models/LTX-Video")).expanduser()
    return p if p.is_absolute() else ROOT / p


def status() -> dict:
    d = model_dir()
    have_model = (d / "model_index.json").exists() and any((d / "transformer").glob("*.safetensors")) and any((d / "text_encoder").glob("*.safetensors"))
    try:
        import torch
        cuda = torch.cuda.is_available()
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0
    except Exception:
        cuda, vram = False, 0
    return {"ready": bool(have_model and cuda), "model_downloaded": have_model, "cuda": cuda, "vram_gb": vram,
            "resolution": f"{os.getenv('LTX_WIDTH', '480')}x{os.getenv('LTX_HEIGHT', '864')}", "frames": int(os.getenv("LTX_FRAMES", "49"))}


def unload_ollama() -> None:
    """The local LLM holds ~5 GB of the 8 GB GPU; ask Ollama to drop it before video generation."""
    try:
        from . import llm
        if llm.ollama_model():
            requests.post(f"{llm.OLLAMA_URL}/api/generate", json={"model": llm.ollama_model(), "keep_alive": 0}, timeout=15)
            time.sleep(2)
    except Exception:
        pass


_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "with", "for", "from", "by", "is", "are", "close", "up", "view", "shot", "scene"}


def relevance(text: str, tags: str) -> int:
    """How many of the scene's words appear in a Pixabay result's tags (exact words or simple plurals)."""
    words = {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}
    have = {t.strip().lower() for t in tags.split(",")} | set(re.findall(r"[a-z]+", tags.lower()))
    return sum(1 for w in words if w in have or w.rstrip("s") in have or w + "s" in have)


def pixabay_photo(query: str, key: str, used: set, context: str = "") -> Path | None:
    """A vertical Pixabay photo to use as the clip's first frame (downloaded, never hotlinked)."""
    cache = IMG_DIR / f"imgsearch_{hashlib.md5(query.encode()).hexdigest()}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        hits = json.loads(cache.read_text())
    else:
        r = requests.get("https://pixabay.com/api/", params={"key": key, "q": query[:100], "image_type": "photo", "orientation": "vertical",
                         "min_height": 1000, "safesearch": "true", "order": "popular", "per_page": 20}, timeout=30)
        r.raise_for_status(); hits = r.json().get("hits", []); cache.write_text(json.dumps(hits))
    # best tag match first (popularity order breaks ties); skip photos whose tags match nothing in the scene
    ranked = sorted((h for h in hits if h["id"] not in used), key=lambda h: -relevance(query + " " + context, h.get("tags", "")))
    for h in ranked:
        if relevance(query + " " + context, h.get("tags", "")) == 0: break
        used.add(h["id"]); path = IMG_DIR / f"photo_{h['id']}.jpg"
        if not path.exists():
            img = requests.get(h["largeImageURL"], timeout=60); img.raise_for_status(); path.write_bytes(img.content)
        return path
    return None


def generate(job_id: str, scenes: list[dict], key: str, max_scenes: int = 6, log: Callable[[str], None] = print, only: set[int] | None = None, mode: str = "i2v") -> dict[int, Path]:
    """Return {scene_index: mp4 path} for the clips that were generated. Never raises.
    mode "i2v" animates a matching Pixabay photo; "t2v" generates from the scene description alone."""
    mode = "t2v" if mode == "t2v" else "i2v"
    st = status()
    if not st["ready"]:
        log(f"[ai video] not available (model_downloaded={st['model_downloaded']}, cuda={st['cuda']})"); return {}
    IMG_DIR.mkdir(parents=True, exist_ok=True); CLIP_DIR.mkdir(parents=True, exist_ok=True)
    W, H = int(os.getenv("LTX_WIDTH", "480")), int(os.getenv("LTX_HEIGHT", "864"))
    jobs, used = [], set()
    for i, sc in enumerate(scenes[:max_scenes]):
        if only is not None and i not in only: continue          # e.g. scenes a better provider already covered
        q = sc.get("stock_query") or sc.get("visual") or ""
        photo = None
        if mode == "i2v":
            try: photo = pixabay_photo(q, key, used, sc.get("visual", "")) if key and q else None
            except Exception as e: log(f"[ai video] no start photo for scene {i}: {e}")
        text = (sc.get("visual") or q).strip().rstrip(".") + STYLE
        jobs.append({"index": i, "prompt": text, "image": str(photo) if photo else None, "out": str(CLIP_DIR / f"{job_id}_ai_{mode}{i}.mp4")})   # per mode: switching modes never reuses the other mode's clip
    done: dict[int, Path] = {j["index"]: Path(j["out"]) for j in jobs if Path(j["out"]).exists()}
    jobs = [j for j in jobs if j["index"] not in done]
    if not jobs: return done
    spec = {"model_dir": str(model_dir()), "width": W, "height": H, "frames": int(os.getenv("LTX_FRAMES", "49")), "steps": int(os.getenv("LTX_STEPS", "30")),
            "guidance": float(os.getenv("LTX_GUIDANCE", "4.0")), "seed": zlib.crc32(job_id.encode()) % 100000, "negative": NEGATIVE, "jobs": jobs}
    spec_path = CONTENT / f"{job_id}_ltx.json"; spec_path.write_text(json.dumps(spec))
    unload_ollama()
    try:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "ltx_worker.py"), str(spec_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
        deadline = time.time() + int(os.getenv("LTX_TIMEOUT_MIN", "120")) * 60
        for line in p.stdout:
            try: ev = json.loads(line)
            except ValueError: continue
            if ev.get("event") == "clip_done": done[ev["index"]] = Path(ev["out"])
            log(f"[ai video] {ev}")
            if time.time() > deadline: p.kill(); log("[ai video] timed out"); break
        p.wait(timeout=30)
        if p.returncode: log(f"[ai video] worker exited {p.returncode}: {p.stderr.read()[-600:]}")
    except Exception as e:
        log(f"[ai video] failed: {e}")
    finally:
        spec_path.unlink(missing_ok=True)
    return {i: p for i, p in done.items() if p.exists()}
