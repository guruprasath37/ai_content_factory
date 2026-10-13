from __future__ import annotations
import asyncio, json, math, os, queue, re, shutil, sqlite3, subprocess, threading, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import edge_tts
import numpy as np
import requests
from dotenv import load_dotenv
from . import aivideo, analytics, apiframe, google_yt, kling, llm, niche
from . import series as sr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent   # project root (content/, assets/, .env, database live here)
CONTENT = ROOT / "content"
ASSETS = ROOT / "assets"
DB_PATH = Path(os.environ.get("DB_PATH", str(ROOT / "content_factory.sqlite3")))
CONTENT.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")

@asynccontextmanager
async def lifespan(_app):
    init_db()
    con = db()   # a restart can leave jobs mid-flight; mark them retryable
    con.execute("UPDATE jobs SET status='render_failed', error='Interrupted by a server restart; press Render / retry.' WHERE status IN ('rendering','generating')")
    con.commit()
    for r in con.execute("SELECT slug, spec_json FROM custom_series"): sr.register(r["slug"], json.loads(r["spec_json"]))
    con.close()
    threading.Thread(target=worker_loop, daemon=True, name="factory-worker").start()
    yield

app = FastAPI(title="AI Content Factory", version="0.2.0", lifespan=lifespan)


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 niche TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
 title TEXT, description TEXT, hashtags TEXT, script_json TEXT, video_path TEXT,
 audio_path TEXT, srt_path TEXT, error TEXT, youtube_id TEXT, instagram_id TEXT,
 series TEXT, topic TEXT, source TEXT DEFAULT 'manual'
);
CREATE TABLE IF NOT EXISTS ideas (
 id INTEGER PRIMARY KEY AUTOINCREMENT, series TEXT NOT NULL, topic TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'new',
 created_at TEXT NOT NULL, job_id TEXT
);
CREATE TABLE IF NOT EXISTS custom_series (slug TEXT PRIMARY KEY, spec_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metrics (
 job_id TEXT NOT NULL, platform TEXT NOT NULL, fetched_at TEXT NOT NULL, views INTEGER, likes INTEGER, comments INTEGER,
 PRIMARY KEY (job_id, platform)
);
"""
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
    for col, ddl in (("series", "TEXT"), ("topic", "TEXT"), ("source", "TEXT DEFAULT 'manual'"), ("provenance", "TEXT"), ("publish_at", "TEXT")):
        if col not in cols: con.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")
    con.commit(); con.close()

def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con

def now(): return datetime.now(timezone.utc).isoformat()

def get_setting(key, default=None):
    con = db(); r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone(); con.close()
    return r["value"] if r else default

def set_setting(key, value):
    con = db(); con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value))); con.commit(); con.close()

def rowdict(row):
    if not row: return None
    d = dict(row)
    for k in ("script_json", "hashtags", "provenance"):
        if d.get(k):
            try: d[k] = json.loads(d[k])
            except Exception: pass
    return d

class GenerateRequest(BaseModel):
    niche: str = "AI tools & technology"
    language: str = "English"
    duration_seconds: int = Field(default=30, ge=15, le=180)
    topic: str | None = None
    series: str = "tiny_tales"
    made_for_kids: bool | None = None   # default: the series' own setting
    ai_video: bool = False               # animate scenes with AI video (falls back tier by tier, per scene)
    ai_provider: str = Field(default="ltx", pattern="^(ltx|kling|apiframe|gemini)$")   # ltx = free & local, kling/apiframe = paid APIs (LTX fills any gaps), gemini = you make one clip in Gemini and upload it
    avoid: list[str] = Field(default_factory=list, max_length=30)      # titles already shown, so the next idea is a different one
    ai_mode: str = Field(default="t2v", pattern="^(t2v|i2v)$")         # t2v = generate from the script/scene text, i2v = animate a matching photo (both engines)

class ReviseRequest(BaseModel):
    instructions: str = Field(min_length=3, max_length=1000)

class JobAction(BaseModel):
    confirm: bool = True

def template_script(niche, topic, language, duration, series="tiny_tales"):
    """Offline fallback for the two original series when no LLM is reachable."""
    if series == "color_lab":
        subject = topic or "What Happens When Every Color Becomes One?"
        data = {
            "title": subject[:70],
            "description": "A playful, original color experiment made with animated shapes. Watch the colors blend and transform.",
            "hashtags": ["ColorLab", "Satisfying", "ColorAnimation", "Shorts", "CreativeAnimation"],
            "hook": "What if every color could meet in one place?",
            "scenes": [
                {"narration": "What if every color could meet in one place?", "visual": "A glowing rainbow orb appears"},
                {"narration": "Red, blue, and yellow begin to dance around each other.", "visual": "Three colorful orbs orbit"},
                {"narration": "They swirl together, making new colors along the way.", "visual": "Orbs blend into a rainbow spiral"},
                {"narration": "Then one bright spectrum blooms across the screen.", "visual": "A spectrum expands and pulses"},
                {"narration": "Every color has a place in the picture.", "visual": "Colors settle into a calm rainbow"},
            ],
        }
    else:
        subject = topic or "The Little Robot Who Found a Star"
        data = {
            "title": subject[:70],
            "description": "An original tiny tale about curiosity, kindness, and helping someone shine. Made with simple animated artwork.",
            "hashtags": ["TinyTales", "KidsStories", "LittleRobot", "Kindness", "Shorts"],
            "hook": "One tiny robot found a light where nobody was looking.",
            "scenes": [
                {"narration": "One tiny robot found a little star hiding in the garden.", "visual": "Little robot spots a glowing star"},
                {"narration": "The star was dim, and it missed the night sky.", "visual": "Star glows softly beside robot"},
                {"narration": "The robot built a tiny paper rocket to help it rise.", "visual": "Robot makes a tiny rocket"},
                {"narration": "Together they floated above the sleepy flowers.", "visual": "Robot and star float upward"},
                {"narration": "The star returned to the sky and twinkled a thank-you.", "visual": "Star shines in the sky"},
                {"narration": "The robot learned that small acts can make a big light.", "visual": "Robot waves under the stars"},
            ],
        }
    data["series"] = series
    return data

def make_script(series, niche, topic, language, duration, single=False, avoid=()):
    """LLM-written script (local Ollama by default). Falls back to a built-in template for the two original
    series if the LLM is unreachable, so the factory keeps running; the fallback is recorded in the script."""
    try:
        data = sr.generate_script(series, topic, language, duration, niche, single=single, avoid=avoid)
        data["script_source"] = f"{llm.LAST['provider']}:{llm.LAST['model']}"
        data["script_provider"], data["script_model"], data["script_fell_back_from"] = llm.LAST["provider"], llm.LAST["model"], llm.LAST["fell_back_from"]
    except Exception as e:
        if single or series not in ("tiny_tales", "color_lab"): raise   # the templates are multi-scene
        data = template_script(niche, topic, language, duration, series)
        data["script_source"] = "template"; data["llm_error"] = str(e)[:300]
        data["script_provider"], data["script_model"], data["script_fell_back_from"] = "template", None, None
    return data


def find_font(bold=False):
    candidates = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
                  ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for p in candidates:
        if Path(p).exists(): return p
    return None

def scene_bounds(scenes, duration):
    if all("_slot" in s for s in scenes): return [tuple(s["_slot"]) for s in scenes]   # fixed slots from a padded voice-over
    weights = [max(1, len(re.findall(r"\w+", s["narration"]))) for s in scenes]
    total = sum(weights); cur = 0.0; out = []
    for w in weights:
        end = cur + duration * w / total
        out.append((cur, end)); cur = end
    return out

def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)

def _lerp(a, b, t): return a + (b - a) * t

class _Pen:
    """Draws in 540x960 design units onto an image scaled by m."""
    def __init__(self, img, m):
        self.im, self.m = img, m
        self.d = ImageDraw.Draw(img, "RGBA")
    def circle(self, cx, cy, r, fill, outline=None, w=0):
        m = self.m
        self.d.ellipse((cx*m-r*m, cy*m-r*m, cx*m+r*m, cy*m+r*m), fill=fill, outline=outline, width=int(w*m))
    def ellipse(self, cx, cy, rx, ry, fill, outline=None, w=0):
        m = self.m
        self.d.ellipse((cx*m-rx*m, cy*m-ry*m, cx*m+rx*m, cy*m+ry*m), fill=fill, outline=outline, width=int(w*m))
    def rrect(self, x0, y0, x1, y1, rad, fill, outline=None, w=0):
        m = self.m
        self.d.rounded_rectangle((x0*m, y0*m, x1*m, y1*m), radius=rad*m, fill=fill, outline=outline, width=int(w*m))
    def line(self, pts, fill, w):
        m = self.m
        self.d.line([(x*m, y*m) for x, y in pts], fill=fill, width=max(1, int(w*m)), joint="curve")
    def poly(self, pts, fill):
        m = self.m
        self.d.polygon([(x*m, y*m) for x, y in pts], fill=fill)
    def arc(self, cx, cy, rx, ry, a0, a1, fill, w):
        m = self.m
        self.d.arc((cx*m-rx*m, cy*m-ry*m, cx*m+rx*m, cy*m+ry*m), a0, a1, fill=fill, width=max(1, int(w*m)))

def make_animated_background(job_id, title, series, duration, scenes):
    """Render an original procedural 9:16 animation (not AI-generated footage): scene-driven,
    with glow, particles, parallax, camera motion and transitions. Frames are piped to ffmpeg."""
    import colorsys, random
    from PIL import ImageChops, ImageFilter
    W, H, FPS = 720, 1280, 30
    LM = (W * 1.5) / 540          # main overlay layer scale (design units -> px), supersampled then downsized
    GM = (W / 2) / 540            # glow layer scale (half res, blurred)
    rng = np.random.RandomState(int(job_id[:8], 16))
    bounds = scene_bounds(scenes, duration)
    total = int(duration * FPS) + 2
    fp_b, fp_r = find_font(True), find_font(False)
    title_font = ImageFont.truetype(fp_b, int(27 * LM)) if fp_b else ImageFont.load_default()
    tag_font = ImageFont.truetype(fp_b or fp_r, int(14 * LM)) if (fp_b or fp_r) else ImageFont.load_default()

    # Static helpers
    yy, xx = np.mgrid[0:H, 0:W]
    vig = (1 - 0.55 * np.clip(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2.2, 0, 1) ** 1.3).astype(np.float32)[..., None]
    bloom_lut = [int(i * (0.6 if series == "color_lab" else 0.4)) for i in range(256)] * 3
    lw, lh = 240, 427
    lyy, lxx = np.mgrid[0:lh, 0:lw]
    fx = ((lxx / lw - 0.5) * 0.5625).astype(np.float32); fy = (lyy / lh - 0.5).astype(np.float32)

    stars = [(rng.rand() * 540, rng.rand() * 600, 0.6 + rng.rand() * 1.8, rng.rand() * 6.28, 1 + rng.rand() * 3) for _ in range(150)]
    flies = [(rng.rand() * 540, 620 + rng.rand() * 280, rng.rand() * 6.28, 0.4 + rng.rand()) for _ in range(26)]
    flowers = [(18 + i * 24 + rng.rand() * 10, rng.rand() * 6.28, rng.randint(0, 5)) for i in range(22)]
    parts = [(rng.rand() * 6.28, 0.2 + rng.rand() * 0.9, rng.rand(), rng.rand(), 1.5 + rng.rand() * 3.5) for _ in range(90)]
    petal_cols = [(255, 140, 190), (255, 215, 110), (140, 230, 255), (190, 160, 255), (255, 170, 120)]

    back_y = lambda x: 705 + 24 * math.sin(x * 0.011 + 1.0) + 12 * math.sin(x * 0.027)
    front_y = lambda x: 800 + 20 * math.sin(x * 0.008 + 3.0) + 10 * math.sin(x * 0.02 + 1)

    def scene_at(t):
        for i, (a, b) in enumerate(bounds):
            if t < b: return i, t - a
        return len(bounds) - 1, t - bounds[-1][0]

    def draw_title(pen, t):
        a = min(1.0, t / 0.5, max(0.0, (3.4 - t) / 0.7))
        if a <= 0: return
        words, lines, cur = title.split(), [], ""
        for wd in words:
            test = (cur + " " + wd).strip()
            if pen.d.textbbox((0, 0), test, font=title_font)[2] > 470 * LM and cur: lines.append(cur); cur = wd
            else: cur = test
        if cur: lines.append(cur)
        lines = lines[:2]; ah = int(255 * a)
        top = 70 - (1 - a) * 14
        pen.rrect(24, top, 516, top + 22 + 36 * len(lines), 20, (8, 10, 30, int(150 * a)), (255, 255, 255, int(60 * a)), 1)
        for j, ln in enumerate(lines):
            pen.d.text((40 * LM, (top + 10 + j * 36) * LM), ln, font=title_font, fill=(255, 255, 255, ah), stroke_width=int(LM), stroke_fill=(20, 20, 50, ah))

    def ring(pen, cx, cy, lt, color):
        if lt < 1.1:
            a = int(200 * (1 - lt / 1.1) ** 1.5); r = lt * 420
            pen.circle(cx, cy, r, None, color + (a,), 5 * (1 - lt / 1.1) + 1)

    # ---------------- Color Lab ----------------
    def lab_frame(t, q, si, lt):
        s = (np.sin(fx * 7 + t * 0.9) + np.sin(fy * 5 - t * 1.1) + np.sin((fx + fy) * 6 + t * 0.7) + np.sin(np.hypot(fx, fy) * 14 - t * 2.2))
        hue = s * 0.11 + q * 0.6 + si * 0.13
        base = np.stack([0.5 + 0.5 * np.cos(6.2832 * (hue + o)) for o in (0.0, 0.33, 0.67)], -1) * 0.62 + 0.2
        f = np.zeros((lh, lw), np.float32); acc = np.zeros((lh, lw, 3), np.float32)
        conv = _smooth((q - 0.62) / 0.28); n = 8
        for i in range(n):
            ang = t * (0.5 + 0.12 * i) * (-1 if i % 2 else 1) + i * 6.2832 / n
            R = (0.1 + 0.09 * (1 + math.sin(t * 0.6 + i + si))) * (1 - 0.7 * conv)
            bx = math.cos(ang) * R * 0.9; by = math.sin(ang * 1.2) * R * 1.8 - 0.05
            r = 0.045 + 0.02 * math.sin(t * 2 + i) + 0.02 * math.exp(-lt * 4)
            fi = (r * r) / ((fx - bx) ** 2 + (fy - by) ** 2 + 1e-4)
            c = colorsys.hsv_to_rgb((i / n + q * 0.5 + t * 0.03) % 1, 0.95, 1.0)
            f += fi; acc += fi[..., None] * np.array(c, np.float32)
        col = acc / (f[..., None] + 1e-4)
        m = np.clip((f - 0.75) / 0.6, 0, 1); m = (m * m * (3 - 2 * m))[..., None]
        rim = (np.clip((f - 0.7) / 0.3, 0, 1) * (1 - np.clip((f - 1.3) / 0.9, 0, 1)))[..., None]
        out = base * (1 - m) + col * m * 1.1 + rim * 0.4 + 0.18 * math.exp(-lt * 6)
        im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        layer = Image.new("RGBA", (int(W * 1.5), int(H * 1.5)), (0, 0, 0, 0)); pen = _Pen(layer, LM)
        for a0, sp, ph, ph2, size in parts:
            rr = 40 + 330 * ((t * 0.07 * (0.5 + sp) + ph) % 1.0)
            ang = a0 + t * 0.35 * sp + rr * 0.004
            px, py = 270 + math.cos(ang) * rr * 0.9, 430 + math.sin(ang) * rr * 1.25
            al = int(230 * math.sin(math.pi * min(1, rr / 370)) * (0.55 + 0.45 * math.sin(t * 4 + ph2 * 9)))
            c = tuple(int(255 * v) for v in colorsys.hsv_to_rgb((ph2 + q * 0.5) % 1, 0.35, 1.0))
            pen.circle(px, py, size, c + (max(0, al),))
        ring(pen, 270, 430, lt, (255, 255, 255))
        draw_title(pen, t)
        pen.d.text((22 * LM, 880 * LM), "COLOR LAB", font=tag_font, fill=(255, 255, 255, 170))
        return im, layer

    # ---------------- Tiny Tales ----------------
    def tales_frame(t, q, si, lt):
        top = np.array(_lerp_c((26, 22, 78), (5, 7, 28), q)); bot = np.array(_lerp_c((150, 84, 150), (28, 28, 88), q))
        ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        grad = np.ascontiguousarray(np.broadcast_to(top * (1 - ys) + bot * ys, (H, W, 3))).astype(np.uint8)
        im = Image.fromarray(grad)
        layer = Image.new("RGBA", (int(W * 1.5), int(H * 1.5)), (0, 0, 0, 0)); pen = _Pen(layer, LM)
        glow = Image.new("RGBA", (W // 2, H // 2), (0, 0, 0, 0)); gp = _Pen(glow, GM)

        # moon + stars
        night = 0.3 + 0.7 * q
        pen.circle(430, 120, 36, (255, 247, 215, 255)); gp.circle(430, 120, 70, (255, 240, 190, 90))
        pen.circle(445, 112, 31, (0, 0, 0, 0))  # erase to leave a crescent
        for sx, sy, size, ph, sp in stars:
            x = (sx - t * 2.5 * size) % 540
            a = int(255 * night * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * sp + ph))))
            pen.circle(x, sy, size * 0.9, (255, 255, 235, a))
            if size > 1.9: pen.line([(x - size * 2.2, sy), (x + size * 2.2, sy)], (255, 255, 235, a // 2), 0.7); pen.line([(x, sy - size * 2.2), (x, sy + size * 2.2)], (255, 255, 235, a // 2), 0.7)
        k = int(t // 6); tt = t - 6 * k
        if tt < 0.9:
            r = random.Random(k * 7 + 3); x0 = 120 + r.random() * 400; y0 = 60 + r.random() * 160
            hx, hy = x0 - tt * 420, y0 + tt * 230
            for j in range(10):
                pen.circle(hx + j * 14, hy - j * 7.7, 2.6 - j * 0.22, (255, 255, 255, int(230 * (1 - j / 10) * (1 - tt / 0.9))))
            gp.circle(hx, hy, 9, (255, 255, 255, 120))

        # hills with parallax sway
        pts = [(x, back_y(x + math.sin(t * 0.3) * 4)) for x in range(-10, 560, 10)]
        pen.poly(pts + [(550, 960), (-10, 960)], (34, 28, 86, 255))
        pen.line(pts, (150, 130, 230, 90), 1.5)
        fpts = [(x, front_y(x)) for x in range(-10, 560, 10)]
        pen.poly(fpts + [(550, 960), (-10, 960)], (14, 15, 46, 255))
        pen.line(fpts, (110, 230, 210, 90), 1.5)
        for fxp, ph, ci in flowers:
            gy = front_y(fxp); sway = math.sin(t * 1.6 + ph) * 4; hgt = 22 + 10 * math.sin(ph * 3)
            pen.line([(fxp, gy + 4), (fxp + sway * 0.5, gy - hgt * 0.5), (fxp + sway, gy - hgt)], (60, 200, 150, 255), 2)
            c = petal_cols[ci]; pulse = 0.6 + 0.4 * math.sin(t * 2 + ph)
            pen.circle(fxp + sway, gy - hgt, 5.5, c + (255,)); pen.circle(fxp + sway, gy - hgt, 2.2, (255, 250, 200, 255))
            gp.circle(fxp + sway, gy - hgt, 12, c + (int(110 * pulse),))
        for fx0, fy0, ph, sp in flies:
            x = fx0 + 28 * math.sin(t * 0.5 * sp + ph); y = fy0 + 18 * math.cos(t * 0.7 * sp + ph * 2)
            a = 0.5 + 0.5 * math.sin(t * 2.5 * sp + ph * 3)
            pen.circle(x, y, 1.8, (230, 255, 150, int(255 * a))); gp.circle(x, y, 8, (200, 255, 120, int(150 * a)))

        # story beats
        flt = _smooth((q - 0.36) / 0.14) * (1 - _smooth((q - 0.78) / 0.14))
        rise = _smooth((q - 0.42) / 0.36)
        rx = 190 + flt * 60; ground = front_y(rx)
        ry = ground - 172 - flt * 330 + 6 * math.sin(t * 2.0)
        star_gx, star_gy = 410, front_y(410) - 38
        sx = _lerp(star_gx, 340, rise) + 8 * math.sin(t * 1.5) * rise
        sy = _lerp(star_gy, 190, rise) + 6 * math.cos(t * 1.8)
        bright = 0.5 + 0.5 * _smooth((q - 0.5) / 0.3)

        # star (little face, glow, trail)
        if 0.05 < rise < 0.98:
            for j in range(14):
                pen.circle(sx + math.sin(t * 5 + j) * 8, sy + 16 + j * 12, 4 - j * 0.25, (255, 230, 120, int(170 * (1 - j / 14))))
        gp.circle(sx, sy, 56 * (0.8 + 0.2 * math.sin(t * 3)), (255, 225, 100, int(190 * bright)))
        rot = t * 0.6; sp_pts = []
        for j in range(10):
            a = -math.pi / 2 + j * math.pi / 5 + math.sin(rot) * 0.12
            r = 34 if j % 2 == 0 else 15
            sp_pts.append((sx + math.cos(a) * r, sy + math.sin(a) * r))
        pen.poly(sp_pts, (255, int(200 + 30 * bright), int(70 + 60 * bright), 255))
        pen.circle(sx - 6, sy - 2, 2.4, (80, 50, 40, 255)); pen.circle(sx + 6, sy - 2, 2.4, (80, 50, 40, 255))
        pen.arc(sx, sy + 2, 6, 5, 10, 170, (80, 50, 40, 255), 1.6)

        # paper rocket carried toward the star
        if flt > 0.05 and rise < 0.97:
            kx = _lerp(rx + 100, sx - 30, rise); ky = _lerp(ry + 70, sy + 50, rise); ang = -0.5
            pr = [(kx + 26, ky), (kx - 14, ky - 11), (kx - 8, ky), (kx - 14, ky + 11)]
            pen.poly(pr, (250, 248, 255, 255)); pen.poly([(kx - 8, ky), (kx - 14, ky + 11), (kx + 26, ky)], (205, 215, 245, 255))
            for j in range(6): pen.circle(kx - 16 - j * 7, ky + math.sin(t * 20 + j) * 2, 4 - j * 0.5, (255, 190, 90, int(220 * (1 - j / 6))))

        # robot
        kk = 1.18; wave = 0.5 + 0.5 * (1 if q > 0.84 else 0.4)
        P = lambda dx, dy: (rx + dx * kk, ry + dy * kk)
        if flt > 0.05:   # thruster flame
            fl = 24 + 10 * math.sin(t * 30)
            pen.poly([P(-22, 150), P(22, 150), P(0, 150 + fl)], (255, 170, 70, 230)); pen.poly([P(-11, 150), P(11, 150), P(0, 150 + fl * 0.6)], (255, 240, 170, 255))
            gp.circle(*P(0, 175), 40, (255, 170, 80, 160))
        gp.ellipse(rx, ground + 4, 70, 11, (0, 0, 0, 0))
        pen.ellipse(rx, ground + 6, 62 * (1 - flt * 0.4), 9 * (1 - flt * 0.4), (0, 0, 0, int(110 * (1 - flt * 0.6))))
        arm_l = math.sin(t * 1.8) * 6
        pen.line([P(-50, 80), P(-78, 110 + arm_l), P(-84, 128 + arm_l)], (150, 215, 240, 255), 13 * kk)
        wv = math.sin(t * 7) * 16 * wave
        pen.line([P(50, 80), P(82, 70 - wave * 30), P(96 + wv * 0.6, 24 - wave * 36 + wv * 0.4)], (150, 215, 240, 255), 13 * kk)
        pen.circle(*P(-84, 130 + arm_l), 9 * kk, (200, 240, 252, 255)); pen.circle(*P(96 + wv * 0.6, 24 - wave * 36 + wv * 0.4), 9 * kk, (200, 240, 252, 255))
        for dx in (-26, 26): pen.rrect(*P(dx - 11, 146), *P(dx + 11, 164), 6, (120, 190, 220, 255))
        pen.rrect(*P(-52, 56), *P(52, 152), 26 * kk, (96, 200, 236, 255), (236, 250, 255, 255), 2.5)
        pen.rrect(*P(-34, 74), *P(34, 128), 14 * kk, (70, 160, 205, 255))
        pc = 0.6 + 0.4 * math.sin(t * 3)
        pen.circle(*P(0, 100), 12 * kk, (255, int(180 + 60 * pc), 110, 255)); gp.circle(*P(0, 100), 26, (255, 200, 120, int(190 * pc)))
        sway = math.sin(t * 2.2) * 6
        pen.line([P(0, -50), P(sway * 0.6, -66), P(sway, -84)], (190, 235, 250, 255), 4 * kk)
        pen.circle(*P(sway, -90), 9 * kk, (255, 120, 185, 255)); gp.circle(*P(sway, -90), 22, (255, 120, 185, int(120 + 100 * pc)))
        pen.rrect(*P(-60, -50), *P(60, 50), 30 * kk, (182, 234, 250, 255), (245, 255, 255, 255), 2.5)
        pen.rrect(*P(-46, -30), *P(46, 32), 22 * kk, (22, 30, 74, 255))
        bt = t % 3.4; blink = 0.12 if bt < 0.13 else 1.0
        lx, ly = sx - rx, sy - ry; nrm = math.hypot(lx, ly) or 1; lookx, looky = lx / nrm * 5, ly / nrm * 3
        for dx in (-22, 22):
            ex, ey = P(dx + lookx, -2 + looky)
            pen.ellipse(ex, ey, 10 * kk, 12 * kk * blink, (120, 240, 255, 255)); gp.circle(ex, ey, 18, (110, 235, 255, 150))
            if blink > 0.5: pen.circle(ex + 3, ey - 4, 3, (255, 255, 255, 255))
        mx, my = P(0, 22); pen.arc(mx, my, 11 * kk, 7 * kk, 15, 165, (120, 240, 255, 255), 2.2)
        pen.circle(*P(-40, 14), 6 * kk, (255, 130, 170, 90)); pen.circle(*P(40, 14), 6 * kk, (255, 130, 170, 90))

        ring(pen, sx, sy, lt, (255, 230, 140))
        draw_title(pen, t)
        pen.d.text((22 * LM, 880 * LM), "TINY TALES", font=tag_font, fill=(255, 255, 255, 170))

        gl = glow.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BICUBIC)
        base = Image.alpha_composite(im.convert("RGBA"), gl)
        return base, layer

    def _lerp_c(a, b, t): return tuple(_lerp(a[i], b[i], t) for i in range(3))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(CONTENT / f"{job_id}_background.mp4")]
    bg = CONTENT / f"{job_id}_background.mp4"
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for n in range(total):
            t = n / FPS; q = min(1.0, t / max(duration, 1)); si, lt = scene_at(t)
            im, layer = (lab_frame if series == "color_lab" else tales_frame)(t, q, si, lt)
            im = Image.alpha_composite(im.convert("RGBA"), layer.resize((W, H), Image.LANCZOS)).convert("RGB")
            bl = im.resize((W // 6, H // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BICUBIC).point(bloom_lut)
            im = ImageChops.screen(im, bl)
            z = 1.045 + 0.04 * q + 0.02 * math.exp(-lt * 5)
            cx, cy = W / 2 + 6 * math.sin(t * 0.5), H / 2 + 8 * math.cos(t * 0.4)
            im = im.resize((W, H), Image.BICUBIC, box=(cx - W / 2 / z, cy - H / 2 / z, cx + W / 2 / z, cy + H / 2 / z))
            arr = np.asarray(im, dtype=np.float32) * vig
            fade = min(1.0, t / 0.4, max(0.0, (duration - t) / 0.5 + 0.2))
            if fade < 1: arr = arr * fade
            proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    finally:
        try: proc.stdin.close()
        except Exception: pass
        err = proc.stderr.read().decode(errors="ignore"); proc.wait()
    if proc.returncode: raise RuntimeError("Background animation failed: " + err[-1000:])
    return bg

STOCK_DIR = ASSETS / "pixabay"
STOCK_DIR.mkdir(exist_ok=True)

def pixabay_search(query, key):
    """Search Pixabay videos; responses are cached for 24h as the API terms require."""
    import hashlib, time
    cache = STOCK_DIR / f"search_{hashlib.md5(query.encode()).hexdigest()}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        return json.loads(cache.read_text())
    r = requests.get("https://pixabay.com/api/videos/", params={"key": key, "q": query[:100], "per_page": 30, "safesearch": "true", "order": "popular"}, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    cache.write_text(json.dumps(hits))
    return hits

def hue_variety(hue, sat):
    """0..1: how many distinct hues are present among the saturated pixels (rich multi-color beats one flat tint)."""
    mask = sat > 0.35
    if mask.sum() < 50: return 0.0
    hist, _ = np.histogram(hue[mask], bins=12, range=(0, 1))
    return float((hist / hist.sum() > 0.05).sum() / 12)

def clip_vividness(h):
    """Score a clip by its thumbnail: bright, saturated footage wins over dark/muddy footage."""
    import io
    try:
        r = requests.get(h["videos"]["medium"]["thumbnail"], timeout=20); r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("HSV").resize((64, 64))
        px = np.asarray(im, dtype=np.float32) / 255
        sat, val = px[..., 1], px[..., 2]
        score = sat.mean() * 0.6 + val.std() * 0.5 + sat.std() * 0.3 + hue_variety(px[..., 0], sat) * 0.4
        if val.mean() > 0.82 or val.mean() < 0.22: score -= 0.5   # washed-out or near-black
        return float(score)
    except Exception:
        return 0.0

def fetch_stock_clip(queries, need, used, key, vivid=False):
    """Download (and keep on our server, per Pixabay's hotlinking rule) the best clip for the queries."""
    seen, hits = set(), []
    for q in queries:
        for h in pixabay_search(q, key)[:12]:
            if h["id"] not in used and h["id"] not in seen: seen.add(h["id"]); hits.append(h)
    if not hits: raise RuntimeError(f"No Pixabay footage found for '{queries[0]}'")
    if vivid:
        hits.sort(key=lambda h: (h["duration"] < need, -clip_vividness(h)))
    else:
        # long enough for the scene, then portrait ones, keeping Pixabay's popularity order
        hits.sort(key=lambda h: (h["duration"] < need, -aivideo.relevance(queries[0], h.get("tags", "")), h["videos"]["medium"]["height"] <= h["videos"]["medium"]["width"]))
    h = hits[0]; used.add(h["id"])
    v = h["videos"]["medium"] if 0 < h["videos"]["medium"]["size"] < 60_000_000 else h["videos"]["small"]
    path = STOCK_DIR / f"{h['id']}.mp4"
    if not path.exists():
        with requests.get(v["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
        path.with_suffix(".part").rename(path)
    return path

def make_stock_background(job_id, series, duration, scenes, key, ai_clips=None, sources=None):
    """Montage of Pixabay footage, one clip per scene, cropped to 9:16 with a soft grade and dip transitions."""
    cfg = sr.SERIES[series]; lab = cfg["vivid"]
    fallback = ["vibrant rainbow colorful liquid paint", "neon colorful abstract"] if lab else ["cinematic nature landscape", "city skyline timelapse"]
    prefix = cfg["stock_prefix"]
    used, segs = set(), []
    try:
        for i, ((a, b), sc) in enumerate(zip(scene_bounds(scenes, duration), scenes)):
            dur = b - a + (0.05 if i == len(scenes) - 1 else 0)
            query = (prefix + re.sub(r"[^\w\s]", " ", sc.get("stock_query") or sc.get("visual", ""))).strip()
            seg = CONTENT / f"{job_id}_seg{i}.mp4"
            # Only Color Lab drifts the hue over time (its rainbow theme); on realistic footage that would visibly recolour the subject.
            drift = ",hue=h=t*10:s=1.05" if series == "color_lab" else ""
            grade = (f"eq=saturation=1.3:contrast=1.12:brightness=0.02{drift},vignette=PI/6" if lab
                     else "eq=saturation=1.15:brightness=-0.05,vignette=PI/5")
            tail = f",fade=t=in:st=0:d=0.2,fade=t=out:st={max(0, dur - 0.2):.2f}:d=0.2,format=yuv420p"
            enc = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(seg)]
            if ai_clips and i in ai_clips:
                api = ("kling" if "_kl" in ai_clips[i].name else "apiframe" if "_af" in ai_clips[i].name else
                       "gemini" if "_gm" in ai_clips[i].name else None)   # native 9:16 clips from an API or Gemini
                # LTX clips are ~2 s: slow them a little (and blend frames). A single-shot video's API clip was generated in fast
                # motion, so it is slowed to fill the scene (up to SLOWMO_MAX x) with motion-compensated in-between frames.
                # All AI clips then play forward then backward (ping-pong) and loop, which only shows if the clip is still too short.
                pp = CONTENT / f"{job_id}_pp{i}.mp4"
                factor = 1.0
                if api and len(scenes) == 1:
                    try: factor = min(float(os.getenv("SLOWMO_MAX", "2.0")), max(1.0, dur / probe_duration(ai_clips[i])))
                    except Exception: factor = 1.0
                slow = (f"setpts={factor:.3f}*PTS," if factor > 1.05 else "") if api else "setpts=1.4*PTS,"
                r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ai_clips[i]), "-filter_complex",
                                    f"[0:v]{slow}split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[o]", "-map", "[o]", "-an", "-c:v", "libx264", "-crf", "14", str(pp)], capture_output=True, text=True)
                if not r.returncode:
                    smooth = "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1," if factor > 1.05 else "fps=30,"
                    vf = ("scale=720:1280:force_original_aspect_ratio=increase:flags=lanczos,crop=720:1280," + smooth if api else
                          "scale=720:1280:flags=lanczos,minterpolate=fps=30:mi_mode=blend,") + grade + tail
                    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(pp), "-t", f"{dur:.3f}", "-vf", vf] + enc, capture_output=True, text=True)
                pp.unlink(missing_ok=True)
                if not r.returncode:
                    segs.append(seg)
                    if sources is not None: sources.append(api or "ltx")
                    continue
                print(f"[ai video] scene {i} clip unusable, using stock footage: {r.stderr[-200:]}")
            clip = fetch_stock_clip([query] + fallback, dur, used, key, vivid=lab)
            vf = "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=30," + grade + tail
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(clip), "-t", f"{dur:.3f}", "-vf", vf] + enc, capture_output=True, text=True)
            if r.returncode: raise RuntimeError("Stock segment failed: " + r.stderr[-800:])
            segs.append(seg)
            if sources is not None: sources.append("stock")
        listing = CONTENT / f"{job_id}_segs.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in segs))
        bg = CONTENT / f"{job_id}_background.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(bg)], capture_output=True, text=True)
        if r.returncode: raise RuntimeError("Stock concat failed: " + r.stderr[-800:])
        return bg
    finally:
        for p in segs: p.unlink(missing_ok=True)
        (CONTENT / f"{job_id}_segs.txt").unlink(missing_ok=True)

def make_background(job_id, title, series, duration, scenes, ai_video=False, ai_provider="ltx", ai_mode="t2v"):
    """Per-scene visuals, best first: Kling or Apiframe clip (paid, if chosen) -> local LTX clip (if AI video requested) -> Pixabay
    stock footage -> drawn animation. Returns (video path, [source of each scene: "kling" | "apiframe" | "ltx" | "stock" | "drawn"])."""
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    visual = sr.SERIES[series]["visual"]
    if (visual == "stock" or (ai_video and ai_provider == "gemini")) and (key or ai_video):
        ai_clips = {}
        if ai_video:
            if ai_provider == "kling":
                ai_clips = kling.generate(job_id, scenes, key, log=print, mode=ai_mode)
            elif ai_provider == "apiframe":
                ai_clips = apiframe.generate(job_id, scenes, key, log=print, mode=ai_mode)
            elif ai_provider == "gemini":   # the clip you made in Gemini and uploaded
                clip = aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4"
                ai_clips = {0: clip} if clip.exists() else {}
            missing = {i for i in range(min(len(scenes), 6)) if i not in ai_clips}
            if missing and ai_provider != "gemini":   # local LTX covers whatever Kling/Apiframe did not (no key, clip cap reached, a failed clip); no-op if LTX is not set up
                ai_clips = {**aivideo.generate(job_id, scenes, key, log=print, only=missing, mode=ai_mode), **ai_clips}
        sources = []
        try: return make_stock_background(job_id, series, duration, scenes, key, ai_clips, sources), sources
        except Exception as e: print(f"[stock footage unavailable, using procedural animation] {e}")
    bg = make_animated_background(job_id, title, "tiny_tales" if visual == "procedural_tales" else "color_lab", duration, scenes)
    return bg, ["drawn"] * len(scenes)

def make_srt(scenes, duration):
    # Short 1-4 word caption chunks (breaking at punctuation), timed by character count within each scene.
    chunks = []
    for (a, b), sc in zip(scene_bounds(scenes, duration), scenes):
        if "_speech" in sc: a, b = sc["_speech"]   # captions follow the spoken words, not the quiet that follows them
        group, phrases = [], []
        for w in sc["narration"].split():
            group.append(w)
            if len(group) >= 4 or re.search(r"[,.!?;:]$", w): phrases.append(" ".join(group)); group = []
        if group: phrases.append(" ".join(group))
        total = sum(len(p) for p in phrases) or 1; cur = a
        for p in phrases:
            end = cur + (b - a) * len(p) / total
            chunks.append((cur, end, re.sub(r"[,;:]$", "", p))); cur = end
    def stamp(sec):
        ms = int(round(sec*1000)); hh, rem = divmod(ms, 3600000); mm, rem = divmod(rem, 60000); ss, ms = divmod(rem, 1000)
        return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"
    body = "\n".join(f"{i}\n{stamp(s0)} --> {stamp(s1)}\n{text}\n" for i, (s0, s1, text) in enumerate(chunks, 1))
    p = CONTENT / f"{uuid.uuid4().hex}_captions.srt"
    p.write_text(body, encoding="utf-8")
    return p

VOICES = {"English": "en-US-AriaNeural", "Hindi": "hi-IN-SwaraNeural", "Tamil": "ta-IN-PallaviNeural"}

def tts_voice(language="English"):
    return os.getenv("TTS_VOICE") or VOICES.get(language, "en-US-AriaNeural")

def synthesize_padded(scenes, target, audio_path, language="English"):
    """Visual-first formats: each scene's line is spoken at the start of its own slot, then quiet, so the video lasts the
    chosen duration instead of the few seconds the sparse narration takes. Sets each scene's _slot and _speech times."""
    parts, cur = [], 0.0
    try:
        for i, sc in enumerate(scenes):
            part = audio_path.with_name(f"{audio_path.stem}_s{i}.mp3")
            asyncio.run(synthesize(sc["narration"].strip(), part, language)); parts.append(part)
            speech = probe_duration(part)
            slot = max(speech + 0.6, target / len(scenes))
            sc["_speech"], sc["_slot"] = (cur, cur + speech), (cur, cur + slot); cur += slot
        ins = sum([["-i", str(p)] for p in parts], [])
        pads = "".join(f"[{i}:a]aresample=24000,apad=whole_dur={sc['_slot'][1] - sc['_slot'][0]:.3f}[a{i}];" for i, sc in enumerate(scenes))
        graph = pads + "".join(f"[a{i}]" for i in range(len(scenes))) + f"concat=n={len(scenes)}:v=0:a=1[o]"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + ins + ["-filter_complex", graph, "-map", "[o]", "-c:a", "libmp3lame", "-b:a", "128k", str(audio_path)], capture_output=True, text=True)
        if r.returncode: raise RuntimeError("Voice-over padding failed: " + r.stderr[-800:])
    finally:
        for p in parts: p.unlink(missing_ok=True)

async def synthesize(text, audio_path, language="English"):
    voice = tts_voice(language)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(audio_path))

def probe_duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], text=True)
    return max(1.0, float(out.strip()))

def script_info(data):
    """Which model wrote the script (older jobs only have the 'provider:model' string)."""
    prov, model = data.get("script_provider"), data.get("script_model")
    if not prov and data.get("script_source"):
        prov, _, model = data["script_source"].partition(":")
    return {"provider": prov, "model": model or None, "fell_back_from": data.get("script_fell_back_from")}

def ffmpeg_version():
    try: return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10).stdout.split()[2]
    except Exception: return None

def mix_clip_sound(job_id, narration_audio, duration):
    """Keep the Gemini clip's own sound effects (the point of an ASMR shot) under the narration. The sound is slowed by the same
    factor as the picture, with its pitch kept. Returns the mixed track, or the narration alone if the clip has no sound."""
    clip = aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4"
    try:
        factor = min(float(os.getenv("SLOWMO_MAX", "2.0")), max(1.0, duration / probe_duration(clip)))
        out = CONTENT / f"{job_id}_mix.mp3"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(narration_audio), "-i", str(clip), "-filter_complex",
                            f"[1:a]atempo={1 / factor:.4f},volume={os.getenv('CLIP_SOUND_VOLUME', '0.8')}[s];[0:a][s]amix=inputs=2:duration=first:normalize=0[o]",
                            "-map", "[o]", "-c:a", "libmp3lame", "-b:a", "160k", str(out)], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-300:])
        return out
    except Exception as e:
        print(f"[gemini] clip sound not used: {str(e)[:200]}"); return narration_audio

def render_job(job):
    data = json.loads(job["script_json"])
    narration = " ".join(s["narration"].strip() for s in data["scenes"])
    audio = CONTENT / f"{job['id']}.mp3"
    if sr.is_visual(data.get("series", "")) or data.get("single_shot"):
        synthesize_padded(data["scenes"], float(data.get("duration_seconds", 30)), audio, data.get("language", "English"))
    else:
        asyncio.run(synthesize(narration, audio, data.get("language", "English")))
    duration = probe_duration(audio)
    srt = make_srt(data["scenes"], duration)
    series = data.get("series", "tiny_tales")
    bg, sources = make_background(job["id"], data["title"], series, duration, data["scenes"], bool(data.get("ai_video")), data.get("ai_provider", "ltx"), data.get("ai_mode", "t2v"))
    if sources == ["gemini"]: audio = mix_clip_sound(job["id"], audio, duration)
    video = CONTENT / f"{job['id']}.mp4"
    # FFmpeg burns captions into a 9:16 animated background and narration.
    srt_arg = str(srt).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    # NOTE: ffmpeg converts SRT to ASS using a 384x288 reference resolution, so
    # FontSize/MarginV/MarginL/MarginR here are in THAT space, not in 1080x1920
    # pixels. libass then scales by 1920/288 = 6.67x. The previous values
    # (FontSize=18, MarginV=250) rendered as 120px text pushed ~1667px up from
    # the bottom, i.e. off the top of the frame. These are the pixel targets
    # divided by 6.67.
    fonts = str(ASSETS / "fonts").replace("\\", "/").replace(":", "\\:")
    # Poppins ExtraBold, warm yellow with a thick dark outline + soft shadow (sizes are in libass's 288-high space).
    vf = f"scale=1080:1920:flags=lanczos,subtitles='{srt_arg}':fontsdir='{fonts}':force_style='FontName=Poppins ExtraBold,FontSize=12,PrimaryColour=&H0000E8FF,OutlineColour=&H00280F1E,BackColour=&H80000000,BorderStyle=1,Outline=3.2,Shadow=1.5,Alignment=2,MarginV=46,MarginL=14,MarginR=14'"
    cmd = ["ffmpeg", "-y", "-i", str(bg), "-i", str(audio), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("FFmpeg failed: " + result.stderr[-1800:])
    counts = {k: sources.count(k) for k in ("kling", "apiframe", "gemini", "ltx", "stock", "drawn") if k in sources}
    provenance = {
        "script": script_info(data),
        "voice": {"engine": "edge-tts", "voice": tts_voice(data.get("language", "English")), "language": data.get("language", "English")},
        "visuals": {"sources": sources, "counts": counts, "ai_requested": bool(data.get("ai_video")), "ai_provider": data.get("ai_provider", "ltx") if data.get("ai_video") else None, "ai_mode": data.get("ai_mode", "t2v") if data.get("ai_video") else None},
        "assembly": {"tool": "ffmpeg", "version": ffmpeg_version(), "resolution": "1080x1920", "fps": 30},
        "rendered_at": now()}
    return video, audio, srt, provenance

def create_job(series, niche, topic, language, duration, made_for_kids=None, source="manual", ai_video=False, ai_provider="ltx", ai_mode="t2v", avoid=()):
    """Write a script with the LLM and store a script_ready job. Returns (job_id, script)."""
    if series not in sr.SERIES: raise HTTPException(422, f"Unknown series '{series}'")
    job_id = uuid.uuid4().hex[:12]
    con = db()
    con.execute("INSERT INTO jobs(id,created_at,updated_at,niche,status,series,topic,source) VALUES(?,?,?,?,?,?,?,?)", (job_id, now(), now(), niche, "generating", series, topic, source)); con.commit()
    try:
        single = bool(ai_video) and ai_provider == "gemini"   # Gemini makes one clip, so the video is one continuous shot
        data = make_script(series, niche, topic, language, duration, single, avoid)
        if single: data["single_shot"] = True; data["gemini_prompt"] = sr.gemini_prompt({**data, "series": series})
        data["made_for_kids"] = sr.SERIES[series]["kids"] if made_for_kids is None else made_for_kids
        data["duration_seconds"] = duration; data["language"] = language; data["ai_video"] = bool(ai_video); data["ai_provider"] = ai_provider; data["ai_mode"] = ai_mode
        con.execute("UPDATE jobs SET updated_at=?,status='script_ready',title=?,description=?,hashtags=?,script_json=? WHERE id=?", (now(), data.get("title", "AI short"), data.get("description", ""), json.dumps(data.get("hashtags", [])), json.dumps(data, ensure_ascii=False), job_id)); con.commit()
        return job_id, data
    except Exception as e:
        con.execute("UPDATE jobs SET updated_at=?,status='failed',attempts=attempts+1,error=? WHERE id=?", (now(), str(e)[:2000], job_id)); con.commit()
        raise
    finally: con.close()

RENDER_LOCK = threading.Lock()   # one heavy render at a time (manual and autopilot share it)

def do_render(job_id):
    con = db(); job = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job: con.close(); raise HTTPException(404, "Job not found")
    if job["status"] not in ("script_ready", "render_failed", "failed"):
        con.close(); raise HTTPException(409, f"Cannot render job in status {job['status']}")
    con.execute("UPDATE jobs SET status='rendering',updated_at=?,error=NULL WHERE id=?", (now(), job_id)); con.commit()
    try:
        with RENDER_LOCK:
            last_error = None
            for attempt in range(2):   # retry once for transient TTS/FFmpeg failures
                try:
                    fresh = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                    video, audio, srt, prov = render_job(fresh)
                    con.execute("UPDATE jobs SET status='pending_review',updated_at=?,attempts=attempts+?,video_path=?,audio_path=?,srt_path=?,provenance=?,error=NULL WHERE id=?", (now(), attempt, str(video), str(audio), str(srt), json.dumps(prov), job_id)); con.commit()
                    return {"id": job_id, "status": "pending_review", "video_url": f"/video/{job_id}", "message": "Video rendered. Review it before approving.", "provenance": prov}
                except Exception as e: last_error = e
            raise last_error or RuntimeError("render failed")
    except Exception as e:
        con.execute("UPDATE jobs SET status='render_failed',updated_at=?,attempts=attempts+1,error=? WHERE id=?", (now(), str(e)[:2500], job_id)); con.commit()
        raise HTTPException(500, f"Render failed: {str(e)[:700]}")
    finally: con.close()

@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": shutil.which("ffmpeg") is not None, "ffprobe": shutil.which("ffprobe") is not None, "llm": llm.status(), "stock_footage": bool(os.getenv("PIXABAY_API_KEY")), "youtube_data": google_yt.auth_mode(), "ai_video": aivideo.status(), "kling": kling.status(), "apiframe": apiframe.status(), "db": str(DB_PATH)}

@app.get("/series")
def list_series(): return sr.label_choices()

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        job_id, data = create_job(req.series, req.niche, req.topic, req.language, req.duration_seconds, req.made_for_kids, ai_video=req.ai_video, ai_provider=req.ai_provider, ai_mode=req.ai_mode, avoid=[a[:120] for a in req.avoid])
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Script generation failed. Is Ollama running (ollama serve) with a model pulled? Details: {str(e)[:400]}")
    return {"id": job_id, "status": "script_ready", "script": data}

@app.post("/revise/{job_id}")
def revise(job_id: str, req: ReviseRequest):
    """Rewrite a script that is waiting for approval, following the user's instructions. The old script is kept if this fails."""
    con = db(); r = con.execute("SELECT status,series,script_json FROM jobs WHERE id=?", (job_id,)).fetchone(); con.close()
    if not r: raise HTTPException(404, "Job not found")
    if r["status"] != "script_ready": raise HTTPException(409, f"Only a script waiting for approval can be revised (current: {r['status']})")
    old = json.loads(r["script_json"] or "{}")
    try: new = sr.revise_script(r["series"], old, req.instructions.strip(), old.get("language", "English"), int(old.get("duration_seconds", 30)))
    except Exception as e: raise HTTPException(500, f"Revision failed, the previous script is unchanged. Details: {str(e)[:400]}")
    keep = ("made_for_kids", "duration_seconds", "language", "ai_video", "ai_provider", "ai_mode")
    new.update({k: old[k] for k in keep if k in old})
    new["script_source"] = f"{llm.LAST['provider']}:{llm.LAST['model']}"
    new["script_provider"], new["script_model"], new["script_fell_back_from"] = llm.LAST["provider"], llm.LAST["model"], llm.LAST["fell_back_from"]
    new["revisions"] = old.get("revisions", []) + [req.instructions.strip()]
    if old.get("single_shot"):   # a revised script needs a fresh Gemini clip, so an earlier upload is dropped
        new["single_shot"] = True; new["gemini_prompt"] = sr.gemini_prompt({**new, "series": r["series"]})
        (aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4").unlink(missing_ok=True)
    con = db()
    con.execute("UPDATE jobs SET updated_at=?,title=?,description=?,hashtags=?,script_json=? WHERE id=? AND status='script_ready'", (now(), new.get("title", "AI short"), new.get("description", ""), json.dumps(new.get("hashtags", [])), json.dumps(new, ensure_ascii=False), job_id)); con.commit(); con.close()
    return {"id": job_id, "status": "script_ready", "script": new}

GEMINI_CLIP_MAX_MB = 300

@app.post("/clip/{job_id}")
async def upload_clip(job_id: str, request: Request):
    """Attach the clip you generated in Gemini (raw video bytes in the body) to a script that is waiting for approval."""
    con = db(); r = con.execute("SELECT status,script_json FROM jobs WHERE id=?", (job_id,)).fetchone(); con.close()
    if not r: raise HTTPException(404, "Job not found")
    data = json.loads(r["script_json"] or "{}")
    if r["status"] not in ("script_ready", "render_failed") or not data.get("single_shot"):
        raise HTTPException(409, "Clips can only be uploaded for a Gemini script that has not been rendered yet")
    body = await request.body()
    if not body: raise HTTPException(422, "Empty upload")
    if len(body) > GEMINI_CLIP_MAX_MB * 1024 * 1024: raise HTTPException(413, f"Clip is larger than {GEMINI_CLIP_MAX_MB} MB")
    aivideo.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out = aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4"; tmp = out.with_suffix(".upload")
    tmp.write_bytes(body)
    try:
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(tmp)], text=True))
        streams = [s["codec_type"] for s in info.get("streams", [])]
        dur = float(info["format"]["duration"])
        if "video" not in streams or not 1 <= dur <= 60: raise ValueError
    except Exception:
        tmp.unlink(missing_ok=True); raise HTTPException(422, "That file is not a playable video of 1-60 seconds")
    tmp.replace(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"id": job_id, "seconds": round(dur, 2), "width": v.get("width"), "height": v.get("height"), "has_audio": "audio" in streams}

@app.get("/clip/{job_id}")
def clip_info(job_id: str):
    out = aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4"
    if not out.exists(): return {"id": job_id, "uploaded": False}
    return {"id": job_id, "uploaded": True, "seconds": round(probe_duration(out), 2)}

@app.post("/render/{job_id}")
def render(job_id: str):
    con = db(); r = con.execute("SELECT script_json FROM jobs WHERE id=?", (job_id,)).fetchone(); con.close()
    if r and json.loads(r["script_json"] or "{}").get("ai_provider") == "gemini" and json.loads(r["script_json"] or "{}").get("single_shot") \
            and not (aivideo.CLIP_DIR / f"{job_id}_gm_t2v0.mp4").exists():
        raise HTTPException(409, "Upload the clip you made in Gemini first")
    return do_render(job_id)

@app.get("/jobs")
def jobs():
    con=db(); rows=con.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall(); con.close()
    return [rowdict(r) for r in rows]

@app.get("/jobs/{job_id}")
def get_job(job_id:str):
    con=db(); r=con.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone(); con.close()
    if not r: raise HTTPException(404,"Job not found")
    return rowdict(r)

@app.get("/video/{job_id}")
def video(job_id:str):
    con=db(); r=con.execute("SELECT video_path FROM jobs WHERE id=?",(job_id,)).fetchone(); con.close()
    if not r or not r["video_path"] or not Path(r["video_path"]).exists(): raise HTTPException(404,"Rendered video not found")
    return FileResponse(r["video_path"],media_type="video/mp4",filename=f"{job_id}.mp4")

@app.post("/approve/{job_id}")
def approve(job_id:str, action:JobAction):
    con=db(); r=con.execute("SELECT status FROM jobs WHERE id=?",(job_id,)).fetchone()
    if not r: con.close(); raise HTTPException(404,"Job not found")
    if r["status"] != "pending_review": con.close(); raise HTTPException(409,f"Only pending_review jobs can be approved (current: {r['status']})")
    con.execute("UPDATE jobs SET status='approved',updated_at=? WHERE id=?",(now(),job_id)); con.commit(); con.close()
    return {"id":job_id,"status":"approved","next":"Call /publish/{job_id}/youtube or /publish/{job_id}/instagram after configuring credentials."}

def licence_credit(job) -> str:
    """For a re-posted creator video: refuse unless permission is on record, and return the credit line that must be published with it."""
    lic = (json.loads(job["script_json"] or "{}") or {}).get("licensed")
    if not lic: return ""
    perm = lic.get("permission") or {}
    if not (perm.get("how") and perm.get("date")):
        raise HTTPException(409, "This is another creator's video and no permission is recorded for it. Record their permission in the Download tab first.")
    return f"🎥 Video by @{lic.get('handle') or 'the original creator'} on Instagram, re-posted with permission.\nOriginal: {lic.get('post', '')}"

class PublishOptions(BaseModel):
    confirm: bool = True
    publish_at: str | None = None   # ISO time; uploads now as private and YouTube makes it public at this time

@app.post("/publish/{job_id}/youtube")
def publish_youtube(job_id:str, opts: PublishOptions | None = None):
    opts = opts or PublishOptions()
    con=db(); job=con.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
    if not job: con.close(); raise HTTPException(404,"Job not found")
    if job["status"] not in ("approved", "published", "publish_failed"): con.close(); raise HTTPException(409,"Approve the video in the dashboard first")
    if job["youtube_id"]: con.close(); raise HTTPException(409,"This job already has a YouTube video ID; refusing to publish a duplicate")
    try: credit = licence_credit(job)
    except HTTPException: con.close(); raise
    when = None
    if opts.publish_at:
        try: when = datetime.fromisoformat(opts.publish_at.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError: con.close(); raise HTTPException(422, "publish_at must be an ISO date-time")
        if when.timestamp() < time.time() + 15 * 60: con.close(); raise HTTPException(422, "Schedule at least 15 minutes ahead")
    secrets=ROOT / os.getenv("YOUTUBE_CLIENT_SECRETS", "credentials/client_secret.json")
    token=ROOT / os.getenv("YOUTUBE_TOKEN_FILE", "credentials/youtube_token.json")
    if not secrets.exists(): con.close(); raise HTTPException(400,"Missing credentials/client_secret.json. See README for Google OAuth setup.")
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
        creds=None
        if token.exists(): creds=Credentials.from_authorized_user_file(str(token),scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token: creds.refresh(Request())
            else:
                flow=InstalledAppFlow.from_client_secrets_file(str(secrets),scopes)
                creds=flow.run_local_server(port=0)
            token.parent.mkdir(exist_ok=True); token.write_text(creds.to_json()); token.chmod(0o600)
        youtube=build("youtube","v3",credentials=creds)
        desc=(job["description"] or "")+("\n\n"+credit if credit else "")+"\n\n"+" ".join("#"+x.lstrip("#") for x in json.loads(job["hashtags"] or "[]"))
        status={"privacyStatus":"private","publishAt":when.strftime("%Y-%m-%dT%H:%M:%SZ")} if when else {"privacyStatus":"public"}
        body={"snippet":{"title":job["title"][:100],"description":desc[:4900],"tags":[x.lstrip("#") for x in json.loads(job["hashtags"] or "[]")],"categoryId":"24" if credit else "28"},"status":{**status,"selfDeclaredMadeForKids":bool(json.loads(job["script_json"] or "{}").get("made_for_kids", True))}}
        result=youtube.videos().insert(part="snippet,status",body=body,media_body=MediaFileUpload(job["video_path"],mimetype="video/mp4",resumable=True)).execute()
        con.execute("UPDATE jobs SET status='published',updated_at=?,youtube_id=?,publish_at=?,error=NULL WHERE id=?",(now(),result["id"],when.isoformat() if when else None,job_id)); con.commit()
        return {"id":job_id,"status":"published","youtube_video_id":result["id"],"url":f"https://youtube.com/shorts/{result['id']}","publish_at":when.isoformat() if when else None}
    except Exception as e:
        con.execute("UPDATE jobs SET status='publish_failed',updated_at=?,attempts=attempts+1,error=? WHERE id=?",(now(),str(e)[:2000],job_id)); con.commit()
        raise HTTPException(500,f"YouTube publish failed: {str(e)[:600]}")
    finally: con.close()

@app.post("/publish/{job_id}/instagram")
def publish_instagram(job_id:str):
    con=db(); job=con.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
    if not job: con.close(); raise HTTPException(404,"Job not found")
    if job["status"] not in ("approved", "published", "publish_failed"): con.close(); raise HTTPException(409,"Approve the video in the dashboard first")
    if job["instagram_id"]: con.close(); raise HTTPException(409,"This job already has an Instagram media ID; refusing to publish a duplicate")
    base=os.getenv("MEDIA_PUBLIC_BASE_URL", "").rstrip("/")
    ig_id=os.getenv("INSTAGRAM_USER_ID", "")
    token=os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
    version=os.getenv("META_GRAPH_VERSION", "v23.0")
    if not all((base,ig_id,token)):
        con.close(); raise HTTPException(400,"Set MEDIA_PUBLIC_BASE_URL, INSTAGRAM_USER_ID, and INSTAGRAM_ACCESS_TOKEN. Instagram must fetch the MP4 from a public HTTPS URL.")
    if "localhost" in base or "127.0.0.1" in base:
        con.close(); raise HTTPException(400,"MEDIA_PUBLIC_BASE_URL must be publicly reachable HTTPS, not localhost")
    video_url=f"{base}/video/{job_id}"
    try: credit = licence_credit(job)
    except HTTPException: con.close(); raise
    caption=(job["description"] or "")+("\n\n"+credit if credit else "")+"\n\n"+" ".join("#"+x.lstrip("#") for x in json.loads(job["hashtags"] or "[]"))
    try:
        root=f"https://graph.facebook.com/{version}"
        r=requests.post(f"{root}/{ig_id}/media",data={"media_type":"REELS","video_url":video_url,"caption":caption,"share_to_feed":"true","access_token":token},timeout=45); r.raise_for_status(); container_id=r.json()["id"]
        status=None
        for _ in range(18):
            import time; time.sleep(5)
            sr=requests.get(f"{root}/{container_id}",params={"fields":"status_code,status","access_token":token},timeout=20); sr.raise_for_status(); status=sr.json()
            if status.get("status_code")=="FINISHED": break
            if status.get("status_code")=="ERROR": raise RuntimeError(status.get("status","Instagram media processing error"))
        if not status or status.get("status_code")!="FINISHED": raise RuntimeError(f"Instagram media processing not finished: {status}")
        pr=requests.post(f"{root}/{ig_id}/media_publish",data={"creation_id":container_id,"access_token":token},timeout=45); pr.raise_for_status(); media_id=pr.json()["id"]
        con.execute("UPDATE jobs SET status='published',updated_at=?,instagram_id=? WHERE id=?",(now(),media_id,job_id)); con.commit()
        return {"id":job_id,"status":"published","instagram_media_id":media_id}
    except Exception as e:
        con.execute("UPDATE jobs SET status='publish_failed',updated_at=?,attempts=attempts+1,error=? WHERE id=?",(now(),str(e)[:2000],job_id)); con.commit()
        raise HTTPException(500,f"Instagram publish failed: {str(e)[:600]}")
    finally: con.close()


# ---------------------------------------------------------------- ideas, autopilot, analytics
TASKS: "queue.Queue[tuple]" = queue.Queue()
AP = {"current": None, "log": []}
AP_DEFAULTS = {"enabled": "0", "interval_hours": "24", "per_run": "3", "series": json.dumps(["mind_blown", "tech_tips", "color_lab", "tiny_tales"]), "duration": "30", "max_pending": "12", "ai_video": "0", "ai_provider": "ltx", "ai_mode": "t2v"}

def ap_log(msg):
    AP["log"] = (AP["log"] + [f"{datetime.now():%H:%M:%S}  {msg}"])[-80:]

def ap_config():
    c = {k: get_setting("autopilot_" + k, v) for k, v in AP_DEFAULTS.items()}
    return {"enabled": c["enabled"] == "1", "interval_hours": float(c["interval_hours"]), "per_run": int(c["per_run"]),
            "series": json.loads(c["series"]), "duration": int(c["duration"]), "max_pending": int(c["max_pending"]), "ai_video": c["ai_video"] == "1", "ai_provider": c["ai_provider"], "ai_mode": c["ai_mode"]}

def topics_in_use(series_key):
    con = db()
    rows = [r["topic"] for r in con.execute("SELECT topic FROM ideas WHERE series=?", (series_key,))]
    rows += [r["t"] for r in con.execute("SELECT COALESCE(topic,title) t FROM jobs WHERE series=? AND COALESCE(topic,title) IS NOT NULL", (series_key,))]
    con.close()
    return [t for t in rows if t]

def fill_idea_bank(series_key, n=8):
    added = []
    for _ in range(3):   # small models often return fewer ideas than asked; ask again for the remainder
        if len(added) >= n: break
        got = sr.brainstorm(series_key, n - len(added), topics_in_use(series_key))
        con = db()
        for t in got: con.execute("INSERT INTO ideas(series,topic,created_at) VALUES(?,?,?)", (series_key, t, now()))
        con.commit(); con.close(); added += got
    return added

def next_idea(series_key):
    for _ in range(2):
        con = db(); r = con.execute("SELECT * FROM ideas WHERE series=? AND status='new' ORDER BY id LIMIT 1", (series_key,)).fetchone(); con.close()
        if r: return dict(r)
        ap_log(f"Idea bank for {series_key} is empty - brainstorming")
        fill_idea_bank(series_key)
    raise RuntimeError("Could not get ideas from the LLM")

def produce(series_key, idea, duration, source="auto"):
    con = db(); con.execute("UPDATE ideas SET status='used' WHERE id=?", (idea["id"],)); con.commit(); con.close()
    try: job_id, data = create_job(series_key, "auto", idea["topic"], "English", duration, source=source, ai_video=ap_config()["ai_video"], ai_provider=ap_config()["ai_provider"], ai_mode=ap_config()["ai_mode"])
    except Exception:
        con = db(); con.execute("UPDATE ideas SET status='failed' WHERE id=?", (idea["id"],)); con.commit(); con.close(); raise
    con = db(); con.execute("UPDATE ideas SET job_id=? WHERE id=?", (job_id, idea["id"])); con.commit(); con.close()
    ap_log(f"Script ready: {data['title']}  [{series_key}]")
    do_render(job_id)
    ap_log(f"Rendered, waiting for your review: {data['title']}")
    return job_id

def run_batch(n, series_list, duration):
    cfg = ap_config()
    con = db(); pending = con.execute("SELECT COUNT(*) c FROM jobs WHERE status='pending_review'").fetchone()["c"]; con.close()
    if pending >= cfg["max_pending"]:
        ap_log(f"Skipped: {pending} videos are already waiting for review (limit {cfg['max_pending']}). Review some first."); return
    made = 0
    for i in range(n):
        series_key = series_list[i % len(series_list)]
        try:
            AP["current"] = f"{i + 1}/{n} {series_key}"
            produce(series_key, next_idea(series_key), duration)
            made += 1
        except Exception as e:
            ap_log(f"FAILED {series_key}: {getattr(e, 'detail', e)}"[:300])
    ap_log(f"Batch finished: {made}/{n} videos ready for review")

def worker_loop():
    while True:
        task = None
        try: task = TASKS.get(timeout=20)
        except queue.Empty:
            try:
                cfg = ap_config(); last = get_setting("autopilot_last_run")
                if cfg["enabled"] and (not last or (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() >= cfg["interval_hours"] * 3600):
                    task = ("batch", cfg["per_run"])
            except Exception as e: ap_log(f"scheduler error: {e}")
        if not task: continue
        try:
            cfg = ap_config()
            if task[0] == "batch":
                set_setting("autopilot_last_run", now()); ap_log(f"Autopilot batch started ({task[1]} videos)")
                run_batch(task[1], cfg["series"] or ["mind_blown"], cfg["duration"])
            elif task[0] == "idea":
                con = db(); r = con.execute("SELECT * FROM ideas WHERE id=?", (task[1],)).fetchone(); con.close()
                if r:
                    AP["current"] = f"idea: {r['topic']}"
                    produce(r["series"], dict(r), cfg["duration"], source="idea")
        except Exception as e:
            ap_log(f"FAILED: {getattr(e, 'detail', e)}"[:300])
        finally: AP["current"] = None

class IdeaGen(BaseModel):
    series: str
    n: int = Field(default=8, ge=1, le=20)

class IdeaAdd(BaseModel):
    series: str
    topic: str = Field(min_length=4, max_length=120)

class AutopilotConfig(BaseModel):
    enabled: bool | None = None
    interval_hours: float | None = Field(default=None, ge=0.25, le=168)
    per_run: int | None = Field(default=None, ge=1, le=20)
    series: list[str] | None = None
    duration: int | None = Field(default=None, ge=15, le=180)
    max_pending: int | None = Field(default=None, ge=1, le=100)
    ai_video: bool | None = None
    ai_provider: str | None = Field(default=None, pattern="^(ltx|kling|apiframe)$")
    ai_mode: str | None = Field(default=None, pattern="^(t2v|i2v)$")

@app.get("/ideas")
def list_ideas(series: str | None = None):
    con = db()
    rows = con.execute("SELECT * FROM ideas WHERE (?1 IS NULL OR series=?1) ORDER BY id DESC LIMIT 300", (series,)).fetchall(); con.close()
    return [dict(r) for r in rows]

@app.post("/ideas/generate")
def ideas_generate(req: IdeaGen):
    if req.series not in sr.SERIES: raise HTTPException(422, "Unknown series")
    try: return {"added": fill_idea_bank(req.series, req.n)}
    except Exception as e: raise HTTPException(500, f"Idea generation failed: {str(e)[:400]}")

@app.post("/ideas")
def ideas_add(req: IdeaAdd):
    if req.series not in sr.SERIES: raise HTTPException(422, "Unknown series")
    con = db(); con.execute("INSERT INTO ideas(series,topic,created_at) VALUES(?,?,?)", (req.series, req.topic.strip(), now())); con.commit(); con.close()
    return {"ok": True}

@app.delete("/ideas/{idea_id}")
def ideas_delete(idea_id: int):
    con = db(); con.execute("DELETE FROM ideas WHERE id=?", (idea_id,)); con.commit(); con.close(); return {"ok": True}

@app.post("/ideas/{idea_id}/produce")
def ideas_produce(idea_id: int):
    con = db(); r = con.execute("SELECT status FROM ideas WHERE id=?", (idea_id,)).fetchone(); con.close()
    if not r: raise HTTPException(404, "Idea not found")
    if r["status"] != "new": raise HTTPException(409, "Idea already used")
    TASKS.put(("idea", idea_id)); ap_log(f"Queued idea #{idea_id}")
    return {"queued": True}

@app.get("/autopilot")
def autopilot_status():
    cfg = ap_config(); last = get_setting("autopilot_last_run"); nxt = None
    if cfg["enabled"] and last: nxt = datetime.fromisoformat(last).timestamp() + cfg["interval_hours"] * 3600
    return {"config": cfg, "last_run": last, "next_run": datetime.fromtimestamp(nxt, timezone.utc).isoformat() if nxt else None,
            "current": AP["current"], "queued": TASKS.qsize(), "log": AP["log"][-30:]}

@app.post("/autopilot")
def autopilot_set(c: AutopilotConfig):
    for k, v in c.model_dump(exclude_none=True).items():
        if k == "series":
            bad = [x for x in v if x not in sr.SERIES]
            if bad or not v: raise HTTPException(422, f"Unknown or empty series list: {bad}")
            v = json.dumps(v)
        elif k in ("enabled", "ai_video"): v = "1" if v else "0"
        set_setting("autopilot_" + k, v)
    return autopilot_status()

@app.post("/autopilot/run")
def autopilot_run(count: int | None = None):
    n = count or ap_config()["per_run"]
    TASKS.put(("batch", n)); ap_log(f"Queued a batch of {n} videos")
    return {"queued": n}

@app.post("/reject/{job_id}")
def reject(job_id: str):
    con = db(); r = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not r: con.close(); raise HTTPException(404, "Job not found")
    if r["status"] in ("rendering", "generating", "published"): con.close(); raise HTTPException(409, f"Cannot reject a {r['status']} job")
    con.execute("UPDATE jobs SET status='rejected',updated_at=? WHERE id=?", (now(), job_id)); con.commit(); con.close()
    return {"id": job_id, "status": "rejected"}

@app.get("/analytics")
def analytics_summary():
    con = db()
    try: return analytics.summary(con)
    finally: con.close()

@app.post("/analytics/refresh")
def analytics_refresh():
    con = db()
    try: return analytics.refresh(con, "", os.getenv("INSTAGRAM_ACCESS_TOKEN", ""), os.getenv("META_GRAPH_VERSION", "v23.0"))
    finally: con.close()


# ---------------------------------------------------------------- niche research
class NicheResearch(BaseModel):
    keywords: list[str] | None = None
    region: str = Field(default="IN", min_length=2, max_length=2)
    days: int = Field(default=30, ge=7, le=90)

class NicheProfile(BaseModel):
    profile: str = ""
    n: int = Field(default=5, ge=1, le=8)

class NicheAdopt(BaseModel):
    index: int = Field(ge=0)

def niche_slug(name): return "n_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]

@app.get("/niche")
def niche_state():
    load = lambda k: json.loads(get_setting(k) or "null")
    props = load("niche_proposals")
    for p in props or []: p["adopted_as"] = niche_slug(p["name"]) if niche_slug(p["name"]) in sr.SERIES else None
    return {"youtube_data": google_yt.auth_mode(), "research": load("niche_research"), "proposals": props}

@app.post("/niche/research")
def niche_research(req: NicheResearch):
    if not google_yt.available(): raise HTTPException(400, "Set GOOGLE_APPLICATION_CREDENTIALS or YOUTUBE_API_KEY in .env")
    try: res = niche.research(req.keywords, req.region.upper(), req.days)
    except Exception as e: raise HTTPException(500, f"Research failed: {str(e)[:400]}")
    set_setting("niche_research", json.dumps(res)); set_setting("niche_proposals", "null")
    return res

@app.post("/niche/propose")
def niche_propose(req: NicheProfile):
    res = json.loads(get_setting("niche_research") or "null")
    if not res: raise HTTPException(409, "Run YouTube research first")
    try: props = niche.propose(res, req.profile, req.n)
    except Exception as e: raise HTTPException(500, f"Proposal failed: {str(e)[:400]}")
    set_setting("niche_proposals", json.dumps(props)); return props

@app.post("/niche/adopt")
def niche_adopt(req: NicheAdopt):
    props = json.loads(get_setting("niche_proposals") or "null") or []
    if req.index >= len(props): raise HTTPException(404, "No such proposal")
    p = props[req.index]
    slug = niche_slug(p["name"])
    spec = {"label": p["series_label"], "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False, "custom": True,
            "idea_brief": p["idea_brief"], "tags": p["tags"] + ["Shorts"] if "Shorts" not in p["tags"] else p["tags"],
            "brief": p["brief"] + (f" Stock footage should be: {p['stock_hint']}." if p.get("stock_hint") else "") +
                     " Each scene's stock_query must be a concrete, filmable thing. Only state well-established facts."}
    con = db(); con.execute("INSERT INTO custom_series(slug,spec_json) VALUES(?,?) ON CONFLICT(slug) DO UPDATE SET spec_json=excluded.spec_json", (slug, json.dumps(spec)))
    have = {r["topic"].lower() for r in con.execute("SELECT topic FROM ideas WHERE series=?", (slug,))}
    new = [t for t in p.get("first_topics", []) if t.lower() not in have]   # adopting twice must not duplicate ideas
    for t in new: con.execute("INSERT INTO ideas(series,topic,created_at) VALUES(?,?,?)", (slug, t, now()))
    con.commit(); con.close()
    sr.register(slug, spec)
    return {"series": slug, "label": spec["label"], "ideas_added": len(new)}

# ---- Instagram downloader (public posts/reels) via the yt-dlp CLI (brew install yt-dlp; keep it updated, Instagram changes often).
# Optional: INSTAGRAM_COOKIES_BROWSER=chrome|firefox|safari|edge uses that browser's Instagram login when a post needs it.
DOWNLOADS = CONTENT / "downloads"
IG_URL = re.compile(r"^https?://(?:www\.)?instagram\.com/(?:[\w.]+/)?(?:p|reels?|tv)/([\w-]{5,})", re.I)
DL_NAME = re.compile(r"^ig_[\w-]+\.(?:mp4|webm|mkv|mov|json)$")

class DownloadRequest(BaseModel):
    url: str = Field(min_length=10, max_length=500)

def _dl_meta(video: Path) -> dict:
    meta = video.with_suffix(".json")
    try: d = json.loads(meta.read_text())
    except Exception: d = {}
    st = video.stat()
    return {"file": video.name, "size_mb": round(st.st_size / 1048576, 1), "saved_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(), **d}

def _playable(video: Path) -> Path:
    """Instagram often serves VP9, which QuickTime/Safari can't play. Re-encode anything that isn't H.264 into an H.264 .mp4."""
    try:
        codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)], text=True).strip()
        if codec == "h264" and video.suffix == ".mp4": return video
        out = video.with_name(video.stem + ".h264.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-200:])
        final = video.with_suffix(".mp4"); video.unlink(); out.replace(final); return final
    except Exception as e:
        print(f"[download] kept original codec: {str(e)[:200]}"); return video

@app.post("/downloads/instagram")
def download_instagram(req: DownloadRequest):
    m = IG_URL.match(req.url.strip())
    if not m: raise HTTPException(422, "Paste a link to an Instagram post or reel, like https://www.instagram.com/reel/ABC123/")
    ytdlp = os.getenv("YTDLP_BIN") or shutil.which("yt-dlp") or ("/opt/homebrew/bin/yt-dlp" if Path("/opt/homebrew/bin/yt-dlp").exists() else None)
    if not ytdlp: raise HTTPException(500, "yt-dlp is not installed. Run: brew install yt-dlp")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    url = f"https://www.instagram.com/p/{m.group(1)}/"   # canonical form, drops tracking parameters
    cmd = [ytdlp, "--no-progress", "--no-warnings", "--no-simulate", "-j", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
           "-o", str(DOWNLOADS / "ig_%(id)s.%(ext)s"), "--ffmpeg-location", shutil.which("ffmpeg") or "ffmpeg"]
    if os.getenv("INSTAGRAM_COOKIES_BROWSER"): cmd += ["--cookies-from-browser", os.getenv("INSTAGRAM_COOKIES_BROWSER")]
    try: r = subprocess.run(cmd + ["--", url], capture_output=True, text=True, timeout=int(os.getenv("YTDLP_TIMEOUT", "180")))
    except subprocess.TimeoutExpired: raise HTTPException(504, "Instagram took too long to respond; try again")
    items = []
    for line in r.stdout.splitlines():
        try: d = json.loads(line)
        except ValueError: continue
        files = sorted(DOWNLOADS.glob(f"ig_{d.get('id')}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        video = next((p for p in files if p.suffix in (".mp4", ".webm", ".mkv", ".mov")), None)
        if not video: continue
        video = _playable(video)
        try: secs = d.get("duration") or round(probe_duration(video), 1)   # Instagram often omits it
        except Exception: secs = None
        meta = {"source_url": req.url.strip(), "post": url, "handle": d.get("channel") or d.get("uploader_id"), "uploader": d.get("uploader"),
                "title": (d.get("title") or "")[:120], "caption": (d.get("description") or "")[:500], "duration": secs, "width": d.get("width"), "height": d.get("height")}
        video.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))
        items.append(_dl_meta(video))
    if not items:
        err = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["no video found"]
        low = err[0].lower()
        if any(k in low for k in ("login", "private", "cookies", "rate", "not available", "401", "403")):
            hint = ("Instagram wants a login for this post (private account, age-restricted, or rate-limited). If it is public, set "
                    "INSTAGRAM_COOKIES_BROWSER=chrome in .env (log in to Instagram in that browser) and restart.")
        elif "no video" in low or "no formats" in low: hint = "That post has no video (it is a photo-only post)."
        else: hint = "Download failed. Try `brew upgrade yt-dlp` (Instagram changes often)."
        raise HTTPException(502, f"{hint} Details: {err[0][:300]}")
    return {"items": items}

@app.get("/downloads")
def list_downloads():
    if not DOWNLOADS.exists(): return []
    vids = [p for p in DOWNLOADS.iterdir() if p.suffix in (".mp4", ".webm", ".mkv", ".mov") and DL_NAME.match(p.name)]
    return [_dl_meta(p) for p in sorted(vids, key=lambda p: p.stat().st_mtime, reverse=True)]

@app.get("/downloads/file/{name}")
def download_file(name: str):
    p = DOWNLOADS / name
    if not DL_NAME.match(name) or not p.exists(): raise HTTPException(404, "File not found")
    return FileResponse(p, filename=name)

@app.delete("/downloads/{name}")
def delete_download(name: str):
    p = DOWNLOADS / name
    if not DL_NAME.match(name) or not p.exists(): raise HTTPException(404, "File not found")
    p.unlink(); p.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": name}

class PermissionRecord(BaseModel):
    how: str = Field(min_length=2, max_length=40)          # e.g. Instagram DM, comment, email
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    proof: str = Field(default="", max_length=500)          # link to the message/comment or a note on where the screenshot is

@app.post("/downloads/{name}/permission")
def record_permission(name: str, rec: PermissionRecord):
    p = DOWNLOADS / name
    if not DL_NAME.match(name) or not p.exists() or p.suffix == ".json": raise HTTPException(404, "File not found")
    meta_p = p.with_suffix(".json"); meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta["permission"] = {**rec.model_dump(), "recorded_at": now()}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False))
    return _dl_meta(p)

class ToReview(BaseModel):
    title: str = Field(min_length=3, max_length=100)
    description: str = Field(default="", max_length=3000)
    hashtags: list[str] = Field(default_factory=list, max_length=15)

@app.post("/downloads/{name}/to-review")
def download_to_review(name: str, req: ToReview):
    """Turn a downloaded reel whose creator gave permission into a job in Review (unchanged video, credit added when published)."""
    p = DOWNLOADS / name
    if not DL_NAME.match(name) or not p.exists() or p.suffix == ".json": raise HTTPException(404, "File not found")
    meta = _dl_meta(p); perm = meta.get("permission")
    if not perm: raise HTTPException(409, "Record the creator's permission first")
    job_id = uuid.uuid4().hex[:12]; video = CONTENT / f"{job_id}.mp4"
    shutil.copyfile(p, video)
    lic = {"handle": meta.get("handle"), "uploader": meta.get("uploader"), "post": meta.get("post"), "file": name, "permission": perm}
    tags = [re.sub(r"[^\w]", "", t) for t in req.hashtags if re.sub(r"[^\w]", "", t)][:15]
    data = {"series": "licensed", "licensed": lic, "made_for_kids": False, "title": req.title}
    prov = {"licensed": lic, "visuals": {"sources": ["licensed"], "counts": {"licensed": 1}, "ai_requested": False}}
    con = db()
    con.execute("INSERT INTO jobs(id,created_at,updated_at,niche,status,series,topic,source,title,description,hashtags,script_json,video_path,provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, now(), now(), "licensed", "pending_review", "licensed", req.title, "licensed", req.title, req.description, json.dumps(tags), json.dumps(data, ensure_ascii=False), str(video), json.dumps(prov, ensure_ascii=False)))
    con.commit(); con.close()
    meta_p = p.with_suffix(".json"); m = json.loads(meta_p.read_text()); m.setdefault("jobs", []).append(job_id); meta_p.write_text(json.dumps(m, ensure_ascii=False))
    return {"id": job_id, "status": "pending_review"}

class DayPlan(BaseModel):
    slots: list[str] = Field(min_length=1, max_length=6)   # ISO times, in order

@app.post("/plan/day")
def plan_day(plan: DayPlan):
    """Schedule one day: 2 licensed re-posts and 1 own video (oldest approved first), in the order licensed, own, licensed."""
    con = db(); rows = con.execute("SELECT id,source FROM jobs WHERE status='approved' AND youtube_id IS NULL ORDER BY created_at").fetchall(); con.close()
    lic = [r["id"] for r in rows if r["source"] == "licensed"]; own = [r["id"] for r in rows if r["source"] != "licensed"]
    want = ["licensed", "own", "licensed", "own", "licensed", "own"][:len(plan.slots)]
    picks, results = [], []
    for kind in want: picks.append((lic if kind == "licensed" else own).pop(0) if (lic if kind == "licensed" else own) else None)
    for slot, (kind, jid) in zip(plan.slots, zip(want, picks)):
        if not jid: results.append({"slot": slot, "kind": kind, "error": f"no approved {kind} video waiting"}); continue
        try: results.append({"slot": slot, "kind": kind, **publish_youtube(jid, PublishOptions(publish_at=slot))})
        except HTTPException as e: results.append({"slot": slot, "kind": kind, "id": jid, "error": e.detail})
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse((Path(__file__).resolve().parent / "dashboard.html").read_text(encoding="utf-8"))
