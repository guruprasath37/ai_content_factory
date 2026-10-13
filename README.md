# AI Content Factory

Automatically produces vertical short videos (YouTube Shorts / Reels) from an idea: an LLM writes the script,
a neural voice narrates it, and the visuals come from stock footage, optional local AI video, or a built-in
animation engine. Videos land in a review queue; **nothing is published without your approval.**

```
niche research ──► idea bank ──► script (LLM) ──► voice-over (edge-tts) ──► visuals ──► captions + ffmpeg ──► review ──► publish
 (YouTube data)    (autopilot)   Groq → Ollama     Microsoft Edge TTS       per scene:                         approve /    YouTube /
                                                                            AI clip → stock → drawn            reject       Instagram
```

## Features

- **Script writing** with a provider chain: Groq (free tier, `gpt-oss-120b` then `gpt-oss-20b`) → local Ollama → optional
  OpenAI-compatible API. If one fails (rate limit, outage), the next is used. Scripts are validated and the model is
  asked to repair bad output.
- **Ten built-in series**
  - *Narrated:* Tiny Tales (animated robot stories), Color Lab, Mind-Blown Facts, Tech Tips, Daily Boost.
  - *Visual-first* (a few punchy words per scene; each scene is a detailed shot prompt; AI video pre-ticked): Satisfying Transformations,
    Miniature Worlds, AI Room Makeovers, Macro ASMR, Impossible Physics.
  - Plus any **niche** you adopt from research.
- **AI video, your choice per video**: engine (Kling paid / LTX free-local) and source - **Script → video** (the scene text alone)
  or **Image → video** (animates a matching photo). LTX keeps the subject much better in image mode; abstract, material-based
  scenes work well in script mode.
- **Niche finder**: scans recent YouTube Shorts for the topics you give it, measures demand and whether small channels
  win there, and has the LLM propose niches grounded in those numbers.
- **Autopilot**: scheduled batches (idea bank → script → render), pauses when too many videos await review.
- **Visuals, best first, per scene**: [Kling](https://klingai.com) AI clip (paid, optional; **text-to-video** or image-to-video)
  → local [LTX-Video 2B](https://huggingface.co/Lightricks/LTX-Video) clip (free, optional) → [Pixabay](https://pixabay.com) stock footage
  → procedurally drawn animation. A failure in one scene never fails the video, and every video shows which engine made it.
- **Captions** burned in as short 1–4 word chunks (Poppins ExtraBold), timed to the narration.
- **Analytics**: pipeline stats and per-video views/likes (YouTube public stats, Instagram insights).
- **Review dashboard** (web UI) and a documented REST API (`/docs`).

## Requirements

- Python 3.10+ and **ffmpeg** (with libass for subtitles): `sudo apt install ffmpeg`; on macOS `brew install ffmpeg-full && brew link --force ffmpeg-full` (Homebrew's plain `ffmpeg` has no subtitles filter)
- At least one script writer: a free [Groq key](https://console.groq.com) **or** [Ollama](https://ollama.com) with a model
  (`ollama pull qwen2.5:7b`)
- Internet access (voice-over uses Microsoft Edge TTS)
- Optional: `brew install yt-dlp` for the **Download** tab (public Instagram posts and reels)
- Optional: a free [Pixabay API key](https://pixabay.com/api/docs/) for stock footage (without it, the drawn animation is used)
- Optional, for local AI video: an NVIDIA GPU with 8 GB+ VRAM, ~32 GB free disk, 16 GB+ RAM

## Quick start

```bash
git clone <your-repo-url> ai-content-factory && cd ai-content-factory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add GROQ_API_KEY (and PIXABAY_API_KEY)
./run.sh                    # open http://127.0.0.1:8000
```

Open the dashboard, choose a series on **Create**, and press *Generate & render*. The finished video appears under **Review**.

### Optional: Kling (paid) AI video

Add `KLING_ACCESS_KEY` and `KLING_SECRET_KEY` to `.env` (see `.env.example`), then on **Create** tick *AI video clips*, choose
the **Kling** engine and a mode: **Script → video** (default for Kling; the scene description alone generates a native 9:16 clip) or
**Image → video** (animates a matching Pixabay photo). Kling bills per clip, so two guards limit spend:
`KLING_MAX_CLIPS_PER_VIDEO` and `KLING_DAILY_CLIP_LIMIT`; scenes beyond them use LTX or stock footage. Check how many units a
clip uses in your Kling console. The integration was tested against a mock of the API, not a live account.

### Optional: local AI video clips

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126     # match your CUDA version
pip install -r requirements-aivideo.txt
python scripts/download_ltx_model.py                                      # ~27 GB -> models/LTX-Video
```

Then tick **AI video clips** on the Create tab (or in Autopilot). On an RTX A1000 (8 GB) one ~2 s clip takes about a minute,
so a 4–6 scene video needs roughly 5–7 minutes of GPU time. Each scene's clip is animated from a Pixabay photo that matches the
scene. The prompts are encoded on the CPU and the local LLM is unloaded first so the 2B model fits in 8 GB.

## Using it

| Tab | What it does |
|---|---|
| **Create** | Pick a series, length, language and (optionally) a topic; generates the script and renders the video. |
| **Autopilot** | Schedule batches, choose which series to rotate, pause when the review queue is full, watch the live log. |
| **Review** | Watch videos, approve / reject, publish to YouTube or Instagram, retry failed renders. |
| **Analytics** | Pipeline counts, results per series, top videos; refresh platform stats. |
| **Niche** | Research what works on YouTube Shorts, get AI-recommended niches, adopt one as a new series. |

The idea bank that feeds Autopilot is managed through the API (`/ideas`, see `/docs`); Autopilot refills it automatically
and adopting a niche seeds it with starter topics.

## Configuration

All settings are environment variables, usually in `.env` — see [.env.example](.env.example) for the full list.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDERS` | Order to try script writers (`groq,ollama` default; `openai` also supported). |
| `GROQ_API_KEY`, `GROQ_MODELS` | Groq key and the models to try, in order. |
| `OLLAMA_URL`, `OLLAMA_MODEL` | Local LLM (default: best installed model). |
| `PIXABAY_API_KEY` | Stock footage and start photos. Results are cached 24 h and downloaded, never hotlinked (Pixabay terms). |
| `GOOGLE_APPLICATION_CREDENTIALS` / `YOUTUBE_API_KEY` | Read-only YouTube access for niche research and stats. |
| `YOUTUBE_CLIENT_SECRETS` | OAuth client for **uploading** to YouTube (a service account cannot upload). |
| `INSTAGRAM_*`, `MEDIA_PUBLIC_BASE_URL` | Instagram Reels publishing (needs a public HTTPS URL for the MP4). |
| `KLING_*` | Kling API keys, model, quality mode, and the spend guards. |
| `LTX_*` | Local AI video resolution, length, steps, model path. |

## Project layout

```
factory/
  app.py          FastAPI app: pipeline (script → voice → visuals → ffmpeg), jobs, autopilot worker, REST API
  series.py       The series definitions, script generation + validation, idea brainstorming
  llm.py          Provider chain (Groq → Ollama → OpenAI-compatible), JSON repair/retry
  niche.py        YouTube research and LLM niche proposals
  google_yt.py    Read-only YouTube Data API access (API key or service account)
  analytics.py    Pipeline stats and platform metrics
  aivideo.py      Orchestrates local AI clips (start photo → LTX → per-scene fallback)
  kling.py        Kling API client (JWT auth, text/image-to-video, polling, clip caps)
  ltx_worker.py   LTX-Video worker process (own process so GPU/RAM are freed afterwards)
  dashboard.html  Single-page web UI
scripts/          download_ltx_model.py
assets/fonts/     Poppins ExtraBold (SIL Open Font License, see OFL.txt)
credentials/      Your key files go here (git-ignored)
```

Runtime data (`content/`, `assets/pixabay/`, `assets/ai_clips/`, `content_factory.sqlite3`, `models/`) is git-ignored.

## API

Interactive docs at `http://127.0.0.1:8000/docs`. Main routes: `POST /generate`, `POST /render/{id}`, `GET /jobs`,
`POST /approve/{id}`, `POST /reject/{id}`, `POST /publish/{id}/youtube|instagram`, `GET|POST /autopilot`,
`POST /autopilot/run`, `GET|POST /ideas`, `GET /analytics`, `POST /niche/research|propose|adopt`, `GET /health`.

## Notes and limits

- **Disclose AI content.** YouTube and Instagram require labelling realistic synthetic media; set it when publishing.
  The "made for kids" flag is set per series and sent to YouTube — review it for your content.
- **Fact accuracy is on you.** The Mind-Blown / Tech Tips prompts ask for well-established facts only, but LLMs make mistakes.
  Review every video before approving.
- **Stock footage relevance is search-based**, so a clip occasionally matches loosely (the visual-first series are meant for AI video, where stock is only a fallback). Credit Pixabay where you publish.
- **No sound effects yet.** Macro ASMR is narrated by the voice-over; synchronized SFX would need an audio-generation step.
- **Free tiers have limits.** Groq limits tokens per day per model; YouTube Data API has a daily quota (a niche scan of
  6 topics uses ~600 of 10,000 units). Check each provider's current limits.
- **Licences of third-party pieces**: LTX-Video weights use the Lightricks open-weights licence; Poppins is OFL; Pixabay content
  follows the Pixabay Content License. Check them before commercial use. Add your own `LICENSE` file for this code.

## Troubleshooting

- `Render failed: [Errno 2] ... 'ffprobe'` → install ffmpeg (`sudo apt install ffmpeg`, or `brew install ffmpeg-full && brew link --force ffmpeg-full` on macOS).
- Script generation fails → check `/health` (the `llm` block lists the chain); add a Groq key or start Ollama.
- AI video says not ready → `/health` shows `ai_video`; it needs the downloaded model and a CUDA GPU.
- Hindi/Tamil captions: Poppins has no Indic glyphs, so the renderer relies on a system fallback font; install one (e.g. Noto Sans Devanagari / Tamil) if captions look wrong.
# ai_content_factory
