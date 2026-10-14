"""Daily ideas: for each content category, look at what is getting views on YouTube Shorts this week, then have the LLM write
ready-to-render scene scripts (visual prompts for AI video), Instagram search hashtags and a highlight label (movie / song / which fact).
Costs 100 YouTube quota units per category per refresh (10,000/day free)."""
from __future__ import annotations
import statistics
from datetime import datetime, timedelta, timezone
from typing import Callable

from . import google_yt as yt
from . import llm
from . import series as sr

ORIGINAL = ("All visuals must be ORIGINAL AI-video shots: no real people's likenesses, no named copyrighted characters, no logos, no on-screen text, "
            "and never quote lyrics or film dialogue. ")

CATEGORIES: list[dict] = [
    {"id": "satisfying", "label": "Satisfying", "queries": ["oddly satisfying", "satisfying transformation"],
     "about": "oddly satisfying visuals: materials cut, morphing, flowing, perfect fits, loops", "highlight": "the satisfying type, e.g. 'Kinetic sand cutting' or 'Liquid metal morph'",
     "rules": "Visuals are rich, concrete shots (materials, finish, colour, motion, camera). Narration is sparse and teasing."},
    {"id": "history", "label": "History facts", "queries": ["history facts", "ancient history shorts"],
     "about": "surprising, well-documented history facts", "highlight": "the fact in a few words, e.g. 'Fact: Cleopatra lived closer to the Moon landing than to the pyramids'",
     "rules": "Only well-established facts, no made-up numbers or dates. Visuals are atmospheric reconstructions (landscapes, objects, architecture), not real people."},
    {"id": "did_you_know", "label": "Did you know", "queries": ["did you know facts", "amazing facts"],
     "about": "surprising true facts from science, nature, the body and everyday life", "highlight": "the fact in a few words",
     "rules": "Only facts you are sure are true; if unsure pick another. Hook in scene 1."},
    {"id": "motivation", "label": "Motivation", "queries": ["motivation shorts", "discipline motivation"],
     "about": "short motivational messages about discipline, resilience, focus and growth", "highlight": "the core message in a few words, e.g. 'Discipline beats motivation'",
     "rules": "Original wording only (no quoting famous speeches). Visuals are cinematic nature/city/silhouette shots with no identifiable people."},
    {"id": "psychology", "label": "Psychology facts", "queries": ["psychology facts", "human behavior facts"],
     "about": "well-supported psychology findings and everyday mind effects", "highlight": "the effect or fact, e.g. 'Fact: the Zeigarnik effect'",
     "rules": "Only established findings; name the effect correctly; no medical advice. Visuals are symbolic and metaphorical."},
    {"id": "music", "label": "Music", "queries": ["trending music shorts", "new song shorts"],
     "about": "what is trending in music: songs, genres, moods and the story behind a sound", "highlight": "a SPECIFIC trending song or artist name (from the titles above when one appears); never a generic phrase",
     "rules": ORIGINAL + "Narration is original commentary (why a sound hits, mood, history of a genre). Visuals are abstract music-mood scenes (waveforms as light, instruments, crowds as silhouettes)."},
    {"id": "edits", "label": "Edits", "queries": ["edit shorts", "cinematic edit"],
     "about": "popular short-form edit styles and what makes them work (transitions, speed ramps, colour grades)", "highlight": "the edit style or trend, e.g. 'Speed-ramp city transition'",
     "rules": ORIGINAL + "Each idea showcases an edit style with original AI footage; narration explains the technique."},
    {"id": "movie_clips", "label": "Movie clips", "queries": ["movie scene shorts", "trending movie clip"],
     "about": "what is trending in film: scenes, trailers and moments people are talking about", "highlight": "a SPECIFIC movie title (take it from the trending titles above when one appears, otherwise a real film you are sure about); never a generic phrase",
     "rules": ORIGINAL + "Do NOT recreate or describe a copyrighted scene shot-for-shot. Narration is original commentary about the film (why the moment works, a production fact you are sure of). "
              "Visuals are original cinematic mood shots inspired by the genre, with no recognisable characters."},
]
CAT = {c["id"]: c for c in CATEGORIES}


def evidence(cat: dict, region: str, days: int = 7) -> list[dict]:
    """Top recent Shorts for the category's search terms (empty if YouTube access is missing or fails)."""
    if not yt.available(): return []
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids: list[str] = []
    for q in cat["queries"][:1]:   # one search per category keeps quota at 100 units
        d = yt.get("search", part="id", q=q + " #shorts", type="video", videoDuration="short", order="viewCount", publishedAfter=after, regionCode=region, maxResults=12)
        ids += [i["id"]["videoId"] for i in d.get("items", []) if i.get("id", {}).get("videoId")]
    out = []
    if ids:
        for it in yt.get("videos", part="snippet,statistics", id=",".join(ids[:50])).get("items", []):
            out.append({"title": it["snippet"]["title"], "channel": it["snippet"]["channelTitle"], "views": int(it.get("statistics", {}).get("viewCount", 0)),
                        "url": f"https://youtube.com/shorts/{it['id']}"})
    return sorted(out, key=lambda x: -x["views"])[:8]


def ideas(cat: dict, ev: list[dict], n: int = 2, duration: int = 30, avoid: list[str] = ()) -> list[dict]:
    budget = sr._budget("general", duration)
    lines = "\n".join(f"- {e['title'][:90]} ({e['views']:,} views)" for e in ev) or "(no live data available; use your own knowledge of what performs well)"
    system = ("You are a short-form video strategist and scriptwriter for vertical Shorts/Reels narrated by a text-to-speech voice. "
              "Narration is natural to say aloud: no emojis, no hashtags, no stage directions, no markdown. Language: English.")
    user = (f"Category: {cat['label']} - {cat['about']}\nWhat is getting views on YouTube Shorts this week in this category:\n{lines}\n\n"
            f"Write {n} DIFFERENT video ideas that ride what is working (take the angle or subject, never copy a title). Rules: {cat['rules']}\n"
            + (f"Already used recently, pick different subjects: {'; '.join(avoid[:20])}\n" if avoid else "") +
            f"Each idea: about {budget} words of narration in total across 4-6 scenes ({duration} seconds), one or two sentences per scene. "
            "Each scene's 'visual' is a detailed prompt for an AI video generator (subject, materials, colours, motion, camera, lighting; 30-60 words). "
            f"'highlight' = {cat['highlight']}. 'ig_search_tags' = 8-10 single-word or compact tags (no #) someone would search on Instagram to find this kind of content.\n"
            'Return JSON: {"ideas":[{"topic":"specific topic","highlight":"label","title":"max 70 chars","description":"1 sentence","hook":"first line",'
            '"hashtags":["4-8 words"],"ig_search_tags":["tag1","tag2"],"scenes":[{"narration":"...","visual":"...","stock_query":"2-4 word footage search"}]}]}')

    def check(d):
        items = d.get("ideas") if isinstance(d, dict) else None
        if not isinstance(items, list) or not items: raise ValueError("'ideas' must be a non-empty array")
        good, err = [], None
        for it in items:
            try:
                fit = sr._fit(it, "general", duration)
                fit["topic"] = " ".join(str(it.get("topic") or fit["title"]).split())[:120]
                fit["highlight"] = " ".join(str(it.get("highlight") or fit["topic"]).split())[:120]
                fit["ig_search_tags"] = list(dict.fromkeys("".join(c for c in str(t) if c.isalnum()).lower() for t in (it.get("ig_search_tags") or []) if str(t).strip()))[:10]
                if not fit["ig_search_tags"]: raise ValueError("'ig_search_tags' needs at least 3 tags")
                good.append(fit)
            except (ValueError, TypeError) as e: err = e
        if not good: raise ValueError(f"no valid idea: {err}")
        return good
    return llm.chat_json(system, user, validate=check, temperature=0.85)[:n]


def build(region: str, on_progress: Callable[[dict], None], previous: dict | None = None, n: int = 2, duration: int = 30) -> dict:
    """Generate today's ideas for every category. on_progress gets the partial result after each category so the UI fills in live."""
    used = [i["topic"] for c in (previous or {}).get("categories", []) for i in c.get("ideas", [])]
    res = {"date": datetime.now().astimezone().date().isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(), "region": region, "duration": duration, "categories": [], "errors": []}
    for cat in CATEGORIES:
        entry = {"id": cat["id"], "label": cat["label"], "evidence": [], "median_views": 0, "ideas": []}
        try: entry["evidence"] = evidence(cat, region)
        except Exception as e: res["errors"].append(f"{cat['label']} (YouTube): {str(e)[:120]}")
        if entry["evidence"]: entry["median_views"] = int(statistics.median(e["views"] for e in entry["evidence"]))
        try: entry["ideas"] = ideas(cat, entry["evidence"], n, duration, used)
        except Exception as e: res["errors"].append(f"{cat['label']}: {str(e)[:160]}")
        res["categories"].append(entry); on_progress(res)
    ranked = sorted((c for c in res["categories"] if c["median_views"]), key=lambda c: -c["median_views"])
    res["hot_niche"] = ranked[0]["id"] if ranked else None   # the category pulling the most views this week
    on_progress(res)
    return res
