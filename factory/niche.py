"""Niche research: what is actually working on YouTube Shorts right now, and which niches to enter."""
from __future__ import annotations
import math, re, statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from . import google_yt as yt
from . import llm

# Broad starting points when the user gives no keywords. Each search costs 100 API quota units (10,000/day free).
DEFAULT_KEYWORDS = ["did you know facts", "ai tools", "life hacks", "psychology facts", "space facts", "history facts",
                    "money tips", "satisfying", "motivation", "science experiments", "health tips", "travel facts"]


def _clean(t) -> str:
    """Drop emojis/odd symbols (the model copies them from video titles) and tidy spacing."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-—–:,.'’&?!()/]", "", str(t))).strip()


def _secs(iso: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m: return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def research(keywords: list[str] | None, region: str = "IN", days: int = 30, per_query: int = 20) -> dict[str, Any]:
    keywords = [k.strip() for k in (keywords or DEFAULT_KEYWORDS) if k.strip()][:15]
    after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hits: dict[str, list[str]] = {}          # keyword -> video ids
    for k in keywords:
        d = yt.get("search", part="id", q=k + " #shorts", type="video", videoDuration="short", order="viewCount",
                   publishedAfter=after, regionCode=region, maxResults=per_query)
        hits[k] = [i["id"]["videoId"] for i in d.get("items", []) if i.get("id", {}).get("videoId")]
    ids = sorted({v for vs in hits.values() for v in vs})
    vids: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        for it in yt.get("videos", part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50])).get("items", []):
            vids[it["id"]] = it
    chans = sorted({v["snippet"]["channelId"] for v in vids.values()})
    subs: dict[str, int] = {}
    for i in range(0, len(chans), 50):
        for it in yt.get("channels", part="statistics", id=",".join(chans[i:i + 50])).get("items", []):
            st = it.get("statistics", {})
            subs[it["id"]] = 0 if st.get("hiddenSubscriberCount") else int(st.get("subscriberCount", 0))
    now = datetime.now(timezone.utc)
    rows = []
    for k, vs in hits.items():
        items = []
        for vid in vs:
            v = vids.get(vid)
            if not v or _secs(v["contentDetails"]["duration"]) > 90: continue
            views = int(v["statistics"].get("viewCount", 0))
            age = max(1.0, (now - datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00"))).total_seconds() / 86400)
            s = subs.get(v["snippet"]["channelId"], 0)
            items.append({"id": vid, "title": v["snippet"]["title"], "channel": v["snippet"]["channelTitle"], "views": views,
                          "views_per_day": round(views / age), "subs": s, "url": f"https://youtube.com/shorts/{vid}"})
        if not items: continue
        small_wins = [x for x in items if 0 < x["subs"] < 100_000 and x["views"] >= 50_000]
        rows.append({"keyword": k, "n": len(items), "median_views": int(statistics.median(x["views"] for x in items)),
                     "median_views_per_day": int(statistics.median(x["views_per_day"] for x in items)),
                     "small_channel_wins": len(small_wins), "small_win_share": round(len(small_wins) / len(items), 2),
                     "top": sorted(items, key=lambda x: -x["views"])[:5]})
    if rows:   # opportunity = demand (views) blended with proof that small channels can win
        lv = [math.log10(r["median_views"] + 1) for r in rows]; lo, hi = min(lv), max(lv)
        for r, l in zip(rows, lv): r["opportunity"] = round(100 * (0.5 * ((l - lo) / (hi - lo) if hi > lo else 0.5) + 0.5 * r["small_win_share"]))
    rows.sort(key=lambda r: -r.get("opportunity", 0))
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "region": region, "days": days, "keywords": rows, "quota_units_used": 100 * len(keywords) + 3}


def propose(res: dict, profile: str = "", n: int = 5) -> list[dict]:
    """Ask the local LLM to turn the research into concrete niches the factory can produce."""
    lines = []
    for r in res["keywords"][:12]:
        titles = "; ".join(t["title"][:60] for t in r["top"][:3])
        lines.append(f"- {r['keyword']}: median {r['median_views']:,} views, {r['small_channel_wins']}/{r['n']} videos from channels under 100k subs got 50k+ views, opportunity {r['opportunity']}. Top titles: {titles}")
    user = (f"Real YouTube Shorts data from the last {res['days']} days (region {res['region']}):\n" + "\n".join(lines) +
            f"\n\nAbout the creator / constraints: {profile or 'a faceless channel; videos are made automatically from stock footage with an AI voice-over; 15-60 seconds; no on-camera person; no copyrighted material.'}\n"
            f"No emojis anywhere in your answer. Propose {n} distinct niches to start, favouring high demand AND proof that small channels win, and that can be produced with stock footage + narration. "
            'Return JSON: {"niches":[{"name":"short name","why":"1-2 sentences quoting the actual numbers above (median views, small-channel wins)","series_label":"Name — tagline","brief":"instructions for the scriptwriter: structure, tone, rules (accuracy!)",'
            '"idea_brief":"what topics this channel covers","tags":["Tag1","Tag2","Tag3","Shorts"],"stock_hint":"what kind of footage fits","first_topics":["5 specific video topics"]}]}')

    def check(d):
        items = d.get("niches") if isinstance(d, dict) else None
        if not isinstance(items, list) or not items: raise ValueError("'niches' must be a non-empty array")
        out = []
        for it in items:
            if not isinstance(it, dict) or not all(it.get(k) for k in ("name", "series_label", "brief", "idea_brief")): continue
            for k in ("name", "series_label", "why", "brief", "idea_brief", "stock_hint"):
                if it.get(k): it[k] = _clean(it[k])
            it["tags"] = [re.sub(r"[^\w]", "", str(t)) for t in (it.get("tags") or [])][:6] or ["Shorts"]
            it["first_topics"] = [_clean(t)[:100] for t in (it.get("first_topics") or []) if _clean(t)][:8]
            out.append(it)
        if not out: raise ValueError("each niche needs name, series_label, brief and idea_brief")
        return out
    return llm.chat_json("You are a YouTube Shorts growth strategist who decides niches from data, not hunches.", user, validate=check, temperature=0.7)[:n]
