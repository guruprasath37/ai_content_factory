"""Daily trending topics: YouTube's most-popular chart for a region, summarised into categories and keywords.
Costs ~2 API quota units per refresh (10,000/day free)."""
from __future__ import annotations
import re
from collections import Counter
from datetime import datetime, timezone

from . import google_yt as yt

_STOP = set("the a an of in on at to and or for from by is are was be with this that it its you your i we my our how what why when who "
            "official video videos new full part vs ft feat shorts short trailer live hd 4k episode ep song lyrics watch best top most".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9']{2,}", text.lower()) if w not in _STOP]


def fetch(region: str = "IN", limit: int = 50) -> dict:
    region = region.upper()
    cats = {c["id"]: c["snippet"]["title"] for c in yt.get("videoCategories", part="snippet", regionCode=region).get("items", [])}
    items = yt.get("videos", part="snippet,statistics", chart="mostPopular", regionCode=region, maxResults=min(limit, 50)).get("items", [])
    videos, cat_views, words = [], Counter(), Counter()
    cat_n: Counter = Counter()
    for it in items:
        sn, st = it["snippet"], it.get("statistics", {})
        cat = cats.get(sn.get("categoryId"), "Other"); views = int(st.get("viewCount", 0))
        cat_n[cat] += 1; cat_views[cat] += views
        for w in set(_words(sn["title"]) + [t.lower() for t in sn.get("tags", [])[:8] if " " not in t and t.lower() not in _STOP]): words[w] += 1
        videos.append({"id": it["id"], "title": sn["title"], "channel": sn["channelTitle"], "views": views, "category": cat,
                       "published": sn.get("publishedAt"), "url": f"https://youtube.com/watch?v={it['id']}"})
    return {"fetched_at": datetime.now(timezone.utc).isoformat(), "region": region, "videos": videos,
            "categories": [{"name": c, "count": n, "views": cat_views[c]} for c, n in cat_n.most_common()],
            "keywords": [{"word": w, "count": n} for w, n in words.most_common(40) if n >= 2][:14]}
