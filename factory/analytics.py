"""Pipeline stats and per-video performance (YouTube public stats, Instagram insights)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

import requests

from . import google_yt


def _now(): return datetime.now(timezone.utc).isoformat()


def _upsert(con, job_id, platform, views, likes, comments):
    con.execute("INSERT INTO metrics(job_id,platform,fetched_at,views,likes,comments) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(job_id,platform) DO UPDATE SET fetched_at=excluded.fetched_at,views=excluded.views,likes=excluded.likes,comments=excluded.comments",
                (job_id, platform, _now(), views, likes, comments))


def refresh(con, youtube_api_key: str, ig_token: str, graph_version: str = "v23.0") -> dict[str, Any]:
    """Pull latest numbers for every published video. YouTube needs only an API key (public statistics)."""
    out: dict[str, Any] = {"youtube": 0, "instagram": 0, "errors": []}
    yt = con.execute("SELECT id, youtube_id FROM jobs WHERE youtube_id IS NOT NULL").fetchall()
    if yt and not google_yt.available():
        out["errors"].append("No YouTube credentials: set YOUTUBE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in .env to read YouTube stats.")
    elif yt:
        by_vid = {r["youtube_id"]: r["id"] for r in yt}
        ids = list(by_vid)
        for i in range(0, len(ids), 50):
            try:
                for item in google_yt.get("videos", part="statistics", id=",".join(ids[i:i + 50])).get("items", []):
                    st = item.get("statistics", {})
                    _upsert(con, by_vid[item["id"]], "youtube", int(st.get("viewCount", 0)), int(st.get("likeCount", 0)), int(st.get("commentCount", 0)))
                    out["youtube"] += 1
            except Exception as e:
                out["errors"].append(f"YouTube: {str(e)[:200]}")
    ig = con.execute("SELECT id, instagram_id FROM jobs WHERE instagram_id IS NOT NULL").fetchall()
    if ig and not ig_token:
        out["errors"].append("Set INSTAGRAM_ACCESS_TOKEN to read Instagram insights.")
    elif ig:
        for row in ig:
            try:
                r = requests.get(f"https://graph.facebook.com/{graph_version}/{row['instagram_id']}/insights", params={"metric": "views,likes,comments", "access_token": ig_token}, timeout=30)
                r.raise_for_status()
                vals = {m["name"]: (m.get("values") or [{}])[0].get("value", 0) for m in r.json().get("data", [])}
                _upsert(con, row["id"], "instagram", vals.get("views", 0), vals.get("likes", 0), vals.get("comments", 0))
                out["instagram"] += 1
            except Exception as e:
                out["errors"].append(f"Instagram {row['instagram_id']}: {str(e)[:200]}")
    con.commit()
    return out


def summary(con) -> dict[str, Any]:
    status = {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
    per_series = []
    for r in con.execute("""
        SELECT COALESCE(j.series,'tiny_tales') series, COUNT(*) jobs,
               SUM(j.status='published') published, SUM(j.status IN ('failed','render_failed','publish_failed')) failed,
               SUM(j.source='auto') auto,
               COALESCE(SUM(m.views),0) views, COALESCE(SUM(m.likes),0) likes, COALESCE(SUM(m.comments),0) comments
        FROM jobs j LEFT JOIN metrics m ON m.job_id=j.id GROUP BY series ORDER BY views DESC, jobs DESC"""):
        d = dict(r); d["avg_views"] = round(d["views"] / d["published"], 1) if d["published"] else 0; per_series.append(d)
    top = [dict(r) for r in con.execute("""
        SELECT j.id, j.title, COALESCE(j.series,'tiny_tales') series, m.platform, m.views, m.likes, m.comments, m.fetched_at
        FROM metrics m JOIN jobs j ON j.id=m.job_id ORDER BY m.views DESC LIMIT 10""")]
    daily = [dict(r) for r in con.execute("""
        SELECT substr(created_at,1,10) day, COUNT(*) made FROM jobs WHERE created_at >= date('now','-13 day') GROUP BY day ORDER BY day""")]
    totals = con.execute("SELECT COALESCE(SUM(views),0) v, COALESCE(SUM(likes),0) l, COALESCE(SUM(comments),0) c FROM metrics").fetchone()
    last = con.execute("SELECT MAX(fetched_at) t FROM metrics").fetchone()["t"]
    return {"status": status, "per_series": per_series, "top": top, "daily": daily, "totals": dict(totals), "metrics_updated": last}
