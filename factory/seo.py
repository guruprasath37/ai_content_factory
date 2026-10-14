"""LLM-written title, caption and hashtags for a Short, tuned for how YouTube actually distributes Shorts.

What the prompt encodes (YouTube's own guidance, not tricks): Shorts reach people mainly through viewer response - how many swipe away,
how long they watch, replays, likes/comments/shares. Metadata helps YouTube understand the video and lets searchers find it, but it
cannot make a video spread, and misleading metadata (clickbait the video doesn't pay off, unrelated trending tags) hurts retention and
violates the spam policy. So: truthful hook titles with the main keyword first, a short keyword-rich description with a question that
invites comments, and 3-5 hashtags that are genuinely about the video (YouTube shows the first 3 above the title and ignores ALL
hashtags on a video that has more than 15).
"""
from __future__ import annotations
import re

from . import llm

BANNED = {"fyp", "foryou", "foryoupage", "viral", "trending", "trend", "explore", "follow", "followme", "like", "likeforlike", "subscribe", "sub4sub",
          "views", "boost", "xyz", "reels", "instagram", "tiktok", "love", "instagood", "video", "shortvideo", "shortsfeed", "ytshorts"}


# Broad, genuinely on-topic tags with high search volume per category/series. One of them is always used as the "niche" tag, so each video
# pairs specific tags (what it is) with a popular browse tag (where viewers look), which is how YouTube groups Shorts for discovery.
REACH = {
    "satisfying": ["satisfying", "oddlysatisfying", "asmr"], "history": ["history", "historyfacts", "ancienthistory"], "did_you_know": ["didyouknow", "facts", "amazingfacts"],
    "motivation": ["motivation", "mindset", "discipline"], "psychology": ["psychology", "psychologyfacts", "mindset"], "music": ["music", "newmusic", "musictrends"],
    "edits": ["videoediting", "edit", "cinematic"], "movie_clips": ["movies", "filmmaking", "cinema"],
    "tiny_tales": ["kidsstories", "storytime", "bedtimestories"], "color_lab": ["satisfying", "colors", "artforkids"], "mind_blown": ["didyouknow", "facts", "science"],
    "tech_tips": ["techtips", "technology", "lifehacks"], "daily_boost": ["motivation", "mindset", "positivity"], "transformations": ["satisfying", "oddlysatisfying", "transformation"],
    "miniatures": ["miniature", "miniatures", "tinyworld"], "room_makeover": ["roommakeover", "interiordesign", "homedecor"], "macro_asmr": ["asmr", "macro", "satisfying"],
    "impossible": ["physics", "science", "illusion"],
}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _related(tag: str, corpus_words: set[str], corpus_flat: str) -> bool:
    """A hashtag must share a real word (or stem) with the video's own text; stops the model adding unrelated popular tags."""
    t = _norm(tag)
    if len(t) < 3: return False
    if t in corpus_flat: return True
    return any(len(w) >= 4 and (w in t or t in w or (len(w) >= 5 and t[:5] == w[:5])) for w in corpus_words)


def optimize(ctx: dict, hot_words: list[str] | None = None) -> dict:
    """ctx: title, description, hashtags (existing), narration (full spoken text or caption), topic, highlight, series_label.
    Returns {title, description, hashtags (without #, 3-5, includes Shorts), keywords}."""
    corpus = " ".join(str(ctx.get(k) or "") for k in ("title", "description", "narration", "topic", "highlight", "series_label")) + " " + " ".join(ctx.get("hashtags") or [])
    words = {w for w in re.findall(r"[a-z0-9]+", corpus.lower()) if len(w) >= 4}
    flat = _norm(corpus)
    hot = [w for w in (hot_words or []) if w][:14]
    reach = [t for t in (ctx.get("reach_tags") or []) if t][:6]
    reach_n = {_norm(t) for t in reach}
    system = ("You are a YouTube Shorts growth editor. You write honest metadata that matches the video, because Shorts are distributed by viewer response "
              "(swipe-away rate, watch time, replays, likes, comments, shares) and misleading metadata lowers retention and breaks YouTube's spam policy. Language: English.")
    user = (f"Video summary\n- Format/series: {ctx.get('series_label') or 'general'}\n- Topic: {ctx.get('topic') or ctx.get('title')}\n- Highlight: {ctx.get('highlight') or '-'}\n"
            f"- Current title: {ctx.get('title')}\n- Spoken content: {str(ctx.get('narration') or ctx.get('description') or '')[:900]}\n"
            + (f"- Words that are popular on YouTube right now (use one ONLY if it truly describes this video): {', '.join(hot)}\n" if hot else "") +
            "\nWrite:\n"
            "1. title: max 60 characters, main keyword within the first 40, a curiosity hook the video truthfully pays off, no ALL CAPS words, no emojis, no hashtags.\n"
            "2. description: 2-3 short sentences. Sentence 1 states what the video shows using the main keyword naturally; then one line of context; the last line is a specific question that invites comments. No hashtags, no links, no 'subscribe/like' begging.\n"
            "3. hashtags: exactly 4 hashtags WITHOUT '#', each about this video's actual subject or format: ONE broad niche tag (high search volume), two specific topic tags, one format tag. "
            + (f"Broad niche tag: pick exactly one of these popular tags that fits: {', '.join(reach)}. " if reach else "") +
            "Never use generic spam tags (fyp, viral, trending, foryou, explore). Do not include 'Shorts' (it is added for you).\n"
            "4. keywords: 5 search phrases a viewer would type to find this video.\n"
            'Return JSON: {"title":"...","description":"...","hashtags":["..."],"keywords":["..."]}')

    def check(d):
        if not isinstance(d, dict): raise ValueError("top level must be an object")
        title = re.sub(r"\s+", " ", re.sub(r"#\S+", "", str(d.get("title") or ""))).strip()
        if not title: raise ValueError("'title' is required")
        title = (title[:70].rsplit(" ", 1)[0] if len(title) > 70 else title).strip(" -:,")
        title = " ".join(w.capitalize() if len(w) > 3 and w.isupper() else w for w in title.split())
        desc = re.sub(r"[ \t]+", " ", re.sub(r"#\S+|https?://\S+", "", str(d.get("description") or ""))).strip()
        if len(desc) < 25: raise ValueError("'description' must be 2-3 sentences ending with a question")
        tags, seen = [], set()
        for t in d.get("hashtags") or []:
            tag = re.sub(r"[^\w]", "", str(t))
            n = _norm(tag)
            if not tag or n in seen or n in BANNED or n == "shorts" or len(tag) > 30: continue
            if n not in reach_n and not _related(tag, words, flat): continue   # not about this video (category browse tags are on-topic by definition)
            seen.add(n); tags.append(tag)
        if len(tags) < 2: raise ValueError(f"fewer than 2 hashtags are about this video's own subject (kept: {tags}); use words that appear in the topic or content")
        kw = [re.sub(r"\s+", " ", str(k)).strip()[:60] for k in (d.get("keywords") or []) if str(k).strip()][:6]
        if reach and not any(_norm(t) in reach_n for t in tags): tags = [reach[0]] + [t for t in tags if _norm(t) != _norm(reach[0])]   # always carry one popular on-topic browse tag
        return {"title": title, "description": desc[:600], "hashtags": tags[:4] + ["Shorts"], "keywords": kw}
    return llm.chat_json(system, user, validate=check, temperature=0.6)
