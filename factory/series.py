"""Content formats ("series"), LLM script generation and idea brainstorming."""
from __future__ import annotations
import difflib, json, random, re
from typing import Any

from . import llm

WORDS_PER_SECOND = 2.4   # conversational TTS pace

# visual: procedural_tales = robot story animation, stock = Pixabay footage (vivid = prefer bright, colourful clips)
SERIES: dict[str, dict[str, Any]] = {
    "tiny_tales": {
        "label": "Tiny Tales — robot mini-stories", "visual": "procedural_tales", "kids": True, "stock_prefix": "", "vivid": False,
        "idea_brief": "gentle, original children's mini-story premises starring a little robot (kindness, curiosity, sharing, courage)",
        "brief": ("An original, gentle, age-appropriate mini-story with a recurring little robot character: clear beginning, small problem, kind resolution, "
                  "no peril or frightening content, and a simple emotional takeaway. Avoid copyrighted characters and franchises."),
        "tags": ["TinyTales", "KidsStories", "LittleRobot", "Shorts"],
    },
    "color_lab": {
        "label": "Color Lab — colourful abstract satisfying", "visual": "stock", "kids": True, "stock_prefix": "colorful vibrant ", "vivid": True,
        "idea_brief": "playful colour experiments and colour-mixing ideas (what happens when colours blend, rainbows, gradients, paint, light)",
        "brief": ("A playful, satisfying colour experiment told with short simple narration. Every scene's stock_query must describe real, colourful, "
                  "bright footage (ink in water, paint swirls, neon light, rainbow gradients, kaleidoscopes, liquid art)."),
        "tags": ["ColorLab", "Satisfying", "Colors", "Shorts"],
    },
    "mind_blown": {
        "label": "Mind-Blown Facts — surprising true facts", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "idea_brief": "surprising but well-established facts about space, the human body, animals, oceans, history, physics and everyday science",
        "brief": ("A short 'did you know' video: a punchy hook in scene 1, then 3-5 surprising facts that are well established and true (no myths, no made-up numbers; "
                  "if you are not sure a fact is true, pick a different one), then a one-line closing thought. Each scene's stock_query must be a concrete, "
                  "filmable thing (e.g. 'octopus underwater', 'milky way timelapse')."),
        "tags": ["DidYouKnow", "Facts", "Science", "Shorts"],
    },
    "tech_tips": {
        "label": "Tech Tips — AI tools & productivity", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "idea_brief": "practical tips for using AI tools, free software, keyboard shortcuts, phone/computer tricks and productivity habits",
        "brief": ("A practical how-to: hook with the problem in scene 1, then 3-4 concrete steps or tips someone can try today, then a one-line recap. "
                  "Only recommend things that really exist; do not invent product features. Each scene's stock_query must be a concrete filmable thing "
                  "(e.g. 'typing on laptop', 'smartphone screen', 'person working desk')."),
        "tags": ["TechTips", "AITools", "Productivity", "Shorts"],
    },
    "daily_boost": {
        "label": "Daily Boost — motivation & mindset", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "idea_brief": "short motivational messages about habits, focus, resilience, learning and starting before you feel ready",
        "brief": ("A calm, original motivational message (no quotes from real people, no medical or financial claims): hook, 2-4 sentences of grounded encouragement, "
                  "one small action the viewer can take today. Each scene's stock_query must be a concrete filmable thing (e.g. 'sunrise over mountains', 'runner at dawn')."),
        "tags": ["Motivation", "Mindset", "DailyBoost", "Shorts"],
    },
}

# Specific, proven sub-niches per visual-first series; a random few are offered per idea so they stay varied and niche.
ANGLES = {
 "transformations": [
  "a rusty, broken padlock restored stage by stage into polished, engraved gold",
  "a cracked ceramic bowl rebuilt with flowing liquid-gold kintsugi seams",
  "molten glass blown and pulled into a glowing jellyfish with trailing tentacles",
  "a flat sheet of paper folding itself into a crane, then unfolding into a dragon",
  "a rough log spun on a lathe into a glossy spiral staircase sculpture",
  "a hot caramel pour melting open a chocolate sphere to reveal a dessert inside",
  "a heap of old coins melting into a molten pool that sets into an embossed medallion",
  "an ice cube melting UPWARD into a delicate crystal flower",
  "a dull grey rock tumbled and polished until it reveals a vivid opal interior",
  "melted crayon wax poured into a mould and released as a translucent rose",
  "a pile of sand swirling and fusing into a clear glass hourglass",
  "a dented old car part sanded, primed and painted into mirror-chrome candy red"
 ],
 "macro_asmr": [
  "a hot wire slicing a layered rainbow kinetic-sand tower to reveal hidden colour bands",
  "crushing a block of dyed gym chalk that bursts into pastel powder",
  "a knife cutting a geode made of resin and crushed opal, revealing a glowing galaxy core",
  "shaving paper-thin curls from a pastel starch-coated soap cube",
  "a heated knife gliding through frozen honeycomb so honey slowly bleeds out",
  "a hydraulic press flattening a bundle of coloured wax crayons into a marbled disc",
  "slicing a crystal-clear fruit made of glass-like jelly so bright juice spills out",
  "ferrofluid rising into a spiky crown on a magnet, then collapsing into a mirror pool",
  "peeling a thick glossy film off a sculpted iridescent surface in one continuous strip",
  "carving a perfectly clear ice sphere from a block with a chisel",
  "squeezing a jelly-bead stress ball until it bursts into hundreds of tiny glossy orbs",
  "cutting an onion made of stacked translucent coloured gelatin layers"
 ],
 "miniatures": [
  "a fingertip-sized ramen shop where tiny noodles steam on a working stove",
  "an entire glowing forest with a winding river inside a light bulb",
  "a tiny blacksmith forge with a glowing anvil and flying sparks",
  "a rainy neon Tokyo alley built inside a matchbox",
  "a miniature bakery where tiny croissants rise and brown in a working oven",
  "a lit treehouse village built into a bonsai tree at dusk",
  "a miniature steam train crossing a bridge between bookshelf books",
  "a castle of sugar cubes with a working drawbridge",
  "a tiny underwater world in a fishbowl with a working submarine",
  "a city at night inside a teacup, windows lighting up one by one"
 ],
 "room_makeover": [
  "an abandoned dusty garage becoming a warm Japandi studio",
  "a dingy concrete basement becoming a hidden speakeasy bar",
  "a bare tiny balcony becoming a lush jungle reading nook",
  "an empty attic becoming a stargazing bedroom under a huge skylight",
  "a rusty shipping container becoming a luxury tiny home",
  "a tired 90s kitchen becoming a deep-green and brass chef's kitchen",
  "an empty walk-in closet becoming a secret library with a sliding ladder",
  "a cluttered spare room becoming a neon-lit gaming cave",
  "a cracked concrete bathroom becoming a stone-and-wood spa with a rain shower"
 ],
 "impossible": [
  "water pouring UPWARD out of a glass into a small rain cloud",
  "a marble rolling forever around an impossible Penrose staircase",
  "a shadow that slowly detaches and walks away from its object",
  "a book whose pages pour out as a waterfall",
  "a mirror reflection that moves a moment late, then on its own",
  "sand in an hourglass falling upward while time rewinds",
  "a cube that folds through itself into a rotating tesseract",
  "a balloon that holds a tiny stormy ocean with waves",
  "a door that opens onto the same room seen from above",
  "a marble run that loops back into itself with no start or end"
 ]
}

# ---- Visual-first formats: the picture is the product. Narration is a few punchy words per scene, and each scene's
# "visual" is written as a detailed shot description that doubles as the AI-video prompt. Best with AI video ON.
VISUAL_BRIEF = ("This is a VISUAL-FIRST short: the narration is only a few punchy words per scene (max ~8 words), never a lecture. "
                "Scene 1 is the hook and shows the payoff or the strangest moment immediately - no introduction. "
                "For every scene, 'visual' is a detailed single-shot description for an AI video generator (subject, material, colour, "
                "motion, camera, lighting; 25-45 words, no text on screen, no people's faces) and 'stock_query' is a plain 2-4 word fallback. ")

SERIES.update({
    "transformations": {
        "label": "Satisfying Transformations — shapes, cuts & liquid metal", "visual": "stock", "kids": False, "stock_prefix": "colorful ", "vivid": True,
        "ai_default": True, "wps": 1.5,
        "idea_brief": "satisfying object transformations: shapes morphing, perfect cuts, liquid-metal changes, material swaps and seamless loops",
        "brief": VISUAL_BRIEF + ("Format: one object or material transforms into another in a smooth, oddly satisfying way (e.g. a cube of liquid metal "
                                 "melting into a flower, a glass sphere sliced into perfect petals). Each scene is one stage of the transformation; "
                                 "the last scene should loop back toward the first."),
        "tags": ["Satisfying", "Transformation", "OddlySatisfying", "Shorts"],
    },
    "miniatures": {
        "label": "Miniature Worlds — tiny builds & little machines", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "ai_default": True, "wps": 1.5,
        "idea_brief": "miniature worlds: tiny houses, miniature kitchens and cities, impossible little machines being assembled",
        "brief": VISUAL_BRIEF + ("Format: a tiny world built in miniature. Scene 1 shows the FINISHED miniature as the hook, then the following scenes reveal "
                                 "how it was built, step by step, ending on a loving close-up of the finished piece. Describe the same miniature consistently "
                                 "in every scene (same colours, materials, scale)."),
        "tags": ["Miniature", "Tiny", "Satisfying", "Shorts"],
    },
    "room_makeover": {
        "label": "AI Room Makeovers — empty to luxurious", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "ai_default": True, "wps": 1.5,
        "idea_brief": "dramatic room makeovers: empty or cluttered rooms transforming into luxurious, organised interiors (before-and-after payoffs)",
        "brief": VISUAL_BRIEF + ("Format: a before-and-after room makeover. Scene 1 is the obvious payoff (the stunning finished room), the next scenes show the "
                                 "'before' and the transformation stages, and the last scene returns to the finished room. Describe the SAME room "
                                 "(same layout, window position, camera angle) in every scene so the clips feel continuous."),
        "tags": ["RoomMakeover", "InteriorDesign", "BeforeAndAfter", "Shorts"],
    },
    "macro_asmr": {
        "label": "Macro ASMR — glossy textures & impossible materials", "visual": "stock", "kids": False, "stock_prefix": "colorful ", "vivid": True,
        "ai_default": True, "wps": 1.4,
        "idea_brief": "macro close-ups of glossy textures, geometric cuts, colourful clay, smooth surfaces and fictional materials",
        "brief": VISUAL_BRIEF + ("Format: extreme close-up (macro) shots of a tactile material being cut, pressed, poured or smoothed, with crisp, satisfying motion. "
                                 "Name the sound the viewer would 'hear' in the narration (e.g. 'Slow, glossy slice.') because the picture carries the video. "
                                 "Use vivid colours and clean studio lighting; never show faces or hands in detail."),
        "tags": ["ASMR", "Macro", "Satisfying", "Shorts"],
    },
    "impossible": {
        "label": "Impossible Physics — visual puzzles & loops", "visual": "stock", "kids": False, "stock_prefix": "", "vivid": False,
        "ai_default": True, "wps": 1.5,
        "idea_brief": "impossible physics and visual puzzles: endless staircases, objects folding into themselves, impossible structures, riddles you replay to understand",
        "brief": VISUAL_BRIEF + ("Format: something that cannot happen in real life happens calmly and cleanly (an endless staircase, a cube folding inside out, "
                                 "water flowing upward). Scene 1 shows the impossible moment; the narration poses a tiny riddle ('Where does it end?'); "
                                 "the final scene should make the viewer want to replay the clip."),
        "tags": ["Illusion", "ImpossibleThings", "MindBending", "Shorts"],
    },
})


EXAMPLE = ('{"title":"Why Octopuses Have Three Hearts","description":"Two short sentences about the video.","hashtags":["Octopus","Ocean","Facts","Shorts"],'
           '"hook":"One sentence hook.","scenes":[{"narration":"One natural spoken sentence.","visual":"short visual cue","stock_query":"octopus underwater"}]}')


def register(slug: str, spec: dict) -> None:
    """Add (or replace) a series at runtime - used for niches adopted from research."""
    SERIES[slug] = spec


def label_choices() -> list[dict]:
    return [{"id": k, "label": v["label"], "kids": v["kids"], "visual": v["visual"], "ai_default": bool(v.get("ai_default"))} for k, v in SERIES.items()]


def _wps(series: str) -> float:
    return SERIES[series].get("wps", WORDS_PER_SECOND)


def is_visual(series: str) -> bool:
    """Visual-first formats: sparse narration over long AI shots; the video is padded with quiet to the target length."""
    return "wps" in SERIES.get(series, {})


def scene_range(duration: int, series: str = "", single: bool = False) -> tuple[int, int]:
    """Spoken formats: about one scene per 10 s (4-7 for a 30 s short), so longer videos get more scenes instead of longer
    sentences. Visual-first formats: a single continuous shot (one AI clip, played forward then backward to fill the video)."""
    if single or is_visual(series): return 1, 1
    lo = max(4, round(duration / 10))
    return lo, lo + 3


def _budget(series: str, duration: int, single: bool = False) -> int:
    """Target narration words. A single shot keeps narration sparse (the video is padded with quiet), whatever the series."""
    return int(duration * (min(_wps(series), 1.4) if single else _wps(series)))


def _shape(series: str, lo: int, hi: int) -> str:
    """The scene-count instruction for the prompt."""
    if lo == hi == 1:
        return ("Exactly 1 scene: the whole video is ONE continuous shot, so ignore any scene-by-scene structure above and put the "
                "full arc inside that single shot's motion. Its narration is 1-2 short punchy sentences that tease the payoff without describing every step. "
                "Its 'visual' is a rich prompt for an AI video generator describing that one 8-10 second shot from start to finish "
                "(80-130 words, no text on screen), written as three timed beats: "
                "0-2s HOOK - an unusual, specific object already mid-action so the first frame stops the scroll; "
                "2-7s ESCALATION - the action builds with detailed texture, physics and colour changes; "
                "7-10s PAYOFF - a surprising twist or reveal (hidden inside, unexpected material, perfect final form) that makes people replay. "
                "Be concrete and premium: exact materials and finishes, contrasting colours, scale, camera move, lighting. "
                "Never generic or simple (not just 'a liquid pours' or 'a cube is cut') - it must look like a top-performing video in its niche.")
    return f"{lo}-{hi} scenes, one sentence of narration each."


def _fit(data: Any, series: str, duration: int, single: bool = False) -> dict:
    """Validate/repair an LLM script. Raises ValueError (so llm.chat_json can ask the model to fix it)."""
    if not isinstance(data, dict): raise ValueError("top level must be a JSON object")
    scenes = data.get("scenes")
    lo, hi = scene_range(duration, series, single)
    if not isinstance(scenes, list) or len(scenes) < lo: raise ValueError(f"'scenes' must be an array of at least {lo} object(s)")
    clean = []
    for sc in scenes[:hi + 3]:
        if isinstance(sc, str): sc = {"narration": sc}
        if not isinstance(sc, dict): continue
        narr = re.sub(r"\s+", " ", str(sc.get("narration") or sc.get("text") or "")).strip()
        if len(narr.split()) < 3: continue
        visual = str(sc.get("visual") or sc.get("stock_query") or " ".join(narr.split()[:5])).strip()
        clean.append({"narration": narr, "visual": visual[:1200], "stock_query": str(sc.get("stock_query") or visual).strip()[:100]})
    if len(clean) < lo: raise ValueError("each scene needs a 'narration' sentence of at least 3 words")
    if (single or is_visual(series)) and len(clean) > hi:
        raise ValueError(f"too many scenes ({len(clean)}); this format is exactly {hi} continuous shot(s) - merge them into {hi}")
    budget = _budget(series, duration, single)
    while len(clean) > lo and sum(len(s["narration"].split()) for s in clean) > budget * 1.15: clean.pop()   # too long for the target duration
    words = sum(len(x["narration"].split()) for x in clean)
    if lo == hi == 1 and words > budget * 1.2:   # a single shot is about the picture and its sound; keep the voice sparse
        raise ValueError(f"narration is too long ({words} words); a single-shot video needs at most {int(budget * 1.2)} words - 1-2 short punchy sentences")
    if words < budget * 0.55: raise ValueError(f"narration is too short ({words} words); write about {budget} words across {lo}-{hi} scene(s) so it fills {duration} seconds")
    title = re.sub(r"\s+", " ", str(data.get("title") or "")).strip()[:70]
    if not title: raise ValueError("'title' is required (max 70 characters)")
    tags = [re.sub(r"[^\w]", "", str(t)) for t in (data.get("hashtags") or []) if str(t).strip()]
    tags = [t for t in tags if t][:8]
    for t in SERIES[series]["tags"]:
        if t.lower() not in [x.lower() for x in tags] and len(tags) < 8: tags.append(t)
    desc = str(data.get("description") or "").strip() or f"{title}. A short video."
    return {"title": title, "description": desc[:300], "hashtags": tags, "hook": str(data.get("hook") or clean[0]["narration"]), "scenes": clean, "series": series}


def generate_script(series: str, topic: str | None, language: str, duration: int, niche: str = "", single: bool = False, avoid=()) -> dict:
    s = SERIES[series]
    lo, hi = scene_range(duration, series, single)
    pick = ("choose one specific, high-demand topic yourself - the kind of idea in this niche that gets the most views and replays "
            "on Shorts right now - and make it visually striking" if single else "choose a fresh, specific topic yourself")
    if not topic and (single or is_visual(series)) and ANGLES.get(series):
        offer = random.sample(ANGLES[series], min(4, len(ANGLES[series])))
        pick = ("pick ONE of these proven sub-niches and push it further with your own specific twist (or an equally niche idea of the "
                "same calibre): " + "; ".join(offer))
    system = ("You write scripts for vertical short videos (YouTube Shorts / Reels), narrated by a text-to-speech voice. "
              f"Language: {language}. Keep narration natural to say aloud: no emojis, no hashtags, no stage directions, no markdown.")
    user = (f"Format: {s['label']}\n{s['brief']}\n"
            f"Topic: {topic or pick}\n"
            + (f"Already shown and rejected - pick a clearly DIFFERENT subject, material and action (not a variation of these): {'; '.join(avoid)}\n" if avoid else "") +
            f"Target spoken length: about {_budget(series, duration, single)} words in total across all scenes ({duration} seconds). {_shape(series, lo, hi)}\n"
            f"Title: max 70 characters, specific and curious, not clickbait lies.\n"
            f"Return JSON shaped exactly like this example (different content):\n{EXAMPLE}\n"
            "Every scene MUST have narration, visual and stock_query. hashtags: 4-8 words without #.")
    return llm.chat_json(system, user, validate=lambda d: _fit(d, series, duration, single), temperature=0.85)


def revise_script(series: str, previous: dict, instructions: str, language: str, duration: int) -> dict:
    """Rewrite an existing script following the user's instructions; same format rules and validation as generate_script."""
    s = SERIES[series]; single = bool(previous.get("single_shot"))
    lo, hi = scene_range(duration, series, single)
    old = {k: previous.get(k) for k in ("title", "description", "hashtags", "hook")}
    old["scenes"] = [{k: sc.get(k) for k in ("narration", "visual", "stock_query")} for sc in previous.get("scenes", [])]
    system = ("You rewrite scripts for vertical short videos (YouTube Shorts / Reels), narrated by a text-to-speech voice. "
              "The creator was not happy with the current version, so make bold, clearly noticeable changes - never return a near-copy. "
              f"Language: {language}. Keep narration natural to say aloud: no emojis, no hashtags, no stage directions, no markdown.")
    user = (f"Format: {s['label']}\n{s['brief']}\n"
            f"Previous script (for reference only - do NOT copy its sentences):\n{json.dumps(old, ensure_ascii=False)}\n\n"
            f"The creator's instructions: {instructions}\n"
            "Rewrite the WHOLE script around these instructions: let them shape the hook, every scene, the ending, the title and the "
            "description, not just one line. Write fresh sentences throughout; reuse the old wording only where the instructions explicitly "
            "say to keep something.\n"
            f"Target spoken length: about {_budget(series, duration, single)} words in total across all scenes ({duration} seconds). {_shape(series, lo, hi)}\n"
            f"Title: max 70 characters. Return the full revised script as JSON in the same shape.\n"
            "Every scene MUST have narration, visual and stock_query. hashtags: 4-8 words without #.")
    old_text = " ".join(sc["narration"] or "" for sc in old["scenes"]).lower()

    def check(d: Any) -> dict:
        new = _fit(d, series, duration, single)
        same = difflib.SequenceMatcher(None, old_text, " ".join(sc["narration"] for sc in new["scenes"]).lower()).ratio()
        if same > 0.6: raise ValueError(f"the narration is {same:.0%} identical to the previous script; rewrite every scene with new sentences that follow the instructions")
        return new
    return llm.chat_json(system, user, validate=check, temperature=0.95)


def brainstorm(series: str, n: int, avoid: list[str]) -> list[str]:
    s = SERIES[series]
    avoid_txt = "; ".join(avoid[-40:]) or "none"
    user = (f"Give {n} distinct, specific video topic ideas for a short-video channel.\nChannel focus: {s['idea_brief']}.\n"
            f"Do NOT repeat or closely resemble these existing topics: {avoid_txt}\n"
            'Return JSON: {"ideas":["topic one","topic two"]}. Each idea is a short phrase (under 80 characters).')

    def check(d):
        items = d.get("ideas") if isinstance(d, dict) else d
        if not isinstance(items, list): raise ValueError("'ideas' must be an array of strings")
        out, seen = [], {a.lower() for a in avoid}
        for it in items:
            t = re.sub(r"\s+", " ", str(it.get("topic") if isinstance(it, dict) else it)).strip().strip('"')[:100]
            if len(t) >= 6 and t.lower() not in seen: seen.add(t.lower()); out.append(t)
        if not out: raise ValueError("no usable ideas")
        return out
    return llm.chat_json("You are a creative producer for a short-video channel.", user, validate=check, temperature=0.95)[:n]


def gemini_prompt(script: dict) -> str:
    """A ready-to-paste Gemini (Veo) prompt for a single-shot script. The clip is asked for in fast motion; the renderer
    slows it back to natural speed to fill the video. Wording follows what worked in testing (continuity, sound from frame one)."""
    sc = (script.get("scenes") or [{}])[0]
    kids = SERIES.get(script.get("series", ""), {}).get("kids")
    style = ("Charming, colourful 3D animated style, soft lighting, friendly and safe for young children." if kids
             else "Hyper-realistic, ultra-detailed, cinematic and deeply satisfying.")
    return ("Create a vertical 9:16 video, one continuous shot, no cuts, no text or captions on screen.\n\n"
            f"{sc.get('visual', '').strip()}\n\n"
            "Motion: sped up to about 1.5x real speed - the whole action happens briskly and completes within the clip.\n"
            "Continuity: objects stay consistent for the whole shot; nothing that was cut, moved or transformed snaps back to how it was. "
            "Let the final moment settle slowly during the last 2-3 seconds.\n"
            "Sound: rich, crisp sound effects that match the action, starting from the very first frame. No music, no voice.\n"
            f"{style}")
