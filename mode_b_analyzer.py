"""
mode_b_analyzer.py  -  Chordly "Mode B": analyze an acoustic cover.

Pipeline
  1. extract_raw_chords()   -> beat-synced, Viterbi-smoothed chord timeline (librosa)
  2. detect_instrument()    -> which instrument is being played (vision model)
  3. transcribe_vocals()    -> Whisper transcript, romanized (Groq + chords_db)
  4. analyze_cover()        -> combines everything, asks the text model for a chord sheet,
                               then overwrites chord shapes with verified voicings.

Requires: groq, librosa, numpy, python-dotenv  (+ your local chords_db.py)
Needs GROQ_API_KEYS or GROQ_API_KEY in a .env file (or in your environment).
"""

import base64
import glob
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

import librosa
import numpy as np
from dotenv import load_dotenv
from groq import Groq

from chords_db import get_verified_voicing, romanize_indic_text

# Load environment variables (override=True so updated keys are always read)
load_dotenv(override=True)

_raw_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
GROQ_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
VISION_MODEL = "qwen/qwen3.8-27b"
TEXT_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3"

SR = 22050
SELF_TRANSITION = 0.85
ALLOWED_INSTRUMENTS = ["Ukulele", "Violin", "Guitar", "Piano", "Other"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_rate_limit_error(err: Exception) -> bool:
    err_str = str(err).lower()
    status_code = getattr(err, "status_code", None)
    return (
        status_code == 429
        or "429" in err_str
        or "rate_limit" in err_str
        or "rate limit" in err_str
    )


def call_groq_chat(messages: list, model: str, temperature: float = 0.1) -> str:
    """
    Calls Groq chat completions, rotating through GROQ_KEYS on 429 / rate_limit errors.
    """
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys found. Set GROQ_API_KEYS or GROQ_API_KEY in .env.")

    last_err = None
    for idx, key in enumerate(GROQ_KEYS):
        try:
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if _is_rate_limit_error(e):
                last_err = e
                if idx < len(GROQ_KEYS) - 1:
                    print(f"[Groq] Rate limit (429) on key #{idx + 1}; rotating to next key...")
                    continue
            raise

    raise last_err


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _safe_json_loads(text: str):
    """
    Parse JSON from an LLM reply without response_format.
    Handles <think> blocks, ```json fences, text around the JSON, and trailing commas.
    Returns a dict/list, or None if nothing could be parsed.
    """
    if not text:
        return None
    text = _strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):  # fix trailing commas
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _chat(model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.2) -> str:
    """One chat call -> cleaned text."""
    raw = call_groq_chat(messages=messages, model=model, temperature=temperature)
    return _strip_thinking(raw)


# --------------------------------------------------------------------------- #
# 1. Chord extraction (librosa)
# --------------------------------------------------------------------------- #
def _build_chord_templates():
    """24 templates (12 major + 12 minor), each a unit-length 12-bin vector."""
    names, templates = [], []
    for quality, intervals, suffix in (("maj", (0, 4, 7), ""), ("min", (0, 3, 7), "m")):
        for root in range(12):
            vec = np.zeros(12)
            for iv in intervals:
                vec[(root + iv) % 12] = 1.0
            templates.append(vec / np.linalg.norm(vec))
            names.append(NOTE_NAMES[root] + suffix)
    return names, np.array(templates)  # shape (24, 12)


def extract_raw_chords(audio_path: str) -> dict:
    """
    Returns {"bpm": int, "raw_timeline": [{"time": s, "end": s, "chord": "Am"}, ...]}
    """
    y, sr = librosa.load(audio_path, sr=SR)

    # Keep the harmonic (pitched) part, drop drums/percussion
    y_harmonic = librosa.effects.harmonic(y, margin=3.0)

    # Beats are tracked on the original signal
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))

    if len(beat_frames) == 0:
        return {"bpm": bpm, "raw_timeline": []}

    # Chroma = energy in each of the 12 pitch classes, per frame
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # One chroma column per beat (median of the frames inside each beat)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # Beat boundaries (same ones sync() used), converted to seconds
    bounds = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1], pad=True)
    starts = librosa.frames_to_time(bounds[:-1], sr=sr)
    ends = librosa.frames_to_time(bounds[1:], sr=sr)
    n_beats = min(chroma_sync.shape[1], len(starts))
    chroma_sync = chroma_sync[:, :n_beats]

    # Cosine similarity of every beat against every chord template
    names, templates = _build_chord_templates()
    norms = np.linalg.norm(chroma_sync, axis=0, keepdims=True)
    chroma_unit = chroma_sync / np.maximum(norms, 1e-9)
    sim = np.clip(templates @ chroma_unit, 0.0, None)  # (24, n_beats)

    # Turn similarities into per-beat probabilities (squared = sharper contrast)
    prob = sim ** 2 + 1e-6
    prob = prob / prob.sum(axis=0, keepdims=True)

    # Viterbi: stay on the same chord with p=0.85, so 1-beat flickers get smoothed away
    transition = librosa.sequence.transition_loop(24, SELF_TRANSITION)
    states = librosa.sequence.viterbi(prob, transition)

    # Merge consecutive identical chords into timed changes
    timeline = []
    for i, state in enumerate(states):
        chord = names[int(state)]
        if timeline and timeline[-1]["chord"] == chord:
            timeline[-1]["end"] = round(float(ends[i]), 2)
        else:
            timeline.append({
                "time": round(float(starts[i]), 2),
                "end": round(float(ends[i]), 2),
                "chord": chord,
            })

    return {"bpm": bpm, "raw_timeline": timeline}


# --------------------------------------------------------------------------- #
# 2. Instrument detection (vision model, text fallback)
# --------------------------------------------------------------------------- #
def _encode_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _match_instrument(reply: str):
    """Find one of the allowed words inside the model's reply."""
    reply = (reply or "").lower()
    for name in ALLOWED_INSTRUMENTS:
        if re.search(rf"\b{name.lower()}\b", reply):
            return name
    return None


def _detect_instrument_from_title(title: str) -> str:
    prompt = (
        f'Video title: "{title}"\n'
        "Which instrument is most likely being played in this cover? "
        "Reply with ONLY ONE word from: Ukulele, Violin, Guitar, Piano, Other."
    )
    try:
        reply = _chat(TEXT_MODEL, [{"role": "user", "content": prompt}], max_tokens=512, temperature=0)
        return _match_instrument(reply) or "Other"
    except Exception as e:
        print(f"[detect_instrument] text fallback failed: {e}")
        return "Other"


def detect_instrument(frame_paths: list, title: str) -> str:
    """Returns one of: Ukulele, Violin, Guitar, Piano, Other."""
    frames = [p for p in (frame_paths or []) if p and os.path.exists(p)][:2]
    if not frames:
        return _detect_instrument_from_title(title)

    content = [{
        "type": "text",
        "text": (
            "Look at these video frames from a music cover. Which instrument is being played? "
            "Reply with ONLY ONE word from: Ukulele, Violin, Guitar, Piano, Other."
        ),
    }]
    for path in frames:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(path)}})

    try:
        reply = _chat(VISION_MODEL, [{"role": "user", "content": content}], max_tokens=512, temperature=0)
        found = _match_instrument(reply)
        if found:
            return found
    except Exception as e:
        print(f"[detect_instrument] vision call failed: {e}")

    return _detect_instrument_from_title(title)


# --------------------------------------------------------------------------- #
# 3. Vocal transcription (Whisper on Groq) + romanization
# --------------------------------------------------------------------------- #
def transcribe_vocals(audio_path: str, title: str = "") -> str:
    """Whisper transcript -> romanized text (Hinglish/Manglish/Tanglish). Returns "" if nobody sings."""
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys found. Set GROQ_API_KEYS or GROQ_API_KEY in .env.")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = None
    last_err = None
    for idx, key in enumerate(GROQ_KEYS):
        try:
            client = Groq(api_key=key)
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_bytes),
                model=WHISPER_MODEL,
                prompt=f"{title}",  # title only: instruction words get echoed back by Whisper,
                response_format="text",
            )
            break
        except Exception as e:
            if _is_rate_limit_error(e):
                last_err = e
                if idx < len(GROQ_KEYS) - 1:
                    print(f"[Groq Whisper] Rate limit (429) on key #{idx + 1}; rotating to next key...")
                    continue
            raise
    else:
        raise last_err

    # With response_format="text" the SDK may return a plain string or an object with .text
    transcript = result if isinstance(result, str) else getattr(result, "text", str(result))
    romanized = romanize_indic_text(transcript.strip())
    # No singing? Whisper often hallucinates "Thank you." etc. -> treat that as empty ("")
    return _clean_whisper_transcript(romanized, title)


# --------------------------------------------------------------------------- #
# 4. Full analysis
# --------------------------------------------------------------------------- #
REQUIRED_KEYS = [
    "identified_song", "original_artist", "detected_instrument", "arranged_for_instrument",
    "estimated_key", "tempo_bpm", "playing_technique", "capo_tip", "chord_voicings",
    "clean_progression", "play_along_sections", "sung_transcript",
]


def _clean_title_for_search(title: str) -> str:
    """'Sawaar Loon Ukulele Cover | Monali Thakur #shorts' -> 'Sawaar Loon Monali Thakur'."""
    text = re.sub(r"#\S+|https?://\S+", " ", title or "")
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)  # drop (Official Video), [Lyrics] ...
    text = re.sub(
        r"\b(ukulele|guitar|piano|violin|acoustic|unplugged|cover|covered by|lyrics?|"
        r"lyrical|official|video|audio|reels?|shorts?|chords?|tutorial|version|live)\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"[|\-_–—:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_official_lyrics(title: str, max_chars: int = 3000) -> str:
    """
    Look the song up on LRCLIB and return its lyrics, romanized, to use as a SPELLING
    reference only. Picks the BEST matching result (not just the first one).
    Returns "" if nothing good is found or the network fails (never crashes).
    """
    clean_title = _clean_title_for_search(title)
    if not clean_title:
        return ""

    url = "[https://lrclib.net/api/search?q=](https://lrclib.net/api/search?q=)" + urllib.parse.quote(clean_title)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Chordly/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            results = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[lrclib] lookup failed: {e}")
        return ""

    wanted = clean_title.lower()
    best_item, best_score = None, 0.0
    for item in results or []:
        lyrics = item.get("plainLyrics") or ""
        if item.get("instrumental") or not lyrics.strip():
            continue
        track = str(item.get("trackName", "")).lower()
        track_and_artist = f"{track} {str(item.get('artistName', '')).lower()}"
        score = max(
            SequenceMatcher(None, wanted, track).ratio(),
            SequenceMatcher(None, wanted, track_and_artist).ratio(),
        )
        words = re.findall(r"\w+", track)
        if words and all(word in wanted for word in words):  # whole track name is inside the title
            score = max(score, 0.9)
        if score > best_score:
            best_item, best_score = item, score

    if best_item is None or best_score < 0.45:
        return ""

    lyrics = re.sub(r"#\S+|https?://\S+", "", best_item["plainLyrics"])  # no hashtags / links
    lyrics = romanize_indic_text(lyrics.strip())
    header = f"{best_item.get('trackName', '')} - {best_item.get('artistName', '')}".strip(" -")
    return (header + "\n" + lyrics)[:max_chars]


# --------------------------------------------------------------------------- #
# Instrumental (no vocals) covers
# --------------------------------------------------------------------------- #
INSTRUMENTAL_TEXT = "(Instrumental cover — no vocals detected)"

# Short filler sentences Whisper makes up when it only hears music or silence
_FILLER_SENTENCES = {
    "thank you", "thank you so much", "thank you very much", "thank you for watching",
    "thanks for watching", "thanks for watching and see you next time", "thanks",
    "please subscribe", "subscribe", "like and subscribe", "please like and subscribe",
    "bye", "bye bye", "goodbye", "you", "music", "applause", "laughter", "silence",
    "subtitles by the amara org community",
    # the same thing in romanized Hindi / Tamil / Malayalam / Telugu
    "dhanyavaad", "dhanyavad", "shukriya", "nandri", "nanni", "dhanyavadalu",
}


def _normalize_words(text: str) -> str:
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", (text or "").lower())  # [Music], (applause)
    text = re.sub(r"[^\w\s]", " ", text)                           # punctuation, music notes
    return re.sub(r"\s+", " ", text).strip()


def _clean_whisper_transcript(text: str, title: str = "") -> str:
    """
    Whisper invents short phrases ("Thank you.", "Thanks for watching.", "Bye.") when there is no singing.
    Drops those filler sentences; if fewer than 5 real words are left (or it is just the title echoed
    back), the transcript counts as empty ("").
    """
    text = re.sub(r"\[.*?\]|\(.*?\)|[♪♫]+", " ", text or "")
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    kept = [piece.strip() for piece in pieces
            if _normalize_words(piece) and _normalize_words(piece) not in _FILLER_SENTENCES]
    cleaned = " ".join(kept).strip()

    norm = _normalize_words(cleaned)
    words = norm.split()
    if len(words) < 5:
        return ""
    if norm == _normalize_words(title):               # Whisper just repeated the prompt (the title)
        return ""
    if len(set(words)) == 1:                          # "la la la la la" / "you you you you"
        return ""
    return cleaned


def _is_instrumental(transcript: str, title: str = "") -> bool:
    """True when nobody is singing (the cleaned transcript is empty)."""
    return not _clean_whisper_transcript(transcript, title)


def _canon_chord(name) -> str:
    """'d' -> 'D', 'f#m' -> 'F#m', 'Bb' stays 'Bb' (only the root letter is case-fixed)."""
    text = str(name).strip().replace("♯", "#").replace("♭", "b")
    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", text)
    return match.group(1).upper() + match.group(2) + match.group(3) if match else text


def _dedupe_chords(chords) -> list:
    """Unique chord names, first-seen order, same chord written in different case counted once."""
    seen, unique = set(), []
    for chord in chords or []:
        name = _canon_chord(chord)
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _strip_tags(line: str) -> str:
    """'Sa[Am]waar [G]loon' -> 'Sawaar loon'"""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\[\]]+\]", "", str(line))).strip()


def _line_texts(sections) -> list:
    lines = []
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        raw = section.get("lines", [])
        lines += raw.splitlines() if isinstance(raw, str) else [str(x) for x in (raw or [])]
    return lines


def _lyric_lines(sections) -> list:
    """Lines that contain real words (not bar lines, not cue lines)."""
    found = []
    for line in _line_texts(sections):
        text = line.strip()
        if not text or text.lower().startswith("cue"):
            continue
        if re.search(r"[^\W\d_]{2,}", re.sub(r"\[[^\[\]]+\]", "", text)):
            found.append(text)
    return found


def _keep_only_bars_and_cues(sections) -> list:
    """Instrumental covers: keep ONLY chord-bar lines and cue lines - zero lyrics, zero filler like 'Thank you'."""
    cleaned = []
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        keep = []
        for line in _line_texts([section]):
            text = line.strip()
            norm = _normalize_words(text)
            if not text or any(bad in norm for bad in ("thank you", "thanks for watching", "subscribe")):
                continue
            leftover = re.sub(r"\[[^\[\]]+\]|[/|\s]", "", text)
            if (not leftover and "[" in text) or text.lower().startswith("cue"):
                keep.append(text)
        if any("[" in line for line in keep):
            cleaned.append({"section_name": section.get("section_name", "Bars"), "lines": keep})
    return cleaned


_CUES = {
    "Violin": "Cue: long smooth bows - down-bow on beat 1, up-bow on beat 3; play the double stop on each chord.",
    "Ukulele": "Cue: strum D - D U - U D U (one bar = 4 beats); play the first bar softly.",
    "Guitar": "Cue: strum D - D U - U D U, or pick each chord low to high across the 4 beats.",
    "Piano": "Cue: left hand plays the root on beat 1; right hand holds the chord on beats 1 and 3.",
}


def _build_bar_sections(timeline: list, bpm: int, instrument: str,
                        bars_per_line: int = 4, lines_per_section: int = 4, max_bars: int = 64) -> list:
    """
    Deterministic bar sheet straight from the detected chords (used when the LLM gives none):
    one bar = 4 beats, "[Fm] / / / | [G] / / /", with a bowing/strumming cue at the top of each section.
    """
    if not timeline:
        return []

    seconds_per_beat = 60.0 / max(int(bpm), 1)
    beats = []  # (chord, starts_here)
    for seg in timeline:
        count = max(1, round((seg["end"] - seg["time"]) / seconds_per_beat))
        for k in range(count):
            beats.append((seg["chord"], k == 0))

    bars = []
    for i in range(0, len(beats), 4):
        group = beats[i:i + 4]
        parts = []
        for j, (chord, starts_here) in enumerate(group):
            parts.append(f"[{chord}]" if (j == 0 or starts_here) else "/")
        parts += ["/"] * (4 - len(parts))
        bars.append(" ".join(parts))
    bars = bars[:max_bars]

    cue = _CUES.get(instrument, "Cue: keep a steady 4-beat pulse and change chords on the first beat of each bar.")
    bar_lines = [" | ".join(bars[i:i + bars_per_line]) for i in range(0, len(bars), bars_per_line)]

    sections = []
    for s in range(0, len(bar_lines), lines_per_section):
        first_bar = s * bars_per_line + 1
        last_bar = min((s + lines_per_section) * bars_per_line, len(bars))
        sections.append({
            "section_name": f"Bars {first_bar}-{last_bar}",
            "lines": [cue] + bar_lines[s:s + lines_per_section],
        })
    return sections


def _has_bar_sheet(sections) -> bool:
    """True if at least one line looks like a chord bar line, e.g. '[Fm] / / / | [G] / / /'."""
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        for line in section.get("lines", []) or []:
            if "[" in str(line) and "/" in str(line):
                return True
    return False


def _is_bar_or_cue_line(line: str) -> bool:
    text = str(line).strip()
    return ("|" in text and "/" in text) or text.lower().startswith("cue")


def _enforce_short_lines(sections, max_words: int = 7, skip_split: bool = False) -> list:
    """
    Safety net: if the model still returns a long line (or one big string), split it into
    short singable lines of about 4-7 words. Chord tags like [Am] stay attached to their word.
    Bar lines ("[C] / / / | [G] / / /") and cue lines are never split.
    """
    fixed = []
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        lines = section.get("lines", [])
        if isinstance(lines, str):                       # one giant string -> list of lines
            lines = lines.splitlines() or [lines]
        new_lines = []
        for line in lines:
            if skip_split or _is_bar_or_cue_line(line):
                if str(line).strip():
                    new_lines.append(str(line).strip())
                continue
            words = str(line).split()
            if len(words) <= max_words:
                if words:
                    new_lines.append(" ".join(words))
                continue
            pieces = math.ceil(len(words) / max_words)   # balanced chunks, e.g. 12 -> 6 + 6
            size = math.ceil(len(words) / pieces)
            for i in range(0, len(words), size):
                new_lines.append(" ".join(words[i:i + size]))
        fixed.append({"section_name": section.get("section_name", "Section"), "lines": new_lines})
    return fixed


def _timeline_for_prompt(timeline: list, max_items: int = 120) -> str:
    items = [f'{t["time"]}s {t["chord"]}' for t in timeline[:max_items]]
    return " | ".join(items) if items else "(no chords detected)"


ANALYSIS_SYSTEM_PROMPT = """You are an expert acoustic-cover music teacher who writes play-along chord sheets.

RULES

1. SONG IDENTITY
   Use the video title, the sung transcript and the official lyrics reference (if given) to identify
   the real song and its original artist. If you are not sure, use "Unknown" - never guess wildly.

2. ONLY WHAT WAS SUNG
   Include ONLY the lines that were actually sung in the cover video's transcript.
   NEVER invent extra lines, and NEVER add a Chorus (or any section) that was not sung in the audio.
   If there are no sung lyrics at all (an instrumental cover), never invent any - follow rule 8 instead.

3. IGNORE SPOKEN INTROS AND OUTROS
   Creators often talk before or after playing, for example "Hey guys, welcome back to my channel",
   "Comment below which song next", "Thanks for watching, like and subscribe".
   COMPLETELY IGNORE all spoken talking, greetings, intros, outros and chit-chat.
   play_along_sections must start strictly from the FIRST SUNG LYRIC LINE and end at the last sung lyric line.
   None of the spoken talk may appear in play_along_sections or in sung_transcript.
   If the transcript holds ONLY spoken talking and no singing at all, treat the cover as instrumental:
   return "" for sung_transcript and follow rule 8.

4. ZERO-TYPO SPELLING: NATURAL, AND CROSS-CHECKED WITH THE OFFICIAL LYRICS
   The transcript was auto-transcribed and romanized, so it uses strict ITRANS-style spellings
   ("havaa", "rahee", "jindagee", "kutichchu") and can contain phonetic mishearings.
   a) Rewrite every word in natural, everyday conversational Hinglish / Manglish / Tanglish
      as people type in chats: "hawa", "rahi", "zindagi", "sawaar", "kudichu".
   b) If an official lyrics reference is given, use ITS exact spelling and words to fix every mishearing in the
      transcript (for example "jhooti" -> "jhonke", "ut gaye" -> "rooth gaye"). Match each sung line to the
      closest official line and use the official words.
   c) The official lyrics are a SPELLING REFERENCE ONLY. Keep ONLY the exact lines that were actually sung in the
      cover video. Never add a line just because it exists in the official lyrics.
      If the official lyrics clearly belong to a different song than the one sung, ignore them.

5. SHORT, SINGABLE LINES (sung covers)
   In every play_along_sections entry, "lines" MUST be a JSON list of short singable lines:
   4 to 7 words each, split at natural musical breaths and rhymes.
   NEVER put a verse, a sentence or a paragraph into one long string. Split it into several short lines.
   (This does not apply to the instrumental bar lines of rule 8.)

6. CHORD MARKERS MID-LINE, RIGHT WHERE THE CHORD CHANGES
   Place each [Chord] tag directly before the exact syllable or word where that chord change happens,
   in the middle of the line when that is where it changes. A line usually has 1 to 3 tags.
   Example of a correct line: "[Fm]Hawa ke jhonke [G]aaj mausam se [F]rooth gaye"
   Use ONLY chords from the detected chord list, written as plain names (C, Am, F#, Bm).

7. OUTPUT FORMAT
   Reply with ONE JSON object only - no markdown fences, no commentary.

8. INSTRUMENTAL COVERS (no vocals)
   If the input says INSTRUMENTAL COVER, nobody sings, so there are NO lyrics. Do not invent any words.
   Instead write a clean measure-by-measure bar sheet in play_along_sections:
   - one line = 4 bars (measures) separated by "|", and every bar = 4 beats;
   - put the [Chord] tag on the beat where the chord lands and "/" for every beat that keeps ringing, e.g.
     "[D] / / / | [A] / / / | [Bm] / / / | [G] / / /"   or, for a mid-bar change,   "[C] / [G] /";
   - use the detected chord timeline and the BPM to decide how many beats each chord lasts, in the order played;
   - group the lines into named sections (for example "Intro", "Theme A", "Theme B"), 2 to 4 bar lines each;
   - make the FIRST line of every section a short cue line without chord tags, starting with "Cue:", with a
     bowing cue for Violin (down-bow / up-bow, long or short bows), a strumming or picking cue for Ukulele and
     Guitar (for example "D - D U - U D U"), or a hand pattern for Piano, written for the arranged instrument.
   Write ZERO lyrics and no filler words of any kind (never "Thank you", "Thanks for watching", "Subscribe").
   sung_transcript must be exactly "(Instrumental cover — no vocals detected)".
"""


def _build_prompt(title, bpm, timeline, detected, arranged, transcript,
                  reference_lyrics="", instrumental=False) -> str:
    if instrumental:
        transcript_block = "(INSTRUMENTAL COVER - no vocals detected. Follow rule 8: write a bar sheet, no lyrics.)"
        reference_block = "(not used for instrumental covers)"
        extra_notes = (
            "- This is an INSTRUMENTAL COVER. play_along_sections example:\n"
            '  [{"section_name": "Intro", "lines": ["Cue: down-bow on beat 1, up-bow on beat 3", '
            '"[Fm] / / / | [G] / / / | [F] / / / | [Dm] / / /"]}]\n'
            f'- sung_transcript must be exactly "{INSTRUMENTAL_TEXT}".\n'
        )
    else:
        transcript_block = transcript or "(no vocals detected)"
        reference_block = (
            reference_lyrics
            if reference_lyrics
            else "(none found - rely on the transcript and your knowledge of the song)"
        )
        extra_notes = (
            "- sung_transcript: the cleaned sung lyrics only (no spoken intro/outro), natural spelling, one short line per\n"
            '  musical phrase, with NO chord markers. If nobody actually sings, return "" (see rules 3 and 8).\n'
        )

    return f"""INPUT
- Video title: {title or "unknown"}
- Detected instrument in video: {detected}
- Arrange for instrument: {arranged}
- Tempo (BPM): {bpm}
- Beat-synced chord changes detected from the audio (time + chord): {_timeline_for_prompt(timeline)}
- Romanized sung transcript from the cover video (may start/end with spoken talking, and may contain
  mishearings and strict ITRANS spellings):
\"\"\"
{transcript_block}
\"\"\"
- Official lyrics from LRCLIB (SPELLING REFERENCE ONLY - do NOT add lines that were not sung in the cover):
\"\"\"
{reference_block}
\"\"\"

FIELD NOTES
- clean_progression: the main repeating progression as chord names, e.g. ["C","G","Am","F"].
- chord_voicings: list every chord used as a key; the value can be a placeholder string (it will be replaced).
- tempo_bpm must be {bpm}. detected_instrument must be "{detected}". arranged_for_instrument must be "{arranged}".
- capo_tip: one short sentence (or "No capo needed"). playing_technique: one short sentence for {arranged}.
{extra_notes}
Reply with ONLY one JSON object with EXACTLY these keys:
{{
  "identified_song": "",
  "original_artist": "",
  "detected_instrument": "",
  "arranged_for_instrument": "",
  "estimated_key": "",
  "tempo_bpm": 0,
  "playing_technique": "",
  "capo_tip": "",
  "chord_voicings": {{}},
  "clean_progression": [],
  "play_along_sections": [{{"section_name": "Verse 1", "lines": ["[Fm]Hawa ke jhonke [G]aaj", "[G]mausam se [F]rooth gaye"]}}],
  "sung_transcript": ""
}}"""


def _collect_chords(result: dict) -> list:
    """Every distinct chord used anywhere in the result, in order of appearance."""
    seen = []

    def add(c):
        c = str(c).strip()
        if c and c not in seen:
            seen.append(c)

    for c in result.get("clean_progression", []) or []:
        add(c)
    for sec in result.get("play_along_sections", []) or []:
        for line in (sec.get("lines", []) if isinstance(sec, dict) else []):
            for c in re.findall(r"\[([^\[\]]+)\]", str(line)):
                add(c)
    for c in (result.get("chord_voicings") or {}).keys():
        add(c)
    return seen


def analyze_cover(media_data: dict, target_instrument: str = None) -> dict:
    audio_path = media_data["audio_path"]
    frame_paths = media_data.get("frame_paths", [])
    title = media_data.get("title", "")

    # Gather the ingredients
    raw = extract_raw_chords(audio_path)
    detected = detect_instrument(frame_paths, title)
    transcript = transcribe_vocals(audio_path, title)   # already cleaned: "" when nobody sings

    # Same rule for every instrument (Violin, Piano, Guitar, Ukulele): singing or not?
    instrumental = _is_instrumental(transcript, title)
    # LRCLIB spelling reference (only useful when someone actually sings)
    reference_lyrics = "" if instrumental else _fetch_official_lyrics(title)

    # Which instrument do we arrange for?
    if target_instrument:
        final_instrument = target_instrument
    else:
        final_instrument = detected if detected != "Other" else "Guitar"

    prompt = _build_prompt(
        title, raw["bpm"], raw["raw_timeline"], detected, final_instrument,
        transcript, reference_lyrics, instrumental,
    )

    # Ask the text model (retry once if the JSON is unreadable)
    result = None
    for _ in range(2):
        reply = _chat(
            TEXT_MODEL,
            [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        parsed = _safe_json_loads(reply)
        if isinstance(parsed, dict):
            result = parsed
            break
    if result is None:
        result = {}

    # Fill in anything missing, and force the facts we measured ourselves
    defaults = {
        "identified_song": "Unknown", "original_artist": "Unknown", "estimated_key": "Unknown",
        "playing_technique": "", "capo_tip": "No capo needed",
        "chord_voicings": {}, "clean_progression": [], "play_along_sections": [],
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    result["detected_instrument"] = detected
    result["arranged_for_instrument"] = final_instrument
    result["tempo_bpm"] = raw["bpm"]
    result["raw_transcript"] = transcript  # what Whisper + romanizer produced after filler removal

    sections = result.get("play_along_sections")
    if not instrumental:
        # The model may have found that the "transcript" was only spoken talking (no real singing)
        sung = str(result.get("sung_transcript", "")).strip()
        if sung.startswith("(Instrumental"):
            sung = ""
        lyric_lines = _lyric_lines(sections)
        if not sung and not lyric_lines:
            instrumental = True
        elif not sung:
            sung = "\n".join(_strip_tags(line) for line in lyric_lines)

    result["is_instrumental"] = instrumental
    if instrumental:
        # NO vocals: only clean chord bars + cues, never lyrics or "Thank you"
        result["sung_transcript"] = INSTRUMENTAL_TEXT
        sections = _keep_only_bars_and_cues(sections)
        if not _has_bar_sheet(sections):
            sections = _build_bar_sections(raw["raw_timeline"], raw["bpm"], final_instrument)
        result["play_along_sections"] = _enforce_short_lines(sections, skip_split=True)
    else:
        # Vocals: chords-over-lyrics, 4-7 words per line, [Chord] markers mid-line
        result["sung_transcript"] = sung
        result["play_along_sections"] = _enforce_short_lines(sections)

    # Never any duplicate chords in clean_progression or chord_voicings
    result["clean_progression"] = _dedupe_chords(result.get("clean_progression"))
    if not result["clean_progression"]:  # fallback: unique chords in order of appearance
        result["clean_progression"] = _dedupe_chords(t["chord"] for t in raw["raw_timeline"])[:8]

    # Deterministic Voicing Override: every shape comes from your verified database
    verified = {}
    for chord in _dedupe_chords(_collect_chords(result)):
        try:
            verified[chord] = get_verified_voicing(
                chord, final_instrument, default="(shape not in database yet)"
            )
        except Exception as e:
            print(f"[voicing] could not get voicing for {chord}: {e}")
            verified[chord] = None
    result["chord_voicings"] = verified

    # Keep the raw timeline handy for debugging / UI
    result["raw_timeline"] = raw["raw_timeline"]
    return result


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def _two_line_layout(line: str):
    """
    Turn "Sa[Am]waar [G]loon" into two aligned lines (chords above lyrics):
        Am     G
        Sawaar loon
    Returns (chord_line, lyric_line). chord_line is "" when the line has no chords.
    """
    parts = re.split(r"\[([^\[\]]+)\]", str(line))  # text, chord, text, chord, ...
    lyric_line = ""
    chord_line = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:            # plain lyric text
            lyric_line += part
        else:                     # a chord: sit above the next syllable
            position = len(lyric_line)
            if chord_line:        # never let two chords touch
                position = max(position, len(chord_line) + 1)
            chord_line = chord_line.ljust(position) + part
    return chord_line.rstrip(), lyric_line.rstrip()


def format_chord_sheet(data: dict) -> str:
    lines = []
    lines.append(f"{data.get('identified_song', 'Unknown')}  -  {data.get('original_artist', 'Unknown')}")
    lines.append("=" * 60)
    lines.append(f"Instrument in video : {data.get('detected_instrument', '?')}")
    lines.append(f"Arranged for        : {data.get('arranged_for_instrument', '?')}")
    lines.append(f"Key: {data.get('estimated_key', '?')}    Tempo: {data.get('tempo_bpm', '?')} BPM")
    lines.append(f"Technique : {data.get('playing_technique', '')}")
    lines.append(f"Capo      : {data.get('capo_tip', '')}")
    lines.append("")
    lines.append("Progression: " + " - ".join(data.get("clean_progression", [])))
    lines.append("")
    lines.append("CHORD SHAPES")
    lines.append("-" * 60)
    for chord, voicing in (data.get("chord_voicings") or {}).items():
        lines.append(f"  {chord:<6} {voicing}")
    lines.append("")

    # Ultimate-Guitar style: chord line above lyric line (use a monospaced font to view)
    for section in data.get("play_along_sections", []):
        lines.append(f"[{section.get('section_name', 'Section')}]")
        for raw_line in section.get("lines", []):
            chord_line, lyric_line = _two_line_layout(raw_line)
            if chord_line:
                lines.append(chord_line)
            if lyric_line:
                lines.append(lyric_line)
        lines.append("")

    lines.append("SUNG TRANSCRIPT")
    lines.append("-" * 60)
    lines.append(data.get("sung_transcript", "") or "(none)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Test block:  python mode_b_analyzer.py
# Looks inside ./temp_media for one audio file and any .jpg/.png frames.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # so Windows can print non-English text

    media_dir = "temp_media"
    audio_files = []
    for pattern in ("*.mp3", "*.wav", "*.m4a", "*.webm", "*.ogg", "*.flac"):
        audio_files.extend(glob.glob(os.path.join(media_dir, pattern)))
    frames = sorted(
        glob.glob(os.path.join(media_dir, "*.jpg"))
        + glob.glob(os.path.join(media_dir, "*.jpeg"))
        + glob.glob(os.path.join(media_dir, "*.png"))
    )

    if not audio_files:
        print(f"No audio file found in '{media_dir}'. Run mode_b_media.py first.")
        sys.exit(1)

    test_media = {
        "audio_path": audio_files[0],
        "frame_paths": frames[:2],
        "title": "Sawaar Loon Ukulele Cover",
    }
    print(f"Analyzing: {test_media['audio_path']}")
    print(f"Frames   : {test_media['frame_paths']}\n")

    result = analyze_cover(test_media)
    print(format_chord_sheet(result))