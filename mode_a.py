import os
import json
import re
import urllib.request
import urllib.parse
from difflib import SequenceMatcher
from typing import Dict, List, Any

# Third-party libraries (assumed installed per requirements)
import yt_dlp
from ddgs import DDGS
from groq import Groq
from dotenv import load_dotenv

# Local helper module (provided by user)
from chords_db import get_verified_voicing, romanize_indic_text

# Load environment variables (override=True so updated keys are always read)
load_dotenv(override=True)

_raw_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
GROQ_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# Constants
MODEL_NAME = "openai/gpt-oss-120b"


def call_groq_chat(messages: list, model: str, temperature: float = 0.1) -> str:
    """
    Calls Groq chat completions, rotating through GROQ_KEYS on 429 / rate_limit errors.
    """
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys found. Set GROQ_API_KEYS or GROQ_API_KEY in .env.")

    last_err = None
    for idx, key in enumerate(GROQ_KEYS):
        try:
            groq_client = Groq(api_key=key)
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            status_code = getattr(e, "status_code", None)
            if status_code == 429 or "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                last_err = e
                if idx < len(GROQ_KEYS) - 1:
                    print(f"[Groq] Rate limit (429) on key #{idx + 1}; rotating to next key...")
                    continue
            raise

    raise last_err


# Rule for natural spelling of romanized lyrics (romanize_indic_text outputs strict ITRANS)
NATURAL_SPELLING_RULE = """
NATURAL SPELLING RULE:
The lyrics may be in strict ITRANS-style romanization, e.g. "havaa", "jindagee", "kutichchu".
Convert every such word into natural, everyday conversational Hinglish / Manglish / Tanglish,
the way people type in chats: "hawa", "zindagi", "kudichu".
Also fix minor spelling typos using the real lyrics of this song.
Do NOT add, remove or reorder sung lines while doing this.
"""

NO_DIALOGUE_RULE = """
NO MOVIE DIALOGUES RULE:
The lyrics text may contain spoken movie dialogues (before, between or after the song).
Remove ALL spoken dialogue, scene descriptions, credits and promo text.
Include ONLY the sung verses and choruses of the song.
"""

# Used when neither LRCLIB nor the user gave real lyrics: never guess words
NO_LYRICS_RULE = """
NO LYRICS AVAILABLE RULE:
There are NO verified lyrics for this song. Do NOT guess, invent or paraphrase any lyric words.
Output ONLY clean 4-beat chord bars in "lines": one line = 4 bars separated by "|", each bar = 4 beats,
with the chord tag on the first beat and "/" for each remaining beat, for example:
"[C] / / / | [G] / / / | [Am] / / / | [F] / / /"
Use the song's real chord progression (from the context if given, otherwise the most likely one for its key),
grouped into sections such as Intro, Verse and Chorus. The lines may contain NOTHING except chord tags, "/" and "|".
"""

# Phrases that mean a YouTube description is promo text, not lyrics
PROMO_MARKERS = ("subscribe", "music label", "directed by")


def _clean_reference_text(text: str, is_description: bool = False) -> str:
    """
    Cleans text BEFORE it is sent to Groq:
      - removes #hashtags and http(s) links
      - for YouTube descriptions: ignores the whole text if it looks like promo
        ("subscribe", "music label", "directed by")
    """
    if not text:
        return ""
    if is_description and any(marker in text.lower() for marker in PROMO_MARKERS):
        return ""
    text = re.sub(r"#\S+|https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)          # tidy leftover spaces
    text = re.sub(r" *\n *", "\n", text)         # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)       # and blank-line runs
    return text.strip()


def _safe_json_loads(raw_text: str) -> dict:
    """
    Cleans Groq's response by stripping Markdown code fences (```json ... ```)
    and parsing the result into a Python dictionary.
    """
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Fallback: try to find anything between { and }
        match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


# ---------------------------------------------------------------------------
# Song + singer verification (iTunes and LRCLIB)
# ---------------------------------------------------------------------------
def _fetch_json(url: str, timeout: int = 10):
    """GET a URL and return parsed JSON, or None if anything goes wrong."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Chordly/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception:
        return None


def _norm(text: str) -> str:
    """Lowercase, drop (From "Movie") brackets and punctuation, squeeze spaces."""
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text or "")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _split_artists(text: str) -> list:
    """'Arijit Singh, Shreya Ghoshal' -> ['arijit singh', 'shreya ghoshal']"""
    parts = re.split(r"\s*(?:,|&|/|\bfeat\.?|\bft\.?|\band\b|\bwith\b)\s*", text or "", flags=re.IGNORECASE)
    return [n for n in (_norm(p) for p in parts) if n]


def _artists_match(a: str, b: str) -> bool:
    """True if the two singer lists share at least one singer (spelling-tolerant: 'K. S. Harisankar' = 'KS Harisankar')."""
    for x in _split_artists(a):
        for y in _split_artists(b):
            if _similar(x, y) >= 0.8 or (min(len(x), len(y)) >= 6 and (x in y or y in x)):
                return True
    return False


def _plain_title(title: str) -> str:
    """'Tum Hi Ho (From "Aashiqui 2")' -> 'Tum Hi Ho'"""
    return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", title or "").strip() or (title or "").strip()


COVER_WORDS = re.compile(r"\b(cover|karaoke|tribute|instrumental|lullaby|made famous|originally performed)\b", re.IGNORECASE)


def _lrclib_candidates(query: str, items: list, seen_ids: set) -> None:
    """Add LRCLIB search results for a query to `items` (no duplicates)."""
    found = _fetch_json("https://lrclib.net/api/search?q=" + urllib.parse.quote(query))
    for item in found or []:
        if item.get("id") not in seen_ids:
            seen_ids.add(item.get("id"))
            items.append(item)


def _verify_song_and_singer(song_title: str, artist: str) -> dict:
    """
    Cross-checks the song with iTunes (up to 5 results) AND LRCLIB's artistName, so the playback
    singer is right and the lyrics belong to the right song.
    Returns {"song_title", "artist", "lyrics", "singer_verified"}.
    """
    singer_verified = False
    original = (_norm(song_title), _norm(artist))

    # LRCLIB first: its artistName is a second opinion on who really sings the song
    lrc_items, seen_ids = [], set()
    _lrclib_candidates(f"{song_title} {artist}".strip(), lrc_items, seen_ids)
    if artist:
        _lrclib_candidates(song_title, lrc_items, seen_ids)
    lrc_artists = [
        item.get("artistName", "") for item in lrc_items
        if (item.get("plainLyrics") or "").strip() and _similar(song_title, item.get("trackName", "")) >= 0.6
    ]

    # iTunes (5 results): best title match; prefer the result whose singer matches what we know
    term = urllib.parse.quote(f"{song_title} {artist}".strip())
    itunes = _fetch_json(f"https://itunes.apple.com/search?term={term}&entity=song&limit=5") or {}
    best_itunes, best_score = None, 0.0
    for item in itunes.get("results", []):
        track, singer = item.get("trackName", ""), item.get("artistName", "")
        title_score = _similar(song_title, track)
        if title_score < 0.5:
            continue
        score = title_score
        if artist and _artists_match(artist, singer):
            score += 0.5                                      # matches the singer we were told
        if any(_artists_match(singer, lrc_artist) for lrc_artist in lrc_artists):
            score += 0.4                                      # LRCLIB agrees on the singer
        if COVER_WORDS.search(f"{track} {singer}"):
            score -= 0.5                                      # cover / karaoke / tribute versions
        if score > best_score:
            best_itunes, best_score = (track, singer), score
    if best_itunes:
        song_title, artist = _plain_title(best_itunes[0]), best_itunes[1]
        singer_verified = True

    # If the title or singer changed, also look up LRCLIB with the corrected names
    if (_norm(song_title), _norm(artist)) != original:
        _lrclib_candidates(f"{song_title} {artist}".strip(), lrc_items, seen_ids)

    # Pick the LRCLIB entry with the best title match; matching artistName confirms the singer
    best_lrc, best_lrc_score = None, 0.0
    for item in lrc_items:
        lyrics = item.get("plainLyrics") or ""
        if item.get("instrumental") or not lyrics.strip():
            continue
        title_score = _similar(song_title, item.get("trackName", ""))
        if title_score < 0.6:
            continue
        artist_ok = _artists_match(artist, item.get("artistName", ""))
        score = title_score + (0.5 if artist_ok else 0.0)
        if score > best_lrc_score:
            best_lrc, best_lrc_score = (item, artist_ok), score

    lyrics_text = ""
    if best_lrc:
        item, artist_ok = best_lrc
        lyrics_text = _clean_reference_text(item.get("plainLyrics") or "")
        if artist_ok:
            singer_verified = True
        elif not artist:
            # iTunes found nothing and we had no singer: take LRCLIB's
            song_title, artist = _plain_title(item.get("trackName", song_title)), item.get("artistName", "")

    return {"song_title": song_title, "artist": artist, "lyrics": lyrics_text, "singer_verified": singer_verified}


# ---------------------------------------------------------------------------
# YouTube description fallback: only accept text that really looks like lyrics
# ---------------------------------------------------------------------------
CREDIT_LINE = re.compile(
    r"^\s*(singers?|vocals?|music|lyrics?|lyricist|composer|film|movie|album|label|cast|starring|"
    r"producer|director|written|artist|song|credits?|mix|mastered|programming|follow|stream|listen)\b.*$",
    re.IGNORECASE,
)


def _lyrics_from_description(text: str) -> str:
    """Drops credit lines and returns the text only if what is left looks like song lyrics."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln and not CREDIT_LINE.match(ln)]
    if len(lines) < 6:
        return ""
    lengths = sorted(len(ln.split()) for ln in lines)
    long_lines = sum(1 for n in lengths if n > 14)
    if lengths[len(lengths) // 2] > 10 or long_lines > len(lines) * 0.2:
        return ""   # paragraphs, not poetry
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clean chord-bar fallback (no guessed words)
# ---------------------------------------------------------------------------
def _ensure_bar_sections(song_data: dict) -> None:
    """If there are no real lyrics, make sure play_along_sections holds ONLY chord bars."""
    sections = song_data.get("play_along_sections") or []
    clean = bool(sections)
    chords = []
    for section in sections:
        lines = section.get("lines", []) if isinstance(section, dict) else []
        if isinstance(lines, str):
            lines = [lines]
        for line in lines:
            chords += re.findall(r"\[([^\[\]]+)\]", str(line))
            if re.sub(r"\[[^\[\]]+\]|[/|\s]", "", str(line)):   # leftover words = invented lyrics
                clean = False
    if clean and chords:
        return

    base = chords[:16] or ["C", "G", "Am", "F"]
    sequence = list(base)
    while len(sequence) % 4:            # pad to whole 4-bar lines by repeating the progression
        sequence.append(base[len(sequence) % len(base)])
    bars = [f"[{chord}] / / /" for chord in sequence]
    song_data["play_along_sections"] = [{
        "section_name": "Chord Progression (repeat)",
        "lines": [" | ".join(bars[i:i + 4]) for i in range(0, len(bars), 4)],
    }]


SUPPORTED_LANGUAGES = ["Hindi", "Malayalam", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali", "Punjabi", "English"]

# Used only if Groq cannot name a song: very famous songs (title only - iTunes + LRCLIB verify the singer)
FALLBACK_SONGS = {
    "Hindi": "Tum Hi Ho",
    "Malayalam": "Jimikki Kammal",
    "Tamil": "Why This Kolaveri Di",
    "Telugu": "Butta Bomma",
    "Kannada": "Belageddu",
    "Marathi": "Zingaat",
    "Bengali": "Ekla Cholo Re",
    "Punjabi": "Brown Munde",
    "English": "Perfect",
}


def _language_in(text: str) -> str:
    """Which supported language is named in the language/vibe text ('' if none)."""
    lowered = (text or "").lower()
    for language in SUPPORTED_LANGUAGES:
        if language.lower() in lowered:
            return language
    return ""


def _canon_chord(name) -> str:
    """'d' -> 'D', 'f#m' -> 'F#m' (only the root letter's case is fixed)."""
    text = str(name).strip().replace("♯", "#").replace("♭", "b")
    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", text)
    return match.group(1).upper() + match.group(2) + match.group(3) if match else text


def pick_song(mood: str, skill_level: str, language_or_vibe: str = "") -> tuple:
    """
    Asks Groq for ONE very popular, widely known song in the requested language, so that iTunes and
    LRCLIB can find it reliably. Returns (song_title, artist). Never fails: falls back to a famous song.
    """
    language = _language_in(language_or_vibe)
    if language:
        where = f"in {language}"
    else:
        where = f"that matches this language/vibe: {language_or_vibe or 'any'}"

    pick_prompt = f"""
    Pick ONE very popular, widely known real song {where} that fits this mood: {mood}.
    Skill Level of the player: {skill_level}.
    Rules:
    - It must be a famous hit (a well-known film song or chart hit) that is on iTunes / Apple Music and
      has lyrics on LRCLIB. NEVER pick an obscure, indie or invented song.
    - Use the official song title, and the main playback singer(s) as "artist"
      (for example "Arijit Singh", "Shreya Ghoshal", "Vineeth Sreenivasan", "Sid Sriram").
    Return ONLY a JSON object with keys "song_title" and "artist".
    """
    for _ in range(2):
        data = _ask_for_json(pick_prompt)
        if data and str(data.get("song_title", "")).strip():
            return str(data["song_title"]).strip(), str(data.get("artist", "")).strip()

    return FALLBACK_SONGS.get(language, FALLBACK_SONGS["English"]), ""


def _ask_for_json(prompt: str):
    """One Groq call that returns a dict, or None if the call or the JSON parsing failed."""
    try:
        raw_reply = call_groq_chat(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
        )
        data = _safe_json_loads(raw_reply)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _sections_ok(song_data) -> bool:
    """True only if play_along_sections is a non-empty list of sections that each have real lines."""
    sections = song_data.get("play_along_sections") if isinstance(song_data, dict) else None
    if not isinstance(sections, list) or not sections:
        return False
    for section in sections:
        if not isinstance(section, dict):
            return False
        lines = section.get("lines")
        if not isinstance(lines, list) or not any(str(line).strip() for line in lines):
            return False
    return True


def get_song_and_chords(
    mood: str, 
    instrument: str, 
    skill_level: str, 
    language_or_vibe: str = "", 
    custom_lyrics: str = "", 
    requested_song: str = ""
) -> dict:
    """
    Main entry point for the app. Picks a song, verifies it with APIs,
    gets lyrics (Romanized), arranges chords, and fetches verified voicings.
    Always returns a complete result (never empty play_along_sections).
    """
    
    # --- STEP 1: PICK THE SONG (Using Groq) ---
    # We ask the AI to pick a famous real song if the user hasn't specified one.
    if not requested_song:
        song_title, artist = pick_song(mood, skill_level, language_or_vibe)
    else:
        song_title = requested_song
        artist = ""

    # Pasted lyrics: no hashtags, links
    custom_lyrics = _clean_reference_text(custom_lyrics)

    # --- STEP 2: VERIFY AND FETCH GROUNDING ---
    
    # A + B) iTunes (5 results) + LRCLIB (artistName): verify the song, the playback singer, and get lyrics
    verified = _verify_song_and_singer(song_title, artist)
    song_title, artist = verified["song_title"], verified["artist"]
    lyrics_text = verified["lyrics"]
    lrclib_lyrics = lyrics_text   # kept for the spelling cross-check below

    # C) Fallback: yt-dlp and DDGS (only when there are no real lyrics at all)
    chords_reference = ""
    if not lyrics_text and not custom_lyrics:
        # Video description: accepted ONLY if it is clean text that really looks like lyrics
        ydl_opts = {'skip_download': True, 'quiet': True, 'noplaylist': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{song_title} {artist} lyrics", download=False)
                description = _clean_reference_text(
                    info['entries'][0].get('description') or "", is_description=True
                )
                lyrics_text = _lyrics_from_description(description)
        except Exception:
            pass
        
        # Search for chord progressions via DDGS
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{song_title} {artist} chords", max_results=2))
                chords_reference = _clean_reference_text(" ".join([r['body'] for r in results]))
        except Exception:
            pass

    # D) Romanize Lyrics
    # If the song is in Hindi, Malayalam, etc., we convert it to Roman script (Manglish/Hinglish).
    final_lyrics = custom_lyrics if custom_lyrics else lyrics_text
    if final_lyrics:
        final_lyrics = romanize_indic_text(final_lyrics)
    has_lyrics = bool(final_lyrics and final_lyrics.strip())

    # Pasted lyrics get cross-checked against LRCLIB's spelling
    spelling_reference = ""
    if custom_lyrics and lrclib_lyrics:
        spelling_reference = romanize_indic_text(lrclib_lyrics)[:3000]

    # --- STEP 3: ARRANGE SONG (Using Groq) ---
    if has_lyrics:
        lyrics_block = f"Lyrics to use:\n{final_lyrics}"
        if spelling_reference:
            lyrics_block += (
                "\n\nSpelling reference from LRCLIB (use ITS exact spelling to fix typos in the lyrics above; "
                "do NOT add any line that is not in the lyrics above):\n" + spelling_reference
            )
        lyrics_rules = f"{NATURAL_SPELLING_RULE}\n{NO_DIALOGUE_RULE}"
    else:
        lyrics_block = "Lyrics to use: NONE (no verified lyrics exist)."
        lyrics_rules = NO_LYRICS_RULE

    arrange_prompt = f"""
    Create a chord sheet for: {song_title} by {artist}.
    The playback singer has been verified as: {artist or "unknown"} - use exactly this for "artist".
    Instrument: {instrument}. Skill Level: {skill_level}.
    {lyrics_block}
    Context/Chords found: {chords_reference}
    {lyrics_rules}
    Return ONLY a JSON object with these exact keys:
    "song_title", "artist", "why_it_fits", "key", "tempo_bpm", "instrument", "skill_level", 
    "playing_technique", "capo_tip", "chord_voicings" (empty dict for now), 
    "play_along_sections" (list of {{"section_name": str, "lines": [str]}}).
    
    In "lines", place chords in brackets like [C] right before the word, e.g., "[C]Hello [G]world".
    If no lyrics exist, use bars: "[C] / / / | [G] / / / | [Am] / / / | [F] / / /".
    """
    song_data = _ask_for_json(arrange_prompt)

    # Never return an empty chart: retry ONCE with a simplified 4-beat bar prompt
    if not _sections_ok(song_data):
        simple_prompt = f"""
        Create a simple chord sheet for "{song_title}" by {artist or "unknown"} for {instrument} ({skill_level}).
        Return ONLY a JSON object with these exact keys:
        "song_title", "artist", "why_it_fits", "key", "tempo_bpm", "instrument", "skill_level",
        "playing_technique", "capo_tip", "chord_voicings" (empty dict), "play_along_sections".
        "play_along_sections" MUST be a NON-EMPTY list of {{"section_name": str, "lines": [str]}}.
        Use ONLY 4-beat chord bars: one line = 4 bars separated by "|", for example
        "[C] / / / | [G] / / / | [Am] / / / | [F] / / /". No lyrics and no other words.
        """
        song_data = _ask_for_json(simple_prompt)
        if song_data is None:
            song_data = {}
        if not _sections_ok(song_data):
            song_data["play_along_sections"] = []   # _ensure_bar_sections below builds a safe default

    # The verified title and singer always win over whatever the AI wrote
    song_data["song_title"] = song_title or song_data.get("song_title", "")
    song_data["artist"] = artist or song_data.get("artist", "")

    # Every field the app and format_chord_sheet need, so no combination ever breaks
    defaults = {
        "why_it_fits": f"A well-known song that fits a {mood.lower()} mood.",
        "key": "Unknown",
        "tempo_bpm": 90,
        "instrument": instrument,
        "skill_level": skill_level,
        "playing_technique": "Keep a steady pulse and change chords on beat 1 of each bar.",
        "capo_tip": "No capo needed",
    }
    for field, value in defaults.items():
        if song_data.get(field) in (None, ""):
            song_data[field] = value

    # No real lyrics (or nothing usable came back) -> only clean chord bars, never guessed words
    if not has_lyrics or not _sections_ok(song_data):
        _ensure_bar_sections(song_data)

    # --- STEP 4: DETERMINISTIC VOICING OVERRIDE ---
    # We identify every unique chord used and get the official fingerings from chords_db.
    unique_chords = []
    for section in song_data.get("play_along_sections", []):
        for line in section.get("lines", []):
            for found in re.findall(r"\[(.*?)\]", str(line)):
                name = _canon_chord(found)
                if name and name not in unique_chords:
                    unique_chords.append(name)
    
    song_data["chord_voicings"] = {}
    for chord in unique_chords:
        # get_verified_voicing ensures 100% accuracy for the specific instrument.
        # default= keeps the app from crashing on chords that are not in chords_db.py yet
        song_data["chord_voicings"][chord] = get_verified_voicing(
            chord, instrument, default="(shape not in database yet)"
        )

    return song_data


def format_chord_sheet(data: dict) -> str:
    """
    Renders the data into a human-readable text format.
    Converts "[C]Lyric" into:
    C
    Lyric
    """
    output = []
    output.append(f"SONG: {data['song_title']} - {data['artist']}")
    output.append(f"KEY: {data['key']} | TEMPO: {data['tempo_bpm']} BPM")
    output.append(f"INSTRUMENT: {data['instrument']} ({data['skill_level']})")
    if data.get("capo_tip"):
        output.append(f"HINT: {data['capo_tip']}")
    output.append("-" * 30)

    for section in data.get("play_along_sections", []):
        output.append(f"\n[{section['section_name']}]")
        for line in section["lines"]:
            # Splitting chords and text for 2-line display
            parts = re.split(r"(\[.*?\])", line)
            chord_line = ""
            lyric_line = ""
            cursor = 0
            
            for part in parts:
                if part.startswith("[") and part.endswith("]"):
                    chord = part[1:-1]
                    # Align chord above the current position in lyric line
                    padding = max(0, len(lyric_line) - len(chord_line))
                    chord_line += (" " * padding) + chord
                else:
                    lyric_line += part
            
            output.append(chord_line)
            output.append(lyric_line)
            
    output.append("\nCHORD FINGERINGS:")
    for chord, voicing in data.get("chord_voicings", {}).items():
        output.append(f"{chord}: {voicing}")
        
    return "\n".join(output)


if __name__ == "__main__":
    # Quick Test Case
    print("Testing mode_a.py...")
    try:
        result = get_song_and_chords(
            mood="happy", 
            instrument="Ukulele", 
            skill_level="Beginner", 
            language_or_vibe="Malayalam"
        )
        print(format_chord_sheet(result))
    except Exception as e:
        print(f"Error during test: {e}")