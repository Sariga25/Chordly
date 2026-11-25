"""
app.py - Chordly: Multimodal AI Music Coach (Streamlit web app)

Run it with:  streamlit run app.py

Mode A: pick a song for your mood/instrument/skill and get a play-along sheet.
Mode B: paste a reel/YouTube link (or upload a file) and get a play-along sheet
        for the cover you're watching, arranged for any instrument.
"""

import re
from html import escape
from pathlib import Path

import streamlit as st

import mode_a
import mode_b_analyzer
import mode_b_media

# This MUST be the first Streamlit command in the file.
st.set_page_config(page_title="Chordly - Multimodal AI Music Coach", page_icon="🎸", layout="wide")

INSTRUMENTS = ["Ukulele", "Violin", "Guitar", "Piano"]
SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"]
MOODS = ["Romantic and soulful", "Happy and chill", "Nostalgic and peaceful", "Energetic and upbeat"]
LANGUAGES = ["Hindi", "Malayalam", "Tamil", "Telugu", "Kannada", "Marathi", "English"]
CUSTOM_OPTION = "Custom..."
AUTO_DETECT = "Auto-Detect from Video"

UPLOAD_FOLDER = Path("temp_uploads")

# ---------------------------------------------------------------------------
# Styling (works in both light and dark mode)
# ---------------------------------------------------------------------------
CSS = """
<style>
.chordly-card {border: 1px solid rgba(128,128,128,0.35); background: rgba(128,128,128,0.08); border-radius: 16px; padding: 12px 10px 14px 10px; text-align: center; margin-bottom: 12px;}
.chordly-title {font-size: 1.5rem; font-weight: 800; margin-bottom: 4px;}
.chordly-tab {font-family: monospace; font-size: 1.05rem; font-weight: 700; margin-top: 2px; letter-spacing: 1px;}
.chordly-chip {display: inline-block; margin: 3px; padding: 5px 12px; border-radius: 999px; background: #ff6b4a; color: #ffffff; font-family: monospace; font-weight: 700; font-size: 1.1rem;}
.chordly-detail {font-size: 0.8rem; opacity: 0.75; margin-top: 8px; line-height: 1.3;}
.chordly-badge {display: inline-block; padding: 4px 12px; margin-right: 8px; border-radius: 999px; font-weight: 600; font-size: 0.9rem; border: 1px solid rgba(128,128,128,0.45); background: rgba(128,128,128,0.12);}
.chordly-badge-accent {background: #ff6b4a; color: #ffffff; border-color: #ff6b4a;}
</style>
"""

# Pitch class (0-11) of every note name, sharps and flats
NOTE_PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


# ---------------------------------------------------------------------------
# Chord cards
# ---------------------------------------------------------------------------
def parse_tab(fingering: str, num_strings: int):
    """Turn '0-0-0-3' or 'x-3-2-0-1-0' into a list like [0, 0, 0, 3] ('x' becomes None).

    Returns None if the text doesn't look like a tab for that many strings.
    """
    text = str(fingering).strip()

    # Normal format: numbers or x separated by dashes
    pattern = r"^\s*([xX\d]+(?:\s*-\s*[xX\d]+){" + str(num_strings - 1) + r"})(?![\d-])"
    match = re.match(pattern, text)
    if match:
        parts = [part.strip() for part in match.group(1).split("-")]
    else:
        # Compact format with no dashes, like 'x32010'
        compact = re.match(r"^\s*([xX\d]{" + str(num_strings) + r"})(?!\d)", text)
        if not compact:
            return None
        parts = list(compact.group(1))

    frets = [None if part.lower() == "x" else int(part) for part in parts]
    if any(fret is not None and fret > 24 for fret in frets):
        return None
    return frets


def build_fretboard_svg(frets: list, string_labels: list) -> str:
    """Draw a small fretboard diagram as an inline SVG string."""
    num_strings = len(frets)
    spacing = 22 if num_strings == 4 else 18
    fret_height = 24
    left, top = 34, 34

    pressed = [fret for fret in frets if fret is not None and fret > 0]
    base_fret = 1
    frets_shown = 4
    if pressed:
        if max(pressed) > 4:
            base_fret = min(pressed)  # slide the window up the neck
        frets_shown = max(4, max(pressed) - base_fret + 1)

    board_width = (num_strings - 1) * spacing
    board_height = frets_shown * fret_height
    width = left + board_width + 24
    height = top + board_height + 30

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto" role="img">'
    ]

    # Fret lines (thick nut at the top when we start at fret 1)
    for j in range(frets_shown + 1):
        y = top + j * fret_height
        thickness = 4 if (j == 0 and base_fret == 1) else 1.2
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{left + board_width}" y2="{y}" '
            f'stroke="currentColor" stroke-opacity="0.6" stroke-width="{thickness}"/>'
        )

    # Strings, open/muted markers, and bottom labels
    for i, fret in enumerate(frets):
        x = left + i * spacing
        svg.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + board_height}" '
            f'stroke="currentColor" stroke-opacity="0.8" stroke-width="1.2"/>'
        )
        if fret is None:
            svg.append(
                f'<text x="{x}" y="{top - 8}" text-anchor="middle" font-size="15" '
                f'fill="currentColor">×</text>'
            )
        elif fret == 0:
            svg.append(
                f'<circle cx="{x}" cy="{top - 12}" r="5" fill="none" '
                f'stroke="currentColor" stroke-width="1.5"/>'
            )
        svg.append(
            f'<text x="{x}" y="{top + board_height + 20}" text-anchor="middle" '
            f'font-size="11" fill="currentColor" fill-opacity="0.7">{escape(string_labels[i])}</text>'
        )

    # If the diagram doesn't start at fret 1, label the first fret shown
    if base_fret > 1:
        svg.append(
            f'<text x="{left - 8}" y="{top + fret_height / 2 + 4}" text-anchor="end" '
            f'font-size="11" fill="currentColor">{base_fret}fr</text>'
        )

    # Finger dots
    for i, fret in enumerate(frets):
        if fret is not None and fret > 0:
            x = left + i * spacing
            y = top + (fret - base_fret + 0.5) * fret_height
            svg.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#ff6b4a"/>')

    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------------------
# Piano keyboard diagram (1.5 octaves, root position, left to right)
# ---------------------------------------------------------------------------
ROOT_GOLD = "#ff9800"      # the root note of a piano chord (deep gold)
CHORD_AMBER = "#ffd54f"    # the 3rd / 5th (and 7th) notes (bright amber)
MELODY_GOLD = "#ffb74d"    # violin: root / primary melody note
HARMONY_BLUE = "#4fc3f7"   # violin: harmony note

WHITE_STEPS = [0, 2, 4, 5, 7, 9, 11]     # semitones above C of C D E F G A B
BLACK_AFTER_WHITE = {0, 1, 3, 4, 5}      # a black key sits right of C, D, F, G, A
BASE_WHITE_KEYS = 10                     # C4 through E5


def parse_piano_notes(fingering: str):
    """Turn 'C-Eb-G' into ['C', 'Eb', 'G']. Returns None if no valid notes are found."""
    names = re.findall(r"[A-G][#b]?", str(fingering))
    names = [name for name in names if name in NOTE_PITCH_CLASS]
    return names or None


def _white_key_semitone(index: int) -> int:
    return (index // 7) * 12 + WHITE_STEPS[index % 7]


def _root_position_semitones(notes: list) -> list:
    """Semitones above C4 for each chord note, stacked upwards from the root (never wrapping around)."""
    pitch_classes = [NOTE_PITCH_CLASS[note] for note in notes]
    semitones = [pitch_classes[0]]
    for pc in pitch_classes[1:]:
        step = (pc - semitones[-1]) % 12 or 12
        semitones.append(semitones[-1] + step)
    return semitones


def build_piano_svg(notes: list) -> str:
    """
    Draw a keyboard from C4 (10 white + 7 black keys, up to E5). The chord is shown in root position:
    root in deep gold, 3rd/5th in bright amber, with right-hand finger numbers 1 (thumb), 3 (middle),
    5 (pinky) printed on the keys. If a chord does not fit (e.g. B major, G7) the keyboard grows to the
    right just enough to keep it in root position.
    """
    semitones = _root_position_semitones(notes)
    fingers = [1, 3, 5] if len(notes) == 3 else [1, 2, 3, 5][:len(notes)]
    lit = {}  # semitone -> (spelled name, is_root, finger)
    for i, (semitone, note) in enumerate(zip(semitones, notes)):
        lit[semitone] = (note, i == 0, fingers[i] if i < len(fingers) else "")

    # How many white keys do we need? (a black key needs the white key to its right)
    white_keys = BASE_WHITE_KEYS
    for semitone in semitones:
        index = max(i for i in range(0, 40) if _white_key_semitone(i) <= semitone)
        needed = index + 1 if _white_key_semitone(index) == semitone else index + 2
        white_keys = max(white_keys, needed)

    white_w, white_h = 22, 104
    black_w, black_h = 14, 62
    left, top = 6, 6
    width = left * 2 + white_keys * white_w
    height = top + white_h + 8

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto" role="img">'
    ]

    # White keys: fill, then the finger number circle, then the label at the bottom
    for i in range(white_keys):
        x = left + i * white_w
        semitone = _white_key_semitone(i)
        info = lit.get(semitone)
        fill = "#ffffff" if not info else (ROOT_GOLD if info[1] else CHORD_AMBER)
        svg.append(
            f'<rect x="{x}" y="{top}" width="{white_w}" height="{white_h}" rx="3" '
            f'fill="{fill}" stroke="#555555" stroke-width="1.2"/>'
        )
        if info:
            cx = x + white_w / 2
            svg.append(f'<circle cx="{cx}" cy="{top + white_h - 32}" r="8" fill="#ffffff" stroke="#555555" stroke-width="1"/>')
            svg.append(
                f'<text x="{cx}" y="{top + white_h - 28}" text-anchor="middle" font-size="11" '
                f'font-weight="700" fill="#222222">{info[2]}</text>'
            )
            label, size, weight, color = info[0], 11, 700, "#222222"
        else:
            letter = "CDEFGAB"[i % 7]
            label = f"{letter}{4 + i // 7}" if letter == "C" else letter
            size, weight, color = 9, 400, "#999999"
        svg.append(
            f'<text x="{x + white_w / 2}" y="{top + white_h - 7}" text-anchor="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(label)}</text>'
        )

    # Black keys (drawn on top of the white keys)
    for i in range(white_keys - 1):
        if i % 7 not in BLACK_AFTER_WHITE:
            continue
        semitone = _white_key_semitone(i) + 1
        info = lit.get(semitone)
        x = left + (i + 1) * white_w - black_w / 2
        fill = "#222222" if not info else (ROOT_GOLD if info[1] else CHORD_AMBER)
        svg.append(
            f'<rect x="{x}" y="{top}" width="{black_w}" height="{black_h}" rx="2" '
            f'fill="{fill}" stroke="#555555" stroke-width="1"/>'
        )
        if info:
            cx = x + black_w / 2
            svg.append(
                f'<text x="{cx}" y="{top + 15}" text-anchor="middle" font-size="9" '
                f'font-weight="700" fill="#222222">{escape(info[0])}</text>'
            )
            svg.append(f'<circle cx="{cx}" cy="{top + 38}" r="6.5" fill="#ffffff" stroke="#555555" stroke-width="1"/>')
            svg.append(
                f'<text x="{cx}" y="{top + 42}" text-anchor="middle" font-size="10" '
                f'font-weight="700" fill="#222222">{info[2]}</text>'
            )

    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------------------
# Violin fingerboard diagram (fretless)
# ---------------------------------------------------------------------------
VIOLIN_PREFIX = re.compile(r"^\s*([GDAE](?:2↓|[0-4])(?:\s*-\s*[GDAE](?:2↓|[0-4]))*)")
VIOLIN_TOKEN = re.compile(r"([GDAE])(2↓|[0-4])")


def parse_violin_voicing(fingering: str):
    """
    Turn 'A2↓-E0 (Root C on A2↓ + Open E | Melody: A2↓)' into
    {'placements': [('A', '2↓'), ('E', '0')], 'melody_string': 'A'}. Returns None if unreadable.
    """
    text = str(fingering)
    match = VIOLIN_PREFIX.match(text)
    if not match:
        return None
    placements = VIOLIN_TOKEN.findall(match.group(1))
    melody = re.search(r"Melody:\s*([GDAE])(?:2↓|[0-4])", text)
    return {
        "placements": placements,
        "melody_string": melody.group(1) if melody else placements[0][0],
    }


def build_violin_svg(placements: list, melody_string: str) -> str:
    """
    Draw a fretless 4-string fingerboard (G D A E). No fret wires: just the nut and faint dashed
    finger-tape guide lines. Gold = root / melody note, blue = harmony note, "2↓" = low 2nd finger.
    """
    strings = ["G", "D", "A", "E"]
    thickness = {"G": 2.6, "D": 2.1, "A": 1.7, "E": 1.2}
    played = dict(placements)                      # string -> "0", "1", "2↓", "2", "3", "4"

    spacing, row_height = 30, 36
    left, top = 58, 46
    board_width = (len(strings) - 1) * spacing
    board_height = 4 * row_height
    width = left + board_width + 30
    height = top + board_height + 30
    number_x = left - 28                           # the left column with 1-4

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto" role="img">'
    ]

    # "Finger" heading and the 1-4 column; faint dashed finger-tape guide lines (no fret wires)
    svg.append(
        f'<text x="{number_x}" y="{top - 14}" text-anchor="middle" font-size="9" '
        f'font-weight="700" fill="currentColor" fill-opacity="0.6">Finger</text>'
    )
    for row in range(1, 5):
        y_mid = top + (row - 0.5) * row_height
        svg.append(
            f'<line x1="{left - 14}" y1="{y_mid}" x2="{left + board_width + 14}" y2="{y_mid}" '
            f'stroke="currentColor" stroke-opacity="0.35" stroke-width="1" stroke-dasharray="3 4"/>'
        )
        svg.append(
            f'<text x="{number_x}" y="{y_mid + 4}" text-anchor="middle" font-size="12" '
            f'fill="currentColor" fill-opacity="0.65">{row}</text>'
        )

    # Nut
    svg.append(
        f'<line x1="{left - 10}" y1="{top}" x2="{left + board_width + 10}" y2="{top}" '
        f'stroke="currentColor" stroke-opacity="0.85" stroke-width="5"/>'
    )

    # Strings (unused strings faint) and labels
    for i, string in enumerate(strings):
        x = left + i * spacing
        opacity = 0.9 if string in played else 0.28
        svg.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + board_height}" '
            f'stroke="currentColor" stroke-opacity="{opacity}" stroke-width="{thickness[string]}"/>'
        )
        svg.append(
            f'<text x="{x}" y="{top + board_height + 20}" text-anchor="middle" font-size="12" '
            f'font-weight="700" fill="currentColor" fill-opacity="0.75">{string}</text>'
        )

    # Notes: open string = ring above the nut; fingers 1-4 = dots (low 2nd finger sits a little higher)
    for i, string in enumerate(strings):
        if string not in played:
            continue
        x = left + i * spacing
        color = MELODY_GOLD if string == melody_string else HARMONY_BLUE
        finger = played[string]
        if finger == "0":
            svg.append(
                f'<circle cx="{x}" cy="{top - 14}" r="6.5" fill="{color}" fill-opacity="0.3" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            continue
        number = int(finger[0])
        y = top + (number - 0.5) * row_height - (10 if "↓" in finger else 0)
        svg.append(f'<circle cx="{x}" cy="{y}" r="12" fill="{color}" stroke="#555555" stroke-width="0.8"/>')
        svg.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="{12 if "↓" in finger else 14}" '
            f'font-weight="700" fill="#222222">{escape(finger).replace("↓", "&#8595;")}</text>'
        )

    svg.append("</svg>")
    return "".join(svg)


def render_chord_card(chord_name: str, fingering: str, instrument: str):
    """Show one chord as a styled card (fretboard, keyboard or fingerboard diagram, or chips)."""
    name = escape(str(chord_name))
    kind = (instrument or "").strip().lower()

    body = ""
    if kind == "ukulele":
        frets = parse_tab(fingering, 4)
        if frets:
            tab_text = "-".join("x" if f is None else str(f) for f in frets)
            body = build_fretboard_svg(frets, ["G", "C", "E", "A"]) + \
                f'<div class="chordly-tab">{escape(tab_text)}</div>'
    elif kind == "guitar":
        frets = parse_tab(fingering, 6)
        if frets:
            tab_text = "-".join("x" if f is None else str(f) for f in frets)
            body = build_fretboard_svg(frets, ["E", "A", "D", "G", "B", "e"]) + \
                f'<div class="chordly-tab">{escape(tab_text)}</div>'
    elif kind == "piano":
        notes = parse_piano_notes(fingering)
        if notes:
            body = (
                build_piano_svg(notes)
                + f'<div class="chordly-tab">{escape(" - ".join(notes))}</div>'
                + f'<div class="chordly-detail"><span style="color:{ROOT_GOLD}">●</span> Root &nbsp; '
                  f'<span style="color:{CHORD_AMBER}">●</span> 3rd / 5th &nbsp; · &nbsp; right hand fingers shown on keys</div>'
            )
    elif kind == "violin":
        voicing = parse_violin_voicing(fingering)
        if voicing:
            text = str(fingering).strip()
            main = text.split("(", 1)[0].strip()
            detail = text[len(main):].strip(" ()")
            body = build_violin_svg(voicing["placements"], voicing["melody_string"]) + \
                f'<div class="chordly-tab">{escape(main)}</div>'
            if detail:
                body += f'<div class="chordly-detail">{escape(detail)}</div>'
            body += (
                f'<div class="chordly-detail"><span style="color:{MELODY_GOLD}">●</span> Root / melody &nbsp; '
                f'<span style="color:{HARMONY_BLUE}">●</span> Harmony &nbsp; · &nbsp; 2↓ = low 2nd finger</div>'
            )

    if not body:
        # A shape we couldn't draw (or an unknown instrument): show the notation as chips.
        text = str(fingering).strip()
        main = re.split(r"[(;]", text, maxsplit=1)[0].strip()
        detail = text[len(main):].strip(" ;()")
        pieces = [piece for piece in re.split(r"[-\s,/]+", main) if piece]
        chips = "".join(f'<span class="chordly-chip">{escape(piece)}</span>' for piece in pieces)
        body = chips if chips else f'<div class="chordly-tab">{escape(text)}</div>'
        if detail:
            body += f'<div class="chordly-detail">{escape(detail)}</div>'

    st.markdown(
        f'<div class="chordly-card"><div class="chordly-title">{name}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def dedupe_chords(chord_voicings) -> list:
    """(chord, fingering) pairs with duplicate chords removed (case-insensitive), keeping the first order."""
    pairs = chord_voicings.items() if isinstance(chord_voicings, dict) else (chord_voicings or [])
    seen, unique = set(), []
    for chord, fingering in pairs:
        name = str(chord).strip()
        key = name.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((name, fingering))
    return unique


def render_chord_grid(chord_voicings, instrument: str, per_row: int = 4):
    """Lay the chord cards out in rows of columns - every chord appears only once."""
    items = dedupe_chords(chord_voicings)
    for start in range(0, len(items), per_row):
        columns = st.columns(per_row)
        for column, (chord, fingering) in zip(columns, items[start:start + per_row]):
            with column:
                render_chord_card(chord, fingering, instrument)


render_chord_cards = render_chord_grid  # old name, kept so nothing else breaks


# ---------------------------------------------------------------------------
# Shared result layout
# ---------------------------------------------------------------------------
def render_play_plan(result: dict, metrics: list, instrument: str, sheet_text: str):
    """Metrics, capo banner, technique, chord cards and the play-along sheet."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

    capo = str(result.get("capo_tip", "")).strip()
    if capo and "no capo needed" not in capo.lower():
        st.info(f"🎯 **Capo tip:** {capo}")

    st.subheader("🎯 How to play it")
    st.write(result.get("playing_technique", ""))

    st.subheader("🎼 Chord shapes")
    render_chord_grid(result.get("chord_voicings", {}), instrument)

    st.subheader("📜 Play-along sheet")
    st.code(sheet_text, language=None)


def choice_with_custom(label: str, options: list, key: str) -> str:
    """A dropdown that also lets the person type their own value."""
    choice = st.selectbox(label, options + [CUSTOM_OPTION], key=key)
    if choice == CUSTOM_OPTION:
        return st.text_input(f"Type your own {label.lower()}", key=f"{key}_custom").strip()
    return choice


def save_upload(uploaded_file) -> str:
    """Save an uploaded file into temp_uploads and return its path."""
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    path = UPLOAD_FOLDER / Path(uploaded_file.name).name
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.title("🎸 Chordly")
st.markdown(
    "**Your multimodal AI music coach.** Get a song picked for your mood and skill level, "
    "or drop in a reel / cover video and get a chord-over-lyrics play-along sheet for "
    "Ukulele, Violin, Guitar or Piano."
)
st.caption("AI chord charts can contain mistakes. Always check them by ear!")

tab_a, tab_b = st.tabs(["🎵 Mode A: Song & Play-Along Coach", "🎬 Mode B: Reel & Cover Analyzer"])

# ---------------------------------------------------------------------------
# Mode A
# ---------------------------------------------------------------------------
with tab_a:
    st.header("Find a song to play")

    left, right = st.columns(2)
    with left:
        a_instrument = st.selectbox("Instrument", INSTRUMENTS, key="a_instrument")
        a_skill = st.selectbox("Skill level", SKILL_LEVELS, key="a_skill")
    with right:
        a_mood = choice_with_custom("Mood", MOODS, key="a_mood")
        a_language = choice_with_custom("Language / vibe", LANGUAGES, key="a_language")

    a_requested = st.text_input(
        "Specific song (optional)", placeholder="e.g. Tum Hi Ho", key="a_requested"
    )
    with st.expander("Custom Lyrics (optional)"):
        a_lyrics = st.text_area(
            "Paste lyrics here and Chordly will place the chords on your words",
            height=150,
            key="a_lyrics",
        )

    if st.button("🎶 Find my song", type="primary", key="a_go"):
        if not a_mood:
            st.warning("Please type a mood first.")
        else:
            with st.spinner("Picking a song and looking up its chords..."):
                try:
                    st.session_state["a_result"] = mode_a.get_song_and_chords(
                        mood=a_mood,
                        instrument=a_instrument,
                        skill_level=a_skill,
                        language_or_vibe=a_language,
                        custom_lyrics=a_lyrics,
                        requested_song=a_requested,
                    )
                except Exception as error:
                    st.session_state.pop("a_result", None)
                    st.error(f"Something went wrong: {error}")

    result_a = st.session_state.get("a_result")
    if result_a:
        st.divider()
        st.subheader(f"🎵 {result_a['song_title']}")
        st.caption(f"by {result_a['artist']}")
        st.write(result_a["why_it_fits"])
        render_play_plan(
            result_a,
            metrics=[
                ("Key", result_a["key"]),
                ("Tempo", f"{result_a['tempo_bpm']} BPM"),
                ("Instrument", result_a["instrument"]),
                ("Level", result_a["skill_level"]),
            ],
            instrument=result_a["instrument"],
            sheet_text=mode_a.format_chord_sheet(result_a),
        )

# ---------------------------------------------------------------------------
# Mode B
# ---------------------------------------------------------------------------
with tab_b:
    st.header("Analyze a reel or cover")

    input_method = st.radio(
        "How do you want to add the video?",
        ["Paste a link", "Upload a file"],
        horizontal=True,
        key="b_method",
    )
    b_url = ""
    b_file = None
    if input_method == "Paste a link":
        b_url = st.text_input(
            "YouTube Short / Reel URL", placeholder="https://www.youtube.com/shorts/...", key="b_url"
        )
    else:
        b_file = st.file_uploader("Upload an .mp4 or .mp3 file", type=["mp4", "mp3"], key="b_file")

    b_choice = st.selectbox(
        "Arrange Play-Along For", [AUTO_DETECT] + INSTRUMENTS, key="b_target_choice"
    )
    b_target = None if b_choice == AUTO_DETECT else b_choice

    if st.button("🎬 Analyze", type="primary", key="b_go"):
        source = b_url.strip() if input_method == "Paste a link" else ""
        if input_method == "Upload a file" and b_file is not None:
            source = save_upload(b_file)

        if not source:
            st.warning("Please paste a link or upload a file first.")
        else:
            try:
                with st.spinner("Downloading and preparing the audio and frames..."):
                    st.session_state["b_media"] = mode_b_media.prepare_media(source)
                with st.spinner("Listening to the audio and working out the chords..."):
                    st.session_state["b_result"] = mode_b_analyzer.analyze_cover(
                        st.session_state["b_media"], b_target
                    )
                    st.session_state["b_target"] = b_choice
            except Exception as error:
                st.session_state.pop("b_media", None)
                st.session_state.pop("b_result", None)
                st.error(f"Something went wrong: {error}")

    elif st.session_state.get("b_media") and st.session_state.get("b_target") != b_choice:
        # The instrument dropdown changed: re-arrange WITHOUT downloading the video again.
        try:
            with st.spinner(f"Re-arranging for {b_choice}..."):
                st.session_state["b_result"] = mode_b_analyzer.analyze_cover(
                    st.session_state["b_media"], b_target
                )
                st.session_state["b_target"] = b_choice
        except Exception as error:
            st.error(f"Something went wrong: {error}")

    media = st.session_state.get("b_media")
    result_b = st.session_state.get("b_result")

    if media and result_b:
        st.divider()
        st.subheader(f"🎬 {media['title']}")

        frame_paths = media.get("frame_paths", [])
        if frame_paths:
            for column, frame in zip(st.columns(len(frame_paths)), frame_paths):
                column.image(frame)
        else:
            st.caption("Audio-only file - no video frames.")
        st.audio(media["audio_path"])

        st.divider()
        st.subheader(f"🎵 {result_b['identified_song']}")
        st.caption(f"by {result_b['original_artist']}")
        st.markdown(
            f'<span class="chordly-badge">👀 Detected: {escape(str(result_b["detected_instrument"]))}</span>'
            f'<span class="chordly-badge chordly-badge-accent">🎯 Arranged for: '
            f'{escape(str(result_b["arranged_for_instrument"]))}</span>',
            unsafe_allow_html=True,
        )
        st.write("")

        render_play_plan(
            result_b,
            metrics=[
                ("Key", result_b["estimated_key"]),
                ("Tempo", f"{result_b['tempo_bpm']} BPM"),
            ],
            instrument=result_b["arranged_for_instrument"],
            sheet_text=mode_b_analyzer.format_chord_sheet(result_b),
        )