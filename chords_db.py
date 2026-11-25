"""
chords_db.py - Chordly's deterministic music rules (no AI guessing)

PART 1  Verified chord fingerings for Ukulele, Guitar, Violin and Piano.
        get_verified_voicing("G#", "Ukulele") always gives the same answer.
        verify_database() re-checks every fingering against music theory.

PART 2  romanize_indic_text() turns Malayalam, Devanagari (Hindi/Marathi),
        Tamil, Telugu, Kannada, Bengali and Gurmukhi (Punjabi) text into
        readable Roman letters (Manglish / Hinglish / Tanglish ...).
"""

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# =============================================================================
# PART 1: VERIFIED CHORD FINGERINGS
# =============================================================================
#
# How to read each instrument:
#   Ukulele : 4 fret numbers for strings G-C-E-A      ("0-0-0-3" = open, open, open, 3rd fret)
#   Guitar  : 6 fret numbers for E-A-D-G-B-e          ("x" = do not play that string)
#   Violin  : a double-stop on two neighbouring strings (only G, D, A, E exist), written as
#             "A2↓-E0 (Root C on A2↓ + Open E | Melody: A2↓)":
#               - the part before the brackets is StringFinger-StringFinger (0 = open string,
#                 1-4 = index, middle, ring, pinky in first position);
#               - the violin has NO frets, so a finger can sit a half step low or high.
#                 "2↓" = LOW 2nd finger (placed close to the 1st finger): C on the A string,
#                 G on the E string, F on the D string, Bb on the G string.
#                 A plain "2" is the normal HIGH 2nd finger (C# on A, G# on E, F# on D, B on G);
#               - "Melody:" names the primary note (the chord's root if it is in the double stop,
#                 otherwise its 3rd) - it is drawn gold, the other note blue.
#   Piano   : the notes of the chord, lowest to highest, joined by "-" ("C-E-G").
#             Root, major/minor 3rd, perfect 5th (plus the minor 7th for 7th chords).
#             Keyboard-style spelling is used: G# major is "G# - C - D#" (C, not B#),
#             C# major is "C# - F - G#" (F, not E#).

CHORD_VOICINGS_DB = {
    "C": {
        "Ukulele": "0-0-0-3",
        "Guitar": "x-3-2-0-1-0",
        "Violin": "A2↓-E0 (Root C on A2↓ + Open E | Melody: A2↓)",
        "Piano": "C-E-G",
    },
    "Cm": {
        "Ukulele": "0-3-3-3",
        "Guitar": "x-3-5-5-4-3",
        "Violin": "D1-A2↓ (Eb on D1 + Root C on A2↓ | Melody: A2↓)",
        "Piano": "C-Eb-G",
    },
    "C#": {
        "Ukulele": "1-1-1-4",
        "Guitar": "x-4-6-6-6-4",
        "Violin": "G1-D2↓ (G# on G1 + F on D2↓ | Melody: D2↓)",
        "Piano": "C#-F-G#",
    },
    "C#m": {
        "Ukulele": "1-4-4-4",
        "Guitar": "x-4-6-6-5-4",
        "Violin": "A2-E0 (Root C# on A2 + Open E | Melody: A2)",
        "Piano": "C#-E-G#",
    },
    "Db": {
        "Ukulele": "1-1-1-4",
        "Guitar": "x-4-6-6-6-4",
        "Violin": "G1-D2↓ (Ab on G1 + F on D2↓ | Melody: D2↓)",
        "Piano": "Db-F-Ab",
    },
    "D": {
        "Ukulele": "2-2-2-0",
        "Guitar": "x-x-0-2-3-2",
        "Violin": "A3-E1 (Root D on A3 + F# on E1 | Melody: A3)",
        "Piano": "D-F#-A",
    },
    "Dm": {
        "Ukulele": "2-2-1-0",
        "Guitar": "x-x-0-2-3-1",
        "Violin": "A3-E1 (Root D on A3 + F on E1 | Melody: A3)",
        "Piano": "D-F-A",
    },
    "Eb": {
        "Ukulele": "0-3-3-1",
        "Guitar": "x-6-8-8-8-6",
        "Violin": "G0-D1 (Open G + Root Eb on D1 | Melody: D1)",
        "Piano": "Eb-G-Bb",
    },
    "Ebm": {
        "Ukulele": "3-3-2-1",
        "Guitar": "x-6-8-8-7-6",
        "Violin": "A3-E1 (Root Eb on A3 + Gb on E1 | Melody: A3)",
        "Piano": "Eb-Gb-Bb",
    },
    "E": {
        "Ukulele": "4-4-4-2",
        "Guitar": "0-2-2-1-0-0",
        "Violin": "A4-E2 (Root E on A4 + G# on E2 | Melody: A4)",
        "Piano": "E-G#-B",
    },
    "Em": {
        "Ukulele": "0-4-3-2",
        "Guitar": "0-2-2-0-0-0",
        "Violin": "G0-D1 (Open G + Root E on D1 | Melody: D1)",
        "Piano": "E-G-B",
    },
    "F": {
        "Ukulele": "2-0-1-0",
        "Guitar": "1-3-3-2-1-1",
        "Violin": "A0-E1 (Open A + Root F on E1 | Melody: E1)",
        "Piano": "F-A-C",
    },
    "Fm": {
        "Ukulele": "1-0-1-3",
        "Guitar": "1-3-3-1-1-1",
        "Violin": "G1-D2↓ (Ab on G1 + Root F on D2↓ | Melody: D2↓)",
        "Piano": "F-Ab-C",
    },
    "F#": {
        "Ukulele": "3-1-2-1",
        "Guitar": "2-4-4-3-2-2",
        "Violin": "A2-E3 (C# on A2 + A# on E3 | Melody: E3)",
        "Piano": "F#-A#-C#",
    },
    "F#m": {
        "Ukulele": "2-1-2-0",
        "Guitar": "2-4-4-2-2-2",
        "Violin": "A0-E1 (Open A + Root F# on E1 | Melody: E1)",
        "Piano": "F#-A-C#",
    },
    "G": {
        "Ukulele": "0-2-3-2",
        "Guitar": "3-2-0-0-0-3",
        "Violin": "G0-D0 (Open G + Open D | Melody: G0)",
        "Piano": "G-B-D",
    },
    "Gm": {
        "Ukulele": "0-2-3-1",
        "Guitar": "3-5-5-3-3-3",
        "Violin": "A1-E2↓ (Bb on A1 + Root G on E2↓ | Melody: E2↓)",
        "Piano": "G-Bb-D",
    },
    "G#": {
        "Ukulele": "5-3-4-3",
        "Guitar": "4-6-6-5-4-4",
        "Violin": "D1-A2↓ (D# on D1 + C on A2↓ | Melody: A2↓)",
        "Piano": "G#-C-D#",
    },
    "G#m": {
        "Ukulele": "1-3-4-2",
        "Guitar": "4-6-6-4-4-4",
        "Violin": "A1-E2 (B on A1 + Root G# on E2 | Melody: E2)",
        "Piano": "G#-B-D#",
    },
    "Ab": {
        "Ukulele": "5-3-4-3",
        "Guitar": "4-6-6-5-4-4",
        "Violin": "D1-A2↓ (Eb on D1 + C on A2↓ | Melody: A2↓)",
        "Piano": "Ab-C-Eb",
    },
    "A": {
        "Ukulele": "2-1-0-0",
        "Guitar": "x-0-2-2-2-0",
        "Violin": "A2-E3 (C# on A2 + Root A on E3 | Melody: E3)",
        "Piano": "A-C#-E",
    },
    "Am": {
        "Ukulele": "2-0-0-0",
        "Guitar": "x-0-2-2-1-0",
        "Violin": "A2↓-E3 (C on A2↓ + Root A on E3 | Melody: E3)",
        "Piano": "A-C-E",
    },
    "Bb": {
        "Ukulele": "3-2-1-1",
        "Guitar": "x-1-3-3-3-1",
        "Violin": "D0-A1 (Open D + Root Bb on A1 | Melody: A1)",
        "Piano": "Bb-D-F",
    },
    "Bbm": {
        "Ukulele": "3-1-1-1",
        "Guitar": "x-1-3-3-2-1",
        "Violin": "A2-E3 (Db on A2 + Root Bb on E3 | Melody: E3)",
        "Piano": "Bb-Db-F",
    },
    "B": {
        "Ukulele": "4-3-2-2",
        "Guitar": "x-2-4-4-4-2",
        "Violin": "A3-E4 (D# on A3 + Root B on E4 | Melody: E4)",
        "Piano": "B-D#-F#",
    },
    "Bm": {
        "Ukulele": "4-2-2-2",
        "Guitar": "x-2-4-4-3-2",
        "Violin": "D0-A1 (Open D + Root B on A1 | Melody: A1)",
        "Piano": "B-D-F#",
    },
    "G7": {
        "Ukulele": "0-2-1-2",
        "Guitar": "3-2-0-0-0-1",
        "Violin": "D2↓-A1 (F on D2↓ + B on A1 | Melody: A1)",
        "Piano": "G-B-D-F",
    },
    "C7": {
        "Ukulele": "0-0-0-1",
        "Guitar": "x-3-2-3-1-0",
        "Violin": "A1-E0 (Bb on A1 + Open E | Melody: E0)",
        "Piano": "C-E-G-Bb",
    },
    "D7": {
        "Ukulele": "2-2-2-3",
        "Guitar": "x-x-0-2-1-2",
        "Violin": "A2↓-E1 (C on A2↓ + F# on E1 | Melody: E1)",
        "Piano": "D-F#-A-C",
    },
    "E7": {
        "Ukulele": "1-2-0-2",
        "Guitar": "0-2-0-1-0-0",
        "Violin": "G1-D0 (G# on G1 + Open D | Melody: G1)",
        "Piano": "E-G#-B-D",
    },
    "A7": {
        "Ukulele": "0-1-0-0",
        "Guitar": "x-0-2-0-2-0",
        "Violin": "D3-A2 (G on D3 + C# on A2 | Melody: A2)",
        "Piano": "A-C#-E-G",
    },
    "Am7": {
        "Ukulele": "0-0-0-0",
        "Guitar": "x-0-2-0-1-0",
        "Violin": "A2↓-E2↓ (C on A2↓ + G on E2↓ | Melody: A2↓)",
        "Piano": "A-C-E-G",
    },
    "Em7": {
        "Ukulele": "0-2-0-2",
        "Guitar": "0-2-0-0-0-0",
        "Violin": "G0-D0 (Open G + Open D | Melody: G0)",
        "Piano": "E-G-B-D",
    },
    "Dm7": {
        "Ukulele": "2-2-1-3",
        "Guitar": "x-x-0-2-1-1",
        "Violin": "A2↓-E1 (C on A2↓ + F on E1 | Melody: E1)",
        "Piano": "D-F-A-C",
    },
}

INSTRUMENTS = ["Ukulele", "Guitar", "Violin", "Piano"]

# Two names for the same note (same key on a piano)
ENHARMONIC = {
    "C#": "Db", "Db": "C#",
    "D#": "Eb", "Eb": "D#",
    "F#": "Gb", "Gb": "F#",
    "G#": "Ab", "Ab": "G#",
    "A#": "Bb", "Bb": "A#",
}

# Ways people write the same chord quality
SUFFIX_ALIASES = {
    "min": "m", "minor": "m", "-": "m",
    "maj": "", "major": "",
    "dom7": "7",
    "min7": "m7", "-7": "m7",
}

INSTRUMENT_ALIASES = {"ukelele": "Ukulele", "uke": "Ukulele"}


def _candidate_names(chord_name: str) -> list:
    """Return the chord names to try, in order (e.g. 'Dbm' -> ['Dbm', 'C#m'])."""
    text = chord_name.strip().replace("♯", "#").replace("♭", "b")
    text = text.split("/")[0]  # 'D/F#' -> 'D' (we ignore the bass note)

    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", text)
    if not match:
        return []

    root = match.group(1).upper() + match.group(2)
    suffix = SUFFIX_ALIASES.get(match.group(3).strip(), match.group(3).strip())

    names = [root + suffix]
    if root in ENHARMONIC:
        names.append(ENHARMONIC[root] + suffix)
    return names


def get_verified_voicing(chord_name: str, instrument: str, default=None) -> str:
    """Return the verified fingering for a chord on an instrument.

    Enharmonic names are treated as equal (G# = Ab, C# = Db, F# = Gb, D# = Eb, A# = Bb).
    If the chord is not in the database: raises KeyError, or returns `default` if you gave one.
    """
    wanted_instrument = INSTRUMENT_ALIASES.get(instrument.strip().lower(), instrument.strip().title())

    if wanted_instrument in INSTRUMENTS:
        for name in _candidate_names(chord_name):
            if name in CHORD_VOICINGS_DB:
                return CHORD_VOICINGS_DB[name][wanted_instrument]

    if default is not None:
        return default
    if wanted_instrument not in INSTRUMENTS:
        raise KeyError(f"Unknown instrument '{instrument}'. Choose from {INSTRUMENTS}.")
    raise KeyError(f"Chord '{chord_name}' is not in the database yet.")


# -----------------------------------------------------------------------------
# Self-check: proves every fingering matches the notes of its chord
# -----------------------------------------------------------------------------
NOTE_TO_PC = {  # pitch class = note number 0-11 (C = 0, C# = 1, ... B = 11)
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
    "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
CHORD_FORMULAS = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10]}

UKULELE_OPEN = [7, 0, 4, 9]            # G C E A
GUITAR_OPEN = [4, 9, 2, 7, 11, 4]      # E A D G B e
VIOLIN_OPEN = {"G": 7, "D": 2, "A": 9, "E": 4}
VIOLIN_STRING_ORDER = ["G", "D", "A", "E"]

# finger -> semitones above the open string it can play (first position). "2↓" = LOW 2nd finger.
VIOLIN_FINGER_OFFSETS = {"0": (0,), "1": (1, 2), "2↓": (3,), "2": (4,), "3": (5, 6), "4": (7,)}
_VIOLIN_TOKEN = r"[GDAE](?:2↓|[0-4])"
VIOLIN_MAIN = re.compile(rf"^({_VIOLIN_TOKEN})-({_VIOLIN_TOKEN}) \((.+) \| Melody: ({_VIOLIN_TOKEN})\)$")
VIOLIN_NOTE_PART = re.compile(rf"^(?:Open ([A-G][#b]?)|(Root )?([A-G][#b]?) on ({_VIOLIN_TOKEN}))$")


def _chord_pitch_classes(chord: str):
    """Split 'Am7' into (root pitch class, [pitch classes of the chord notes])."""
    match = re.match(r"^([A-G][#b]?)(.*)$", chord)
    root = NOTE_TO_PC[match.group(1)]
    return root, [(root + step) % 12 for step in CHORD_FORMULAS[match.group(2)]]


def _fretted_problems(chord: str, tab: str, open_strings: list, label: str) -> list:
    problems = []
    parts = tab.split("-")
    if len(parts) != len(open_strings):
        return [f"{chord} {label}: expected {len(open_strings)} strings, got '{tab}'"]

    _root, tones = _chord_pitch_classes(chord)
    sounding = set()
    played = 0
    for open_pc, part in zip(open_strings, parts):
        if part.lower() == "x":
            continue
        played += 1
        sounding.add((open_pc + int(part)) % 12)

    if not sounding <= set(tones):
        problems.append(f"{chord} {label}: '{tab}' plays notes outside the chord")
    needed = {tones[0], tones[1]} | ({tones[2]} if len(tones) == 3 else {tones[3]})
    if not needed <= sounding:
        problems.append(f"{chord} {label}: '{tab}' is missing root/3rd/5th (or 7th)")
    if played < len(open_strings) - 2:
        problems.append(f"{chord} {label}: '{tab}' plays too few strings")
    return problems


def _violin_problems(chord: str, text: str) -> list:
    match = VIOLIN_MAIN.match(text)
    if not match:
        return [f"{chord} Violin: cannot read '{text}' "
                f"(use 'A2↓-E0 (Root C on A2↓ + Open E | Melody: A2↓)')"]

    token1, token2, description, melody = match.groups()
    string1, string2 = token1[0], token2[0]
    if abs(VIOLIN_STRING_ORDER.index(string1) - VIOLIN_STRING_ORDER.index(string2)) != 1:
        return [f"{chord} Violin: strings {string1} and {string2} are not neighbours"]

    parts = description.split(" + ")
    if len(parts) != 2:
        return [f"{chord} Violin: the description in '{text}' must name two notes joined by ' + '"]

    root, tones = _chord_pitch_classes(chord)
    problems = []
    pitch_classes = []
    placed = []  # (finger number, semitone offset) of every pressed finger

    for token, part in zip((token1, token2), parts):
        part_match = VIOLIN_NOTE_PART.match(part)
        if not part_match:
            problems.append(f"{chord} Violin: cannot read '{part}'")
            continue
        open_note, root_label, note, part_token = part_match.groups()
        string, finger_key = token[0], token[1:]
        if open_note:
            note, part_token = open_note, token          # "Open G" belongs to this string
            if finger_key != "0":
                problems.append(f"{chord} Violin: '{part}' says Open but the finger is {finger_key}")
        elif part_token != token:
            problems.append(f"{chord} Violin: '{part}' does not match {token} in '{text}'")

        pc = NOTE_TO_PC[note]
        pitch_classes.append(pc)
        offset = (pc - VIOLIN_OPEN[string]) % 12
        if offset not in VIOLIN_FINGER_OFFSETS[finger_key]:
            problems.append(f"{chord} Violin: {note} cannot be played with {token} (the finger would be in the wrong spot)")
        elif finger_key != "0":
            placed.append((int(finger_key[0]), offset))
        if pc not in tones:
            problems.append(f"{chord} Violin: {note} is not a note of {chord}")
        if bool(root_label) != (pc == root and finger_key != "0"):
            problems.append(f"{chord} Violin: 'Root' label is wrong for {note} in '{text}'")

    if len(set(pitch_classes)) != 2:
        problems.append(f"{chord} Violin: both notes are the same")
    if root not in pitch_classes and tones[1] not in pitch_classes:
        problems.append(f"{chord} Violin: '{text}' must include the root or the 3rd of {chord}")
    if len(tones) == 4 and tones[3] not in pitch_classes:
        problems.append(f"{chord} Violin: '{text}' must include the 7th of {chord}")

    # The Melody note: the root if it is there, otherwise the 3rd
    if melody not in (token1, token2):
        problems.append(f"{chord} Violin: Melody {melody} is not one of the two notes in '{text}'")
    elif len(pitch_classes) == 2:
        melody_pc = pitch_classes[(token1, token2).index(melody)]
        expected = root if root in pitch_classes else tones[1]
        if tones[1] in pitch_classes or root in pitch_classes:
            if melody_pc != expected:
                problems.append(f"{chord} Violin: the Melody note should be the {'root' if root in pitch_classes else '3rd'}")

    # Can one hand reach both fingers? (neighbouring fingers span 1-2 half steps)
    for finger_a, offset_a in placed:
        for finger_b, offset_b in placed:
            if finger_a < finger_b:
                gap = offset_b - offset_a
                if not (finger_b - finger_a) <= gap <= 2 * (finger_b - finger_a):
                    problems.append(f"{chord} Violin: fingers {finger_a} and {finger_b} cannot reach in '{text}'")
            elif finger_a == finger_b and offset_a != offset_b:
                problems.append(f"{chord} Violin: one finger used twice at different spots in '{text}'")
    return problems


def _piano_problems(chord: str, text: str) -> list:
    root, tones = _chord_pitch_classes(chord)
    notes = text.split("-")
    try:
        pitch_classes = [NOTE_TO_PC[note] for note in notes]
    except KeyError:
        return [f"{chord} Piano: unknown note in '{text}'"]
    if pitch_classes != tones:
        return [f"{chord} Piano: '{text}' does not match the notes of {chord}"]
    return []


def verify_database() -> list:
    """Check every entry. Returns a list of problems (an empty list means all OK)."""
    problems = []
    for chord, voicings in CHORD_VOICINGS_DB.items():
        if sorted(voicings) != sorted(INSTRUMENTS):
            problems.append(f"{chord}: must have exactly {INSTRUMENTS}")
            continue
        problems += _fretted_problems(chord, voicings["Ukulele"], UKULELE_OPEN, "Ukulele")
        problems += _fretted_problems(chord, voicings["Guitar"], GUITAR_OPEN, "Guitar")
        problems += _violin_problems(chord, voicings["Violin"])
        problems += _piano_problems(chord, voicings["Piano"])
    return problems


# =============================================================================
# PART 2: ROMANIZING INDIAN-LANGUAGE TEXT
# =============================================================================
# script name -> (first Unicode number, last Unicode number, indic_transliteration scheme)
SCRIPTS = {
    "devanagari": (0x0900, 0x097F, sanscript.DEVANAGARI),   # Hindi, Marathi
    "bengali":    (0x0980, 0x09FF, sanscript.BENGALI),
    "gurmukhi":   (0x0A00, 0x0A7F, sanscript.GURMUKHI),     # Punjabi
    "tamil":      (0x0B80, 0x0BFF, sanscript.TAMIL),
    "telugu":     (0x0C00, 0x0C7F, sanscript.TELUGU),
    "kannada":    (0x0C80, 0x0CFF, sanscript.KANNADA),
    "malayalam":  (0x0D00, 0x0D7F, sanscript.MALAYALAM),
}
NORTH_SCRIPTS = ("devanagari", "bengali", "gurmukhi")   # they drop the "a" at the end of words
SOUTH_SCRIPTS = ("malayalam", "tamil", "telugu", "kannada")
DANDAS = "\u0964\u0965"                                  # the Hindi full stops । ॥

# All Indian scripts share the same layout, so one set of offsets works for every script
CONSONANT_RANGES = [(0x15, 0x39), (0x58, 0x5F)]
MATRA_OFFSETS = set(range(0x3E, 0x4D)) | {0x57, 0x62, 0x63}   # vowel signs
VIRAMA = 0x4D                                                  # the "no vowel" mark
NUKTA = 0x3C
NASAL_MARKS = {0x01, 0x02, 0x03, 0x70, 0x71}                   # chandrabindu, anusvara, visarga, tippi, addak

# Malayalam "chillu" letters are not handled by the library: turn them into consonant + virama
MALAYALAM_CHILLU = {
    "\u0D7A": "\u0D23\u0D4D", "\u0D7B": "\u0D28\u0D4D", "\u0D7C": "\u0D30\u0D4D",
    "\u0D7D": "\u0D32\u0D4D", "\u0D7E": "\u0D33\u0D4D", "\u0D7F": "\u0D15\u0D4D",
}

# Explicit Devanagari nukta characters (both precomposed and decomposed forms)
# mapped through private-use placeholders so indic_transliteration preserves
# following vowel matras accurately.
DEVANAGARI_NUKTA_OVERRIDES = [
    ("ज़", "\uE000"), ("\u091c\u093c", "\uE000"),
    ("फ़", "\uE001"), ("\u092b\u093c", "\uE001"),
    ("क़", "\uE002"), ("\u0915\u093c", "\uE002"),
    ("ग़", "\uE003"), ("\u0917\u093c", "\uE003"),
    ("ख़", "\uE004"), ("\u0916\u093c", "\uE004"),
]
NUKTA_PLACEHOLDER_TO_ROMAN = {
    "\uE000": "z",
    "\uE001": "f",
    "\uE002": "q",
    "\uE003": "gh",
    "\uE004": "kh",
}

# South Indian short 'i' vowel signs and independent short 'i' vowels
SOUTH_SHORT_I_MATRAS = {
    "malayalam": "\u0d3f",
    "tamil": "\u0bbf",
    "telugu": "\u0c3f",
    "kannada": "\u0cbf",
}
SOUTH_SHORT_I_VOWELS = {
    "malayalam": "\u0d07",
    "tamil": "\u0b87",
    "telugu": "\u0c07",
    "kannada": "\u0c87",
}
SHORT_I_PLACEHOLDER = "\uE005"

VOWEL_LETTERS = "aāiīuūeēoōèò\uE005"


def _detect_script(character: str):
    """Return the script name of a character, or None if it is not Indian-script text."""
    if character in DANDAS:
        return None
    number = ord(character)
    for name, (first, last, _scheme) in SCRIPTS.items():
        if first <= number <= last:
            return name
    return None


# ---- dropping the silent "a" (Hindi "tum", not "tuma") -------------------------
def _is_consonant(offset: int) -> bool:
    return any(first <= offset <= last for first, last in CONSONANT_RANGES)


def _split_aksharas(word: str, base: int) -> list:
    """Cut a word into syllable blocks and note which ones end with a silent-able "a"."""
    aksharas = []
    i, size = 0, len(word)
    while i < size:
        start = i
        if _is_consonant(ord(word[i]) - base):
            conjunct = False
            while True:
                i += 1                                            # the consonant
                if i < size and ord(word[i]) - base == NUKTA:
                    i += 1
                if (i + 1 < size and ord(word[i]) - base == VIRAMA
                        and _is_consonant(ord(word[i + 1]) - base)):
                    conjunct = True                               # joined consonants like "ब्ब"
                    i += 1
                    continue
                break
            has_vowel_mark = False
            if i < size and ord(word[i]) - base in MATRA_OFFSETS:
                i += 1
                has_vowel_mark = True
            elif i < size and ord(word[i]) - base == VIRAMA:
                i += 1
                has_vowel_mark = True
            nasal = False
            while i < size and ord(word[i]) - base in NASAL_MARKS:
                i += 1
                nasal = True
            aksharas.append({"text": word[start:i], "schwa": not has_vowel_mark and not nasal,
                             "conjunct": conjunct, "nasal": nasal})
        else:  # a vowel letter, digit or anything else
            i += 1
            nasal = False
            while i < size and ord(word[i]) - base in NASAL_MARKS:
                i += 1
                nasal = True
            aksharas.append({"text": word[start:i], "schwa": False, "conjunct": False, "nasal": nasal})
    return aksharas


def _delete_schwas(word: str, base: int) -> str:
    """Add a virama (no-vowel mark) where the inherent "a" is not pronounced."""
    aksharas = _split_aksharas(word, base)
    count = len(aksharas)
    delete = [False] * count

    if count > 1 and aksharas[-1]["schwa"]:       # "tuma" -> "tum"
        delete[-1] = True

    for i in range(count - 2, 0, -1):             # middle of the word, right to left
        current, following, previous = aksharas[i], aksharas[i + 1], aksharas[i - 1]
        following_has_vowel = not following["schwa"] or not delete[i + 1]
        if (current["schwa"] and not current["conjunct"] and following_has_vowel
                and not following["conjunct"] and not previous["nasal"]):
            delete[i] = True                      # "kamala" -> "kamla"

    virama = chr(base + VIRAMA)
    return "".join(a["text"] + (virama if cut else "") for a, cut in zip(aksharas, delete))


def _apply_nukta_overrides(run: str) -> str:
    """Replace Devanagari nukta consonants with safe placeholders before transliteration."""
    for src, placeholder in DEVANAGARI_NUKTA_OVERRIDES:
        run = run.replace(src, placeholder)
    # Remove any bare nukta remaining after replacement so it doesn't corrupt vowels
    return run.replace("\u093c", "")


def _restore_nukta_placeholders(text: str) -> str:
    """Restore nukta placeholders to z, f, q, gh, kh with inherent 'a' where appropriate."""
    vowels = set("aeiouāīūēō\uE005")
    out = []
    for i, ch in enumerate(text):
        if ch in NUKTA_PLACEHOLDER_TO_ROMAN:
            out.append(NUKTA_PLACEHOLDER_TO_ROMAN[ch])
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt not in vowels:
                out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def _protect_south_short_i(run: str, script: str) -> str:
    """Ensure South Indian short 'i' matras and independent vowels always map to 'i'."""
    matra = SOUTH_SHORT_I_MATRAS[script]
    vowel = SOUTH_SHORT_I_VOWELS[script]
    virama = chr(SCRIPTS[script][0] + VIRAMA)
    run = run.replace(matra, virama + SHORT_I_PLACEHOLDER)
    run = run.replace(vowel, SHORT_I_PLACEHOLDER)
    return run


# ---- Tamil: the library's Tamil table is built for Sanskrit, so we repair it -------
TAMIL_UNDO_ASPIRATES = [("gh", "k"), ("jh", "c"), ("ḍh", "ṭ"), ("dh", "t"), ("bh", "p")]
TAMIL_INITIAL = {"k": "k", "c": "s", "ṭ": "t", "t": "th", "p": "p"}
TAMIL_VOICED = {"k": "g", "c": "s", "ṭ": "d", "t": "dh", "p": "b"}
TAMIL_DOUBLED = {"k": "kk", "c": "ch", "ṭ": "tt", "t": "th", "p": "pp"}
TAMIL_VOICING_AFTER = VOWEL_LETTERS + "ṅñṇnṉmyrZ"


def _tamil_fix(iso_text: str) -> str:
    """Tamil spells k/g, t/d... the same way; apply the usual Tanglish pronunciation rules."""
    text = unicodedata.normalize("NFC", iso_text).lower()
    text = text.replace("l\u0324", "Z").replace("ḻ", "Z").replace("ṁ", "m").replace("ṃ", "m")
    for aspirated, plain in TAMIL_UNDO_ASPIRATES:
        text = text.replace(aspirated, plain)
    text = text.replace("ṉṟ", "ndr")

    out = []
    i = 0
    while i < len(text):
        letter = text[i]
        if letter == "ṟ" and text[i:i + 2] == "ṟṟ":
            out.append("tr")
            i += 2
            continue
        if letter in TAMIL_INITIAL:
            if text[i + 1:i + 2] == letter:                       # doubled letter = long sound
                out.append(TAMIL_DOUBLED[letter])
                i += 2
                continue
            before = text[i - 1] if i > 0 else ""
            if before and before in TAMIL_VOICING_AFTER:
                if letter == "c" and before in "ṅñṇnṉm":
                    out.append("j")                               # ஞ்ச = "nj"
                else:
                    out.append(TAMIL_VOICED[letter])
            else:
                out.append(TAMIL_INITIAL[letter])
        else:
            out.append(letter)
        i += 1
    return "".join(out)


# ---- turning the library's ISO output (with accents) into plain readable letters ----
def _make_readable(iso_text: str, script: str) -> str:
    text = unicodedata.normalize("NFC", iso_text).lower()
    north = script in NORTH_SCRIPTS

    text = text.replace("l\u0324", "zh").replace("Z", "zh")        # ழ = "zh" (Tamil/Malayalam)
    text = text.replace("ḻ", "zh" if script in ("tamil", "malayalam") else "l")
    text = text.replace("ḷ", "l")
    text = text.replace("jñ", "gy" if north else "jny").replace("kṣ", "ksh")
    text = text.replace("ṭh", "th").replace("ḍh", "dh")

    if script == "malayalam":
        text = text.replace("ṟṟ", "tt").replace("nṟ", "nt")
    text = text.replace("ṟ", "r")

    # anusvara (the dot over a letter)
    text = text.replace("m\u0310", "ṁ").replace("ṃ", "ṁ")
    if north:
        text = re.sub(r"ṁ(?=[pbm])", "m", text).replace("ṁ", "n")
    else:
        text = text.replace("ṁ", "m")

    # ng / nj sounds
    if not north:
        text = re.sub(r"ṅ(?=[" + VOWEL_LETTERS + "])", "ng", text)
    text = text.replace("ṅ", "n")
    text = re.sub(r"ñ(?=[" + VOWEL_LETTERS + "])", "nj", text).replace("ñ", "n")

    text = text.replace("ś", "sh").replace("ṣ", "sh").replace("ḥ", "h")
    text = text.replace("ṇ", "n").replace("ṉ", "n").replace("ṭ", "t").replace("ḍ", "d")

    if script != "tamil":  # Tamil already wrote its own "ch"
        text = re.sub(r"ch|c", lambda m: "chh" if m.group() == "ch" else "ch", text)

    # Strip diacritics cleanly (ā -> a, ī -> i, ū -> u, ē -> e, ō -> o, etc.)
    for accented, plain in {
        "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o", "è": "e", "ò": "o",
        "ṝ": "ri", "ṛ": "ri", "ḹ": "li", "ḷ": "l",
    }.items():
        text = text.replace(accented, plain)

    text = text.replace(SHORT_I_PLACEHOLDER, "i")

    # safety net: drop any remaining combining accent mark
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return text


def _romanize_run(run: str, script: str) -> str:
    """Romanize one word-sized piece of text that is all in one script."""
    first, _last, scheme = SCRIPTS[script]
    run = run.replace("\u200c", "").replace("\u200d", "")

    if script == "malayalam":
        for chillu, replacement in MALAYALAM_CHILLU.items():
            run = run.replace(chillu, replacement)
    if script in NORTH_SCRIPTS:
        run = _delete_schwas(run, first)
    if script == "devanagari":
        run = _apply_nukta_overrides(run)
    if script in SOUTH_SCRIPTS:
        run = _protect_south_short_i(run, script)

    iso_text = transliterate(run, scheme, sanscript.ISO)
    if script == "devanagari":
        iso_text = _restore_nukta_placeholders(iso_text)
    if script == "tamil":
        iso_text = _tamil_fix(iso_text)
    return _make_readable(iso_text, script)


def romanize_indic_text(text: str) -> str:
    """Convert Indian-script text to readable Roman letters; other characters stay as they are."""
    if not text:
        return text

    pieces = []
    run = ""
    run_script = None

    def flush():
        nonlocal run, run_script
        if run:
            pieces.append(_romanize_run(run, run_script))
        run, run_script = "", None

    for character in text:
        script = _detect_script(character)
        if script is None and character in "\u200c\u200d" and run_script is not None:
            run += character                      # invisible joiner inside a word
        elif script is not None and script == run_script:
            run += character
        else:
            flush()
            if script is not None:
                run, run_script = character, script
            else:
                pieces.append("." if character in DANDAS else character)
    flush()
    return "".join(pieces)


# =============================================================================
# Quick test:  python chords_db.py
# =============================================================================
if __name__ == "__main__":
    problems = verify_database()
    if problems:
        print("Database problems found:")
        for problem in problems:
            print("  -", problem)
    else:
        print(f"Database check passed: {len(CHORD_VOICINGS_DB)} chords x {len(INSTRUMENTS)} instruments.")

    print("\nChord lookups:")
    for chord, instrument in [("C", "Ukulele"), ("G", "Guitar"), ("Am", "Violin"), ("Am", "Piano"),
                              ("G#", "Ukulele"), ("Ab", "Ukulele"), ("Gb", "Guitar"), ("D#m", "Piano"),
                              ("Dbm", "Ukulele"), ("E7", "Violin"), ("Dm7", "Piano")]:
        print(f"  {chord:4} on {instrument:8}: {get_verified_voicing(chord, instrument)}")
    print("  Unknown chord with default:", get_verified_voicing("Xyz", "Guitar", default="not available"))

    print("\nRomanizing:")
    samples = [
        ("Hindi", "तुम"), ("Hindi", "प्यार"), ("Hindi", "ज़िंदगी"), ("Hindi", "मोहब्बत"),
        ("Hindi Nukta", "ज़िंदगी ज़रा"),
        ("Malayalam", "പ്രണയം"), ("Malayalam", "ഞാൻ"), ("Malayalam", "എന്റെ"),
        ("Malayalam Short i", "നിനക്ക്"), ("Malayalam Short i", "മിഴി"),
        ("Tamil", "காதல்"), ("Tamil", "தமிழ்"), ("Tamil", "தங்கம்"),
        ("Tamil Short i", "நினைவு"), ("Tamil Short i", "சிரிப்பு"),
        ("Telugu", "ప్రేమ"), ("Kannada", "ಪ್ರೀತಿ"), ("Bengali", "ভালোবাসা"), ("Punjabi", "ਪਿਆਰ"),
        ("Mixed", "I love तुम and ಪ್ರೀತಿ"),
    ]
    for language, sample in samples:
        print(f"  {language:18} {sample:15} -> {romanize_indic_text(sample)}")

    assert romanize_indic_text("ज़िंदगी ज़रा") == "zindagi zara", romanize_indic_text("ज़िंदगी ज़रा")
    assert romanize_indic_text("നിനക്ക്") == "ninakk", romanize_indic_text("നിനക്ക്")
    assert romanize_indic_text("சிரிப்பு") == "sirippu", romanize_indic_text("சிரிப்பு")