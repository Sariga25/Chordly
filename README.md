# 🎸 Chordly — Multimodal AI Music & Pan-India Play-Along Coach

**Chordly** is an individual AI project built to help self-taught musicians learn songs and acoustic social-media covers across **Ukulele, Fretless Violin, Guitar, and Piano**, with dedicated support for **Pan-India regional music** (Hindi, Malayalam, Tamil, Telugu, Kannada, Marathi, Bengali, Punjabi) and English tracks.

---

## ✨ Key Features

### 🎵 Mode A: Pan-India Song & Play-Along Coach (`mode_a.py`)
- **Grounded Song & Singer Verification:** Verifies real playback singers and official tracks using the **iTunes Search API** and **LRCLIB API** (with `yt-dlp` fallback), preventing LLM singer and lyric hallucinations.
- **Pan-India Phonetic Transliteration:** Converts native Indian scripts (Devanagari, Malayalam, Tamil, Telugu, Kannada, Bengali) into readable Romanized **Hinglish, Manglish, and Tanglish** via `indic-transliteration`.
- **Smart Capo & Technique Tips:** Automatically calculates beginner-friendly Capo transpositions for Guitar and Ukulele.
- **Custom Lyrics Auto-Alignment:** Paste lyrics in any Indian or English script to identify the track and align chords over every line.

### 🎬 Mode B: Multimodal Reel & Cover Analyzer (`mode_b_media.py` + `mode_b_analyzer.py`)
- **End-to-End Media Ingestion:** Downloads YouTube Shorts/Reels via `yt-dlp` (or accepts local `.mp4`/`.mp3`/`.wav` uploads) and splits video keyframes and mono `22.05 kHz` audio using `imageio-ffmpeg` and `opencv-python`.
- **Vision Instrument Detection:** Uses **Qwen Vision AI (`qwen/qwen3.8-27b`)** on extracted keyframes to detect whether the cover features Ukulele, Violin, Guitar, or Piano.
- **Harmonic DSP + Viterbi Chord Tracking:** Isolates harmonic frequencies (`librosa.effects.harmonic`), tracks musical beats (`beat_track`), matches against 24 Major/Minor chord templates via cosine similarity, and applies **Viterbi sequence smoothing (`librosa.sequence.viterbi`)**.
- **Vocal vs. Instrumental Detection:** Transcribes sung vocals with **Whisper (`whisper-large-v3`)** (with harmonic-percussive vocal isolation fallback for sustained Violin/Piano covers) or outputs clean measure-by-measure chord bars when no vocals are present.
- **Instant Instrument Switching:** Re-arranges any analyzed cover across Ukulele, Violin, Guitar, and Piano in under a second using `st.session_state`.

### 🎼 Deterministic Music Theory & SVG Diagrams (`chords_db.py` + `app.py`)
- **Ukulele & Guitar:** Dynamic inline SVG fretboard grids (`G-C-E-A` and `E-A-D-G-B-e`).
- **Fretless Violin:** Authentic 4-string `G-D-A-E` fingerboard highlighting root/melody notes vs. double-stops and low-2nd-finger (`2↓`) positions.
- **1.5-Octave Piano:** Root-position keyboard diagrams (`C4–E5`) highlighting root notes, sharps/flats, and right-hand finger numbers (`1 - 3 - 5`).

---

## 📁 Project Structure

```text
Chordly/
├── app.py               # Streamlit web UI & custom SVG chord/fingerboard renderers
├── mode_a.py            # Pan-India song recommendation, grounding & play-along arranger
├── mode_b_media.py      # Video/Reel downloader, FFmpeg audio converter & frame extractor
├── mode_b_analyzer.py   # Multimodal Vision + Whisper + Librosa Viterbi cover analyzer
├── chords_db.py         # Deterministic multi-instrument voicings & Indic transliteration
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes .env, venv, and temporary media folders
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack
- **Frontend & Graphics:** `streamlit`, Custom Inline SVG
- **AI & Speech Models (via `groq`):** `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `whisper-large-v3`
- **Audio DSP & Computer Vision:** `librosa`, `numpy`, `opencv-python`
- **Media & Web Grounding:** `yt-dlp`, `imageio-ffmpeg`, `ddgs`, `beautifulsoup4`, iTunes Search API, LRCLIB API
- **Regional Language Processing:** `indic-transliteration`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com<your-username>/Chordly.git
   cd Chordly
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Groq API key(s):**
   Create a `.env` file in the project root and add your Groq key (supports comma-separated keys in `GROQ_API_KEYS` for automatic rate-limit rotation):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_API_KEYS=your_first_key,your_second_key
   ```

5. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
