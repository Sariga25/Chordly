"""
mode_b_media.py - Chordly, Mode B (media preparation)

Takes a YouTube/web URL or a local .mp4/.mp3 file and prepares it for analysis:
  1. Saves the media into a clean working folder
  2. Extracts a mono 22050 Hz WAV file (audio.wav)
  3. If it's a video, saves 3 still frames (frame_1.jpg, frame_2.jpg, frame_3.jpg)
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import yt_dlp

# Automatically locate the bundled ffmpeg executable inside your virtual environment
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def _prepare_clean_folder(output_dir: str) -> Path:
    """Create the folder if needed, then delete everything already inside it."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    return folder


def _download_from_url(url: str, folder: Path) -> tuple[Path, str]:
    """Download a video with yt-dlp. Returns (file_path, title)."""
    options = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG_PATH,
        "outtmpl": str(folder / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        # Ensure we point to the merged .mp4 file if yt-dlp merged streams
        downloaded_path = folder / "source.mp4"
        if not downloaded_path.exists():
            downloaded_path = Path(ydl.prepare_filename(info))
        title = info.get("title") or downloaded_path.stem

    return downloaded_path, title


def _copy_local_file(source: str, folder: Path) -> tuple[Path, str]:
    """Copy a local file into the folder. Returns (copied_path, title)."""
    original = Path(source)
    copied_path = folder / f"source{original.suffix}"
    shutil.copy2(original, copied_path)
    return copied_path, original.name


def _extract_audio(media_path: Path, audio_path: Path) -> None:
    """Use ffmpeg to create a mono, 22050 Hz WAV file."""
    command = [
        FFMPEG_PATH,
        "-y",                   # overwrite the output file if it exists
        "-i", str(media_path),  # input file
        "-vn",                  # ignore any video, we only want audio
        "-ac", "1",             # 1 audio channel = mono
        "-ar", "22050",         # sample rate = 22050 Hz
        str(audio_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg was not found. Make sure imageio-ffmpeg is installed."
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{error.stderr}")


def _extract_frames(media_path: Path, folder: Path) -> list:
    """Save frames at 25%, 50%, 75% of the video. Returns [] for audio-only files."""
    capture = cv2.VideoCapture(str(media_path))

    if not capture.isOpened():
        return []  # OpenCV can't read it as a video (typical for .mp3)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 0 frames = no video. 1 frame is usually just MP3 cover art, not a real video.
    if total_frames <= 1:
        capture.release()
        return []

    frame_paths = []
    for number, percent in enumerate([0.25, 0.50, 0.75], start=1):
        frame_index = int(total_frames * percent)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # jump to that frame
        success, frame = capture.read()

        if success:
            frame_path = folder / f"frame_{number}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

    capture.release()
    return frame_paths


def prepare_media(source: str, output_dir: str = "temp_media") -> dict:
    """
    Prepare a URL or local file for Chordly.

    Returns: {"audio_path": str, "frame_paths": [str, ...], "title": str}
    """
    source = source.strip().strip('"').strip("'")
    is_url = source.startswith(("http://", "https://"))

    if not is_url:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Could not find the file: {source}")
        if Path(output_dir).resolve() in source_path.resolve().parents:
            raise ValueError(
                f"The source file is inside '{output_dir}', which gets cleared. "
                "Please move it somewhere else."
            )

    folder = _prepare_clean_folder(output_dir)

    if is_url:
        media_path, title = _download_from_url(source, folder)
    else:
        media_path, title = _copy_local_file(source, folder)

    audio_path = folder / "audio.wav"
    _extract_audio(media_path, audio_path)

    frame_paths = _extract_frames(media_path, folder)

    return {
        "audio_path": str(audio_path),
        "frame_paths": frame_paths,
        "title": title,
    }


if __name__ == "__main__":
    user_source = input("Paste a video URL or a local file path: ")

    try:
        result = prepare_media(user_source)
        print("\nDone!")
        print("Title:      ", result["title"])
        print("Audio file: ", result["audio_path"])
        if result["frame_paths"]:
            print("Frames:")
            for path in result["frame_paths"]:
                print("  -", path)
        else:
            print("Frames:      none (audio-only file)")
    except Exception as error:
        print(f"\nSomething went wrong: {error}")
