import os
import re
import wave
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Cache for models list
_models_cache = None


def parse_models_output(models_output: str) -> List[Dict[str, Any]]:
    """Parse the raw TTS model output into structured data."""
    models: List[Dict[str, Any]] = []
    if not models_output or "No models found" in models_output:
        return models

    lines = models_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Name format:') or line.startswith('=') or line.startswith('Path'):
            continue
        if ':' in line and 'tts_models' in line:
            model_part = line.split(':', 1)[1].strip()
            if model_part.startswith('tts_models/'):
                parts = model_part.split('/')
                if len(parts) >= 3:
                    language = parts[1]
                    dataset = parts[2] if len(parts) > 2 else "unknown"
                    model_info = {
                        "model_name": model_part,
                        "language": language,
                        "type": "tts",
                        "dataset": dataset,
                        "full_line": line,
                    }
                    models.append(model_info)
    return models


def get_available_models() -> str:
    """Get raw output of `tts --list_models` with caching and longer initial timeout."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    try:
        logger.info("Fetching TTS models list...")
        env = os.environ.copy()
        env["PATH"] = "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
        env["TTS_HOME"] = "/app/models"

        tts_cmd = "/opt/venv/bin/tts"
        result = subprocess.run([tts_cmd, "--list_models"], capture_output=True, text=True, timeout=180, env=env)
        if result.returncode == 0:
            _models_cache = result.stdout or "No models found. The TTS command returned no output."
            return _models_cache
        # Fallback
        result = subprocess.run(["/opt/venv/bin/python", "-m", "tts", "--list_models"], capture_output=True, text=True, timeout=120, env=env)
        if result.returncode == 0:
            _models_cache = result.stdout or "No models found."
            return _models_cache
        _models_cache = "TTS models not available - this may be due to network issues or missing model downloads"
        return _models_cache
    except subprocess.TimeoutExpired:
        _models_cache = "Models temporarily unavailable - please try again in a few moments"
        return _models_cache
    except Exception as e:
        logger.error("Error fetching models: %s", e, exc_info=True)
        _models_cache = f"Error accessing TTS models: {e}"
        return _models_cache


def preprocess_pt_text(text: str) -> str:
    """Portuguese preprocessing to avoid speaking sentence-ending periods while preserving decimals and abbreviations."""
    text = normalize_text(text)
    if not text:
        return text
    # Protect decimals: 12.34 -> 12<DECIMAL>34
    text = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)
    # Protect common abbreviations by replacing the dot
    abbrs = ["Sr.", "Sra.", "Dr.", "Dra.", "Prof.", "Profa.", "etc.", "p.ex.", "e.g."]
    for ab in abbrs:
        text = text.replace(ab, ab.replace(".", "<DOT>"))
    # Ellipses -> newline pause
    text = re.sub(r"…|\\.\\.\\.", "\n", text)
    # Replace sentence-ending periods with newline (not between digits)
    text = re.sub(r"(?<!\d)\.(?!\d)", "\n", text)
    # Normalize multiple newlines/spaces
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Restore tokens
    text = text.replace("<DECIMAL>", ".").replace("<DOT>", ".")
    return text


def normalize_text(text: str) -> str:
    """Normalize quotes, apostrophes, and spaces that can confuse the TTS CLI.

    - Replace non-breaking spaces with normal spaces
    - Collapse multiple spaces
    """
    if not text:
        return text
    # Curly quotes/apostrophes to straight
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # NBSP and other spaces to normal
    text = text.replace("\u00A0", " ")
    # Collapse multiple spaces
    text = re.sub(r"[ \t\u2009\u200A\u200B\u202F\u205F\u3000]+", " ", text)
    return text


def split_text_into_chunks(text: str, max_length: int = 500) -> List[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    sentences = re.split(r"[.!?\n]+", text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 > max_length and current:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} … {s}".strip(" …") if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def concatenate_wav_files(wav_files: List[str], output_path: str) -> str:
    """Concatenate multiple WAV files into one and remove originals."""
    if not wav_files:
        raise Exception("No WAV files to concatenate")
    if len(wav_files) == 1:
        import shutil
        shutil.copy2(wav_files[0], output_path)
        return output_path

    with wave.open(wav_files[0], 'rb') as first_wav:
        params = first_wav.getparams()
        frames = first_wav.readframes(first_wav.getnframes())
    with wave.open(output_path, 'wb') as output_wav:
        output_wav.setparams(params)
        output_wav.writeframes(frames)
        for wav_file in wav_files[1:]:
            with wave.open(wav_file, 'rb') as wavf:
                frames = wavf.readframes(wavf.getnframes())
                output_wav.writeframes(frames)
    for wav_file in wav_files:
        try:
            Path(wav_file).unlink(missing_ok=True)
        except Exception:
            pass
    return output_path


def get_wav_duration_seconds(wav_path: str) -> float:
    """Return WAV duration in seconds as a float."""
    try:
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate()
            if framerate == 0:
                return 0.0
            return frames / float(framerate)
    except Exception:
        return 0.0


def count_words(text: str) -> int:
    """Heuristic word count using \b\w+\b tokens."""
    return len(re.findall(r"\b\w+\b", text or ""))
