import os
import uuid
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from utils import split_text_into_chunks, concatenate_wav_files

logger = logging.getLogger(__name__)

# Directories
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_tts_command(cmd_list, timeout_seconds: int):
    env = os.environ.copy()
    env["PATH"] = "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
    env["TTS_HOME"] = "/app/models"
    return subprocess.run(
        cmd_list,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )


def synthesize_speech(
    text: str,
    model_name: Optional[str] = None,
    speaker_idx: Optional[int] = None,
    language_idx: Optional[int] = None,
    speaker_wav: Optional[str] = None,
) -> str:
    """Synthesize speech and return path to audio file. Always chunk for voice cloning to avoid truncation."""
    # Always chunk when cloning voice (XTTS can truncate with long inputs or multiple newlines)
    if speaker_wav:
        chunks = split_text_into_chunks(text, max_length=300)
        if len(chunks) == 1:
            return _synthesize_single_chunk(chunks[0], model_name, speaker_idx, language_idx, speaker_wav)
        logger.info("Voice cloning: split into %d chunks", len(chunks))
        chunk_files: List[str] = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (len=%d)", i + 1, len(chunks), len(chunk))
            chunk_file = _synthesize_single_chunk(chunk, model_name, speaker_idx, language_idx, speaker_wav)
            chunk_files.append(chunk_file)
        file_id = str(uuid.uuid4())
        final_output_path = OUTPUT_DIR / f"{file_id}.wav"
        logger.info("Concatenating %d chunks into final audio", len(chunk_files))
        return concatenate_wav_files(chunk_files, str(final_output_path))
    else:
        return _synthesize_single_chunk(text, model_name, speaker_idx, language_idx, speaker_wav)


def _synthesize_single_chunk(
    text: str,
    model_name: Optional[str] = None,
    speaker_idx: Optional[int] = None,
    language_idx: Optional[int] = None,
    speaker_wav: Optional[str] = None,
) -> str:
    file_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"{file_id}.wav"

    cmd = ["/opt/venv/bin/tts", "--text", text, "--out_path", str(output_path)]
    if model_name:
        cmd.extend(["--model_name", model_name])
    if speaker_idx is not None:
        cmd.extend(["--speaker_idx", str(speaker_idx)])
    if language_idx is not None:
        cmd.extend(["--language_idx", str(language_idx)])
    if speaker_wav is not None:
        cmd.extend(["--speaker_wav", speaker_wav])
        if model_name and "xtts_v2" in model_name.lower():
            if "--language_idx" not in cmd:
                cmd.extend(["--language_idx", "pt"])  # default for XTTS v2 voice cloning

    logger.info("Starting TTS synthesis: model=%s, text_length=%d, speaker_wav=%s", model_name, len(text), speaker_wav)
    logger.debug("TTS base command: %s", " ".join(cmd))

    try:
        timeout_duration = 600 if speaker_wav else 120
        result = _run_tts_command(cmd, timeout_duration)
        logger.debug("TTS command completed: return_code=%s", result.returncode)
        logger.debug("TTS stdout: %s", result.stdout)
        logger.debug("TTS stderr: %s", result.stderr)

        if result.returncode == 0 and output_path.exists():
            file_size = output_path.stat().st_size
            logger.info("TTS synthesis successful: %s (%d bytes)", output_path, file_size)
            return str(output_path)

        # Retry logic for XTTS v2 pt-br -> pt
        if result.returncode != 0 and model_name and "xtts_v2" in model_name.lower():
            try:
                lang_arg_index = cmd.index("--language_idx") + 1 if "--language_idx" in cmd else -1
            except Exception:
                lang_arg_index = -1
            current_lang = cmd[lang_arg_index] if lang_arg_index != -1 else None
            if current_lang == "pt-br":
                logger.info("Retrying XTTS v2 synthesis with language_idx=pt")
                cmd_retry = [x if x != "pt-br" else "pt" for x in cmd]
                result = _run_tts_command(cmd_retry, timeout_duration)
                logger.debug("Retry return_code=%s", result.returncode)
                logger.debug("Retry stdout: %s", result.stdout)
                logger.debug("Retry stderr: %s", result.stderr)
                if result.returncode == 0 and output_path.exists():
                    file_size = output_path.stat().st_size
                    logger.info("TTS synthesis successful on retry: %s (%d bytes)", output_path, file_size)
                    return str(output_path)

        # Failure path
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise Exception(f"TTS command failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("TTS synthesis timeout after %d seconds", timeout_duration)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    except Exception:
        logger.error("Error during TTS synthesis", exc_info=True)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
