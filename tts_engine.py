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
    """Run TTS command directly since 'tts' is available in the container."""
    env = os.environ.copy()
    env["TTS_HOME"] = "/app/models"

    # Use the command as-is since 'tts' is available directly
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
    language_idx: Optional[str] = None,
    speaker_wav: Optional[str] = None,
) -> str:
    """Synthesize speech and return path to audio file. Always chunk for voice cloning to avoid truncation."""
    # Determine chunk length based on whether it's a voice cloning task or not.
    # Voice cloning (XTTS) is more sensitive to long inputs.
    max_chunk_length = 300 if speaker_wav else 500
    chunks = split_text_into_chunks(text, max_length=max_chunk_length)

    if len(chunks) == 1:
        return _synthesize_single_chunk(chunks[0], model_name, speaker_idx, language_idx, speaker_wav)

    logger.info("Splitting synthesis into %d chunks (max_length=%d)", len(chunks), max_chunk_length)
    chunk_files: List[str] = []
    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d (len=%d)", i + 1, len(chunks), len(chunk))
        try:
            chunk_file = _synthesize_single_chunk(chunk, model_name, speaker_idx, language_idx, speaker_wav)
            chunk_files.append(chunk_file)
        except Exception as e:
            logger.error("Failed to synthesize chunk %d: %s", i + 1, e)
            # Clean up previously generated chunks if one fails
            for f in chunk_files:
                Path(f).unlink(missing_ok=True)
            raise Exception(f"Failed to process chunk {i+1}/{len(chunks)}: {e}") from e

    file_id = str(uuid.uuid4())
    final_output_path = OUTPUT_DIR / f"{file_id}.wav"
    logger.info("Concatenating %d chunks into final audio file: %s", len(chunk_files), final_output_path)
    return concatenate_wav_files(chunk_files, str(final_output_path))


def _synthesize_single_chunk(
    text: str,
    model_name: Optional[str] = None,
    speaker_idx: Optional[int] = None,
    language_idx: Optional[str] = None,
    speaker_wav: Optional[str] = None,
) -> str:
    file_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / f"{file_id}.wav"

    cmd = ["tts", "--text", text, "--out_path", str(output_path)]
    if model_name:
        cmd.extend(["--model_name", model_name])
    if speaker_idx is not None:
        cmd.extend(["--speaker_idx", str(speaker_idx)])
    if language_idx is not None:
        cmd.extend(["--language_idx", language_idx])
    if speaker_wav is not None:
        cmd.extend(["--speaker_wav", speaker_wav])
        if model_name and "xtts_v2" in model_name.lower():
            if "--language_idx" not in cmd:
                cmd.extend(["--language_idx", "pt"])  # default for XTTS v2 voice cloning

    # For multi-speaker models like XTTS v2, add default speaker if none specified
    if model_name and "xtts_v2" in model_name.lower() and speaker_idx is None and speaker_wav is None:
        cmd.extend(["--speaker_idx", "Aaron Dreschner"])  # Use first available speaker

    # For multilingual models like XTTS v2, add default language if none specified
    # Skip for voice cloning since the reference voice determines the language
    if model_name and "xtts_v2" in model_name.lower() and language_idx is None and speaker_wav is None:
        cmd.extend(["--language_idx", "en"])  # Default to English

    # Prefer GPU when available (boolean flag, no value)
    cmd.append("--use_cuda")

    logger.info("Starting TTS synthesis: model=%s, text_length=%d, speaker_wav=%s", model_name, len(text), speaker_wav)
    logger.debug("TTS base command: %s", " ".join(cmd))

    try:
        timeout_duration = 600 if speaker_wav else 240
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
        error_details = f"TTS command failed: return_code={result.returncode}, stderr='{result.stderr}', stdout='{result.stdout}'"
        logger.error(error_details)
        raise Exception(error_details)

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
