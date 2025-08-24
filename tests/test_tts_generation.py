import os
import sys
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from utils import preprocess_pt_text, split_text_into_chunks, concatenate_wav_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_WAV = PROJECT_ROOT / "reference_voice.wav"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _tts_command_base():
    """Return the preferred way to invoke TTS on this platform.
    Use console script if available, else fallback to `python -m tts`.
    """
    if shutil.which("tts"):
        return ["tts"]
    return [sys.executable, "-m", "tts"]


def _run_tts(args, timeout):
    env = os.environ.copy()
    # Ensure TTS CLI can find models in the repo to avoid re-downloading
    env["TTS_HOME"] = str(MODELS_DIR)
    # Auto-accept Coqui TOS for non-interactive CI/test
    env["COQUI_TOS_AGREED"] = "1"
    cmd = _tts_command_base() + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return proc


@pytest.mark.integration
@pytest.mark.slow
def test_generate_audio_xtts_clone_voice_cli():
    """Mirror app flow: preprocess -> (maybe) chunk -> synthesize each chunk -> concatenate -> persist to output/."""
    if not REFERENCE_WAV.exists():
        pytest.fail("reference_voice.wav not found; we need it to test")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_out = OUTPUT_DIR / "xtts_test.wav"

    model = "tts_models/multilingual/multi-dataset/xtts_v2"
    raw_text = (
        "Um explorador inglês entrou na Amazônia em 1925 e nunca mais voltou. "
        "O coronel Percy Fawcett procurava uma cidade perdida cheia de ouro. "
        "Ele levou o filho e um amigo na expedição. Os três homens sumiram para sempre na selva. "
        "Mais de 100 pessoas morreram tentando encontrá-los. Fawcett havia encontrado ruínas estranhas antes. "
        "Ele chamava a cidade misteriosa de Z. Cartões postais chegaram depois do sumiço, mas eram falsos. "
        "A Amazônia guardou o segredo até hoje. Ninguém sabe se ele encontrou a cidade dourada ou virou comida de onça."
    )

    # 1) Preprocess (same as /synthesize/clone_voice in app.py)
    preprocessed = preprocess_pt_text(raw_text)

    # 2) Chunk if long and using voice cloning (same logic as tts_engine.synthesize_speech)
    if True:
        # Always chunk for voice cloning to avoid XTTS truncating later sentences
        chunks = split_text_into_chunks(preprocessed, max_length=300)
    else:
        chunks = [preprocessed]

    # 3) Synthesize each chunk using CLI (matching flags used in _synthesize_single_chunk)
    chunk_wavs = []
    with TemporaryDirectory() as td:
        td_path = Path(td)
        for i, chunk in enumerate(chunks):
            chunk_out = td_path / f"chunk_{i:02d}.wav"
            args = [
                "--text",
                chunk,
                "--model_name",
                model,
                "--speaker_wav",
                str(REFERENCE_WAV),
                "--language_idx",
                "pt",
                "--out_path",
                str(chunk_out),
            ]
            result = _run_tts(args, timeout=900)
            if result.returncode != 0:
                pytest.fail(
                    f"TTS CLI failed (xtts_v2 clone) on chunk {i}: rc={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
            assert chunk_out.exists(), f"Chunk WAV not created: {chunk_out}"
            assert chunk_out.stat().st_size > 1000, f"Chunk WAV too small: {chunk_out}"
            chunk_wavs.append(str(chunk_out))

        # 4) Concatenate into final_out (same utility as app uses)
        concatenate_wav_files(chunk_wavs, str(final_out))

    assert final_out.exists(), "Final output WAV not created"
    assert final_out.stat().st_size > 1000, "Final output WAV too small"
