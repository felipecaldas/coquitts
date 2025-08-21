import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from utils import (
    parse_models_output,
    get_available_models,
    preprocess_pt_text,
    get_wav_duration_seconds,
    count_words,
)
from tts_engine import synthesize_speech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CoquiTTS API",
    description="REST API for CoquiTTS Text-to-Speech synthesis",
    version="1.0.0"
)

# Request models
class TTSRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
    speaker_idx: Optional[int] = None
    language_idx: Optional[int] = None

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str
    dataset: str
    language: str

class TextRequest(BaseModel):
    text: str

# Directories
output_dir = Path("/app/output")
output_dir.mkdir(exist_ok=True)
reference_audio_dir = Path("/app/reference_audio")
reference_audio_dir.mkdir(exist_ok=True)

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "CoquiTTS API Server",
        "version": "1.0.0",
        "endpoints": {
            "/models": "List all available TTS models",
            "/models/portuguese": "List Portuguese TTS models",
            "/synthesize": "Synthesize speech from text",
            "/synthesize/portuguese": "Quick Portuguese synthesis",
            "/synthesize/clone_voice": "Synthesize speech using your cloned voice (Brazilian Portuguese)",
        },
    }

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "service": "coqui-tts-api"}

@app.get("/models")
async def list_models():
    try:
        models_list = get_available_models()
        if "No models found" in models_list or "Error" in models_list or "unavailable" in models_list:
            return {
                "status": "warning",
                "message": models_list,
                "models": [],
                "count": 0,
            }
        return Response(content=models_list, media_type="text/plain")
    except Exception as e:
        error_msg = f"Error listing models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg, "models": [], "count": 0}

@app.get("/models/portuguese")
async def list_portuguese_models():
    try:
        models_output = get_available_models()
        if isinstance(models_output, str) and (
            "No models found" in models_output or "Error" in models_output or "unavailable" in models_output
        ):
            return {"models": [], "count": 0, "message": models_output, "status": "warning"}

        all_models = parse_models_output(models_output)
        pt_models = [m for m in all_models if m.get("language", "").lower() in ["pt", "portuguese", "pt-br", "pt_br"]]
        multilingual_models = [m for m in all_models if m.get("language", "").lower() == "multilingual"]
        all_pt_models = pt_models + multilingual_models
        logger.info("Found %d Portuguese-specific and %d multilingual models", len(pt_models), len(multilingual_models))
        return {
            "models": all_pt_models,
            "count": len(all_pt_models),
            "portuguese_specific": len(pt_models),
            "multilingual": len(multilingual_models),
            "status": "success",
        }
    except Exception as e:
        error_msg = f"Error listing Portuguese models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg, "models": [], "count": 0}

@app.post("/synthesize")
async def synthesize_text(request: TTSRequest):
    try:
        if not request.text or not request.text.strip():
            logger.warning("Synthesis request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        logger.info("Starting synthesis: text='%s...', model=%s", request.text[:50], request.model_name)
        audio_path = synthesize_speech(
            text=request.text,
            model_name=request.model_name,
            speaker_idx=request.speaker_idx,
            language_idx=request.language_idx,
        )
        logger.info("Synthesis completed successfully: %s", audio_path)
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"synthesis_{Path(audio_path).stem}.wav",
            headers={
                "Content-Disposition": "attachment",
                "X-Audio-Duration": str(round(get_wav_duration_seconds(audio_path), 3)),
                "X-Word-Count": str(count_words(request.text)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Synthesis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/synthesize/clone_voice")
async def synthesize_with_cloned_voice(request: TextRequest):
    try:
        if not request.text or not request.text.strip():
            logger.warning("Voice cloning synthesis request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        reference_audio_path = reference_audio_dir / "reference_voice.wav"
        if not reference_audio_path.exists():
            logger.error("Reference audio not found: %s", reference_audio_path)
            raise HTTPException(
                status_code=404,
                detail=(
                    "Reference voice audio not found. Please add your reference audio file to "
                    "/app/reference_audio/reference_voice.wav"
                ),
            )

        logger.info("Starting voice cloning synthesis: text='%s...', reference=%s", request.text[:50], reference_audio_path)
        processed_text = preprocess_pt_text(request.text)
        audio_path = synthesize_speech(
            text=processed_text,
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            speaker_wav=str(reference_audio_path),
            language_idx="pt",
        )
        logger.info("Voice cloning synthesis completed successfully: %s", audio_path)
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"cloned_voice_synthesis_{Path(audio_path).stem}.wav",
            headers={
                "Content-Disposition": "attachment",
                "X-Audio-Duration": str(round(get_wav_duration_seconds(audio_path), 3)),
                "X-Word-Count": str(count_words(processed_text)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Voice cloning synthesis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/cleanup")
async def cleanup_audio_files():
    try:
        deleted_count = 0
        total_size = 0
        for audio_file in output_dir.glob("*.wav"):
            file_size = audio_file.stat().st_size
            audio_file.unlink()
            deleted_count += 1
            total_size += file_size
            logger.info("Deleted audio file: %s (%d bytes)", audio_file, file_size)
        logger.info("Cleanup completed: %d files, %d bytes freed", deleted_count, total_size)
        return {"message": f"Cleaned up {deleted_count} audio files", "files_deleted": deleted_count, "bytes_freed": total_size}
    except Exception as e:
        logger.error("Error during cleanup: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)