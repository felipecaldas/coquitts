import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import subprocess
import json
import traceback

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

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
    model_name: str
    model_type: str
    dataset: str
    language: str

# Global variables
models_cache = None
output_dir = Path("/app/output")
output_dir.mkdir(exist_ok=True)

def parse_models_output(models_output: str) -> List[Dict[str, Any]]:
    """Parse the raw TTS model output into structured data"""
    models = []
    
    if not models_output or "No models found" in models_output:
        return models
    
    lines = models_output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip header lines and empty lines
        if not line or line.startswith('Name format:') or line.startswith('=') or line.startswith('Path'):
            continue
            
        # Parse numbered model lines like "61: tts_models/pt/cv/vits"
        if ':' in line and 'tts_models' in line:
            # Extract the actual model name after the number
            model_part = line.split(':', 1)[1].strip()
            
            if model_part.startswith('tts_models/'):
                parts = model_part.split('/')
                if len(parts) >= 3:
                    language = parts[1]
                    dataset = parts[2] if len(parts) > 2 else "unknown"
                    model_name = parts[-1] if len(parts) > 2 else parts[2]
                    
                    model_info = {
                        "model_name": model_part,
                        "language": language,
                        "type": "tts",
                        "dataset": dataset,
                        "full_line": line
                    }
                    models.append(model_info)
    
    logger.debug(f"Parsed {len(models)} models from output")
    for model in models:
        logger.debug(f"Found model: {model['model_name']} (language: {model['language']})")
    
    return models

def get_available_models() -> str:
    """Get raw output of tts --list_models command"""
    global models_cache
    
    if models_cache is None:
        try:
            logger.info("Fetching TTS models list...")
            
            # Set up environment
            env = os.environ.copy()
            env["PATH"] = "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
            env["TTS_HOME"] = "/app/models"
            
            logger.info(f"Environment: PATH={env['PATH']}, TTS_HOME={env['TTS_HOME']}")
            
            # Define the TTS command path
            tts_cmd = "/opt/venv/bin/tts"
            
            # Run the actual command with full path - increased timeout for initial model loading
            result = subprocess.run(
                [tts_cmd, "--list_models"],
                capture_output=True,
                text=True,
                timeout=180,
                env=env
            )
            
            logger.info(f"TTS command completed with return code: {result.returncode}")
            
            if result.returncode == 0:
                if not result.stdout.strip():
                    logger.warning("TTS command succeeded but returned no output")
                    models_cache = "No models found. The TTS command returned no output."
                else:
                    models_cache = result.stdout
                    model_count = len([line for line in result.stdout.split('\n') if line.strip() and not line.startswith('Model') and not line.startswith('=')])
                    logger.info(f"Successfully fetched {model_count} models")
            else:
                logger.error(f"TTS command failed: {result.stderr}")
                # Try fallback with Python module
                result = subprocess.run(
                    ["/opt/venv/bin/python", "-m", "tts", "--list_models"],
                    capture_output=True,
                    text=True,
                    timeout=120,  # Increased timeout
                    env=env
                )
                
                if result.returncode == 0:
                    models_cache = result.stdout or "No models found."
                    logger.info("Successfully fetched models using Python module fallback")
                else:
                    error_msg = f"Failed to list TTS models: {result.stderr}"
                    logger.error(error_msg)
                    # Return a fallback message instead of failing
                    models_cache = "TTS models not available - this may be due to network issues or missing model downloads"
                
        except subprocess.TimeoutExpired as e:
            error_msg = "Timeout while fetching models. The TTS command took too long to execute. This is normal on first run when models need to be downloaded."
            logger.warning(error_msg)
            models_cache = "Models temporarily unavailable - please try again in a few moments"
            
        except Exception as e:
            error_msg = f"Error fetching models: {str(e)}"
            logger.error(error_msg, exc_info=True)
            models_cache = f"Error accessing TTS models: {str(e)}"
    
    return models_cache

def synthesize_speech(text: str, model_name: str = None, speaker_idx: int = None, language_idx: int = None) -> str:
    """Synthesize speech and return path to audio file"""
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    output_path = output_dir / f"{file_id}.wav"
    
    # Build TTS command
    cmd = ["/opt/venv/bin/tts", "--text", text, "--out_path", str(output_path)]
    
    if model_name:
        cmd.extend(["--model_name", model_name])
    
    if speaker_idx is not None:
        cmd.extend(["--speaker_idx", str(speaker_idx)])
    
    if language_idx is not None:
        cmd.extend(["--language_idx", str(language_idx)])
    
    logger.info(f"Starting TTS synthesis: model={model_name}, text_length={len(text)}")
    logger.debug(f"TTS command: {' '.join(cmd)}")
    
    try:
        # Run TTS command with proper environment
        env = os.environ.copy()
        env["PATH"] = "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
        env["TTS_HOME"] = "/app/models"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
            env=env
        )
        
        logger.debug(f"TTS command completed: return_code={result.returncode}")
        logger.debug(f"TTS stdout: {result.stdout}")
        logger.debug(f"TTS stderr: {result.stderr}")
        
        if result.returncode == 0:
            if output_path.exists():
                file_size = output_path.stat().st_size
                logger.info(f"TTS synthesis successful: {output_path} ({file_size} bytes)")
                return str(output_path)
            else:
                logger.error(f"TTS command succeeded but output file not found: {output_path}")
                raise Exception("Audio file was not created")
        else:
            logger.error(f"TTS command failed: {result.stderr}")
            raise Exception(f"TTS command failed: {result.stderr}")
            
    except subprocess.TimeoutExpired as e:
        logger.error(f"TTS synthesis timeout after 120 seconds")
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail="Timeout during speech synthesis")
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {str(e)}", exc_info=True)
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error during synthesis: {str(e)}")

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
            "/synthesize/portuguese": "Quick Portuguese synthesis"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "service": "coqui-tts-api"}

@app.get("/models")
async def list_models():
    """List all available TTS models"""
    try:
        models_list = get_available_models()
        
        if "No models found" in models_list or "Error" in models_list or "unavailable" in models_list:
            return {
                "status": "warning",
                "message": models_list,
                "models": [],
                "count": 0
            }
        
        model_count = len([line for line in models_list.split('\n') if line.strip() and not line.startswith('Model') and not line.startswith('=')])
        logger.info(f"Returning {model_count} models")
        
        return Response(
            content=models_list,
            media_type="text/plain"
        )
    except Exception as e:
        error_msg = f"Error listing models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "models": [],
            "count": 0
        }

@app.get("/models/portuguese")
async def list_portuguese_models():
    """List Portuguese TTS models"""
    try:
        models_output = get_available_models()
        
        # Handle error cases
        if isinstance(models_output, str) and ("No models found" in models_output or "Error" in models_output or "unavailable" in models_output):
            return {
                "models": [],
                "count": 0,
                "message": models_output,
                "status": "warning"
            }
        
        # Parse the models output
        all_models = parse_models_output(models_output)
        
        # Filter Portuguese models - include both 'pt' and 'pt-br' codes
        pt_models = [m for m in all_models if m.get("language", "").lower() in ["pt", "portuguese", "pt-br", "pt_br"]]
        
        # Also include multilingual models that support Portuguese
        multilingual_models = [m for m in all_models if m.get("language", "").lower() == "multilingual"]
        
        # Combine Portuguese-specific and multilingual models
        all_pt_models = pt_models + multilingual_models
        
        logger.info(f"Found {len(pt_models)} Portuguese-specific and {len(multilingual_models)} multilingual models")
        
        return {
            "models": all_pt_models,
            "count": len(all_pt_models),
            "portuguese_specific": len(pt_models),
            "multilingual": len(multilingual_models),
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Error listing Portuguese models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "models": [],
            "count": 0
        }

@app.post("/synthesize")
async def synthesize_text(request: TTSRequest):
    """Synthesize speech from text"""
    try:
        # Validate text input
        if not request.text or not request.text.strip():
            logger.warning("Synthesis request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Starting synthesis: text='{request.text[:50]}...', model={request.model_name}")
        
        # Synthesize speech
        audio_path = synthesize_speech(
            text=request.text,
            model_name=request.model_name,
            speaker_idx=request.speaker_idx,
            language_idx=request.language_idx
        )
        
        logger.info(f"Synthesis completed successfully: {audio_path}")
        
        # Return audio file
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"synthesis_{Path(audio_path).stem}.wav",
            headers={"Content-Disposition": "attachment"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Synthesis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/synthesize/portuguese")
async def synthesize_portuguese(text: str):
    """Quick endpoint for Portuguese synthesis with default model"""
    try:
        if not text or not text.strip():
            logger.warning("Portuguese synthesis request rejected: empty text")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Starting Portuguese synthesis: text='{text[:50]}...'")
        
        # Use Portuguese model
        audio_path = synthesize_speech(
            text=text,
            model_name="tts_models/pt/cv/vits",
            speaker_idx=None,
            language_idx=None
        )
        
        logger.info(f"Portuguese synthesis completed successfully: {audio_path}")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"portuguese_synthesis_{Path(audio_path).stem}.wav",
            headers={"Content-Disposition": "attachment"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Portuguese synthesis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/cleanup")
async def cleanup_audio_files():
    """Clean up generated audio files"""
    try:
        deleted_count = 0
        total_size = 0
        
        for audio_file in output_dir.glob("*.wav"):
            file_size = audio_file.stat().st_size
            audio_file.unlink()
            deleted_count += 1
            total_size += file_size
            logger.info(f"Deleted audio file: {audio_file} ({file_size} bytes)")
        
        logger.info(f"Cleanup completed: {deleted_count} files, {total_size} bytes freed")
        return {
            "message": f"Cleaned up {deleted_count} audio files",
            "files_deleted": deleted_count,
            "bytes_freed": total_size
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)