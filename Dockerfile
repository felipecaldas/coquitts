FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install CUDA-enabled PyTorch first, then the rest (coqui-tts will reuse installed torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.3.1+cu121 \
      torchaudio==2.3.1+cu121 \
      torchvision==0.18.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure required directories exist
RUN mkdir -p /app/output /app/reference_audio

# Copy reference voice file if present (build arg allows absence)
COPY reference_voice.wav /app/reference_audio/reference_voice.wav

# Environment
ENV PYTHONUNBUFFERED=1 \
    TTS_HOME=/app/models \
    COQUI_TOS_AGREED=1

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]