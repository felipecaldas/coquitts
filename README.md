# CoquiTTS Docker API with Voice Cloning

A containerized REST API for CoquiTTS Text-to-Speech synthesis with support for Portuguese models and custom voice cloning.

## Features

- 🎯 **Portuguese TTS** - Dedicated endpoint for Portuguese synthesis
- 🎭 **Voice Cloning** - Clone your own voice using XTTS v2
- 🌍 **Multi-language Support** - Access to all CoquiTTS models
- 🐳 **Docker Ready** - Fully containerized application
- 📝 **REST API** - Easy integration with any application

## Quick Start

### 1. Build and Run the Container

```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# Check if it's running
curl http://localhost:8000/health
```

### 2. Test Basic Functionality

```bash
# Test Portuguese synthesis
curl -X POST "http://localhost:8000/synthesize/portuguese" \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, como você está hoje?"}' \
  --output portuguese_test.wav
```

## Voice Cloning Setup

### Step 1: Prepare Your Reference Audio

1. **Audio Requirements:**
   - Duration: 6-10 seconds of clear speech
   - Format: **WAV or MP3** (API supports both!)
   - Content: Natural, clear pronunciation
   - No background noise or music

2. **Option A: Use MP3 Directly (Easiest)**
   ```bash
   # If your audio is already in MP3 format, use it directly!
   # The API will automatically convert it to the proper format
   # Just make sure it contains clear, professional-quality speech
   ```

3. **Option B: Pre-convert for Best Performance**
   ```bash
   # Using ffmpeg to extract and convert from MP3
   ffmpeg -i your_recording.mp3 -ss 00:01:30 -t 00:00:08 -ar 22050 -ac 1 reference_voice.wav
   
   # Using ffmpeg to extract and convert from WAV
   ffmpeg -i your_recording.wav -ss 00:01:30 -t 00:00:08 -ar 22050 -ac 1 reference_voice.wav
   
   # Parameters:
   # -ss: start time (1 minute 30 seconds)
   # -t: duration (8 seconds)
   # -ar: sample rate (22050 Hz)
   # -ac: audio channels (1 = mono)
   ```

### Step 2: Add Reference Audio to Container

#### Option A: Copy to Running Container
```bash
# Copy WAV file (if you pre-converted)
docker cp reference_voice.wav coquitts-docker:/app/reference_audio/reference_voice.wav

# OR copy MP3 file directly (API will auto-convert)
docker cp your_recording.mp3 coquitts-docker:/app/reference_audio/reference_voice.mp3

# Restart the container to ensure it's loaded
docker-compose restart
```

#### Option B: Mount Volume (Recommended)
1. Create a local directory:
   ```bash
   mkdir -p ./reference_audio
   
   # Copy WAV file (if you pre-converted)
   cp reference_voice.wav ./reference_audio/
   
   # OR copy MP3 file directly (API will auto-convert)
   cp your_recording.mp3 ./reference_audio/reference_voice.mp3
   ```

2. Update `docker-compose.yml`:
   ```yaml
   services:
     coqui-tts:
       # ... existing configuration
       volumes:
         - ./reference_audio:/app/reference_audio
   ```

3. Restart the container:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Step 3: Test Voice Cloning

```bash
# Test your cloned voice
curl -X POST "http://localhost:8000/synthesize/clone_voice" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is my cloned voice speaking!"}' \
  --output my_cloned_voice.wav

# Test in Portuguese
curl -X POST "http://localhost:8000/synthesize/clone_voice" \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, esta é a minha voz clonada falando em português!"}' \
  --output my_voice_portuguese.wav
```

## API Endpoints

### 🎭 Voice Cloning
```http
POST /synthesize/clone_voice
Content-Type: application/json

{
  "text": "Your text to synthesize with your cloned voice"
}
```

### 🇵🇹 Portuguese Synthesis
```http
POST /synthesize/portuguese
Content-Type: application/json

{
  "text": "Texto em português para sintetizar"
}
```

### 🌍 General Synthesis
```http
POST /synthesize
Content-Type: application/json

{
  "text": "Text to synthesize",
  "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
  "speaker_idx": 0,
  "language_idx": 0
}
```

### 📋 List Models
```http
GET /models                    # All available models
GET /models/portuguese         # Portuguese-specific models
```

### 🧹 Cleanup
```http
DELETE /cleanup               # Remove generated audio files
```

## Voice Cloning Tips

### 🎯 **Best Practices for Reference Audio**

1. **Quality Matters:**
   - Use your professional microphone recording
   - Choose the clearest, most natural segment
   - Avoid segments with hesitations or filler words

2. **Content Selection:**
   - Pick a segment with varied phonemes
   - Avoid monotone or whispered speech
   - Natural conversational tone works best

3. **Technical Requirements:**
   - 6-10 seconds duration (sweet spot: 8 seconds)
   - 22kHz sample rate
   - Mono channel
   - WAV format

### 🔧 **Troubleshooting**

**Voice cloning returns error 404:**
```bash
# Check if reference audio exists
docker exec coquitts-docker ls -la /app/reference_audio/

# If missing, copy your reference file (WAV or MP3)
docker cp reference_voice.wav coquitts-docker:/app/reference_audio/reference_voice.wav
# OR
docker cp your_recording.mp3 coquitts-docker:/app/reference_audio/reference_voice.mp3
```

**Poor voice quality:**
- Try a different 8-second segment from your recording
- Ensure the reference audio is clean and clear
- Check that the audio format is correct (22kHz, mono, WAV)

**Synthesis takes too long:**
- First run downloads the XTTS v2 model (~1.5GB)
- Subsequent requests should be much faster
- Check container logs: `docker-compose logs -f`

## File Structure

```
coquitts-docker/
├── app.py                 # Main FastAPI application
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose setup
├── README.md            # This file
└── reference_audio/     # Your voice reference files
    └── reference_voice.wav
```

## Advanced Usage

### Multiple Reference Voices

To support multiple reference voices, you can extend the API:

1. Create subdirectories in `reference_audio/`:
   ```
   reference_audio/
   ├── voice1/reference_voice.wav
   ├── voice2/reference_voice.wav
   └── voice3/reference_voice.wav
   ```

2. Modify the endpoint to accept a voice parameter

### Language-Specific Cloning

The XTTS v2 model supports multiple languages. Your cloned voice will work with:
- Portuguese (pt)
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- And more!

## Performance Notes

- **First Request**: May take 2-3 minutes (model download)
- **Subsequent Requests**: 10-30 seconds depending on text length
- **Memory Usage**: ~4GB RAM recommended
- **Storage**: ~2GB for models + generated audio files

## Support

If you encounter issues:

1. Check container logs: `docker-compose logs -f`
2. Verify reference audio format and location
3. Ensure sufficient disk space and memory
4. Test with shorter text first

## License

This project uses CoquiTTS, which is licensed under MPL 2.0. Please refer to the original CoquiTTS repository for licensing details.
