# backend/app.py
import os
import tempfile
import logging
import traceback
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch
import soundfile as sf

from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import load_voices

# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tortoise-TTS Service")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_methods=["*"],
    allow_headers=["*"],
)
# Performance-optimized TTS configuration
tts = TextToSpeech(
    models_dir=MODELS_DIR,
    autoregressive_batch_size=1,  # Reduce memory usage
    enable_redaction=False,       # Disable for faster processing
)

# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "random"
    preset: str = "ultra_fast"  # Changed from "fast" to "ultra_fast"

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    try:
        logger.info(f"Processing text: {req.text[:50]}...")
        
        # Load voice samples
        voices = req.voice.split(",")
        voice_samples, conditioning_latents = load_voices(voices, [])

        # Generate with performance optimizations
        audio_batches = tts.tts_with_preset(
        req.text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset="ultra_fast",          # Built-in fast preset
        k=1,                         # Only generate 1 candidate
        num_autoregressive_samples=1, # Only 1 sample (was 16)
        diffusion_iterations=20,      # Reduced from 30-100
        length_penalty=1.0          # Faster generation
    )
        # Process audio
        wav_data = audio_batches[0].squeeze(0).cpu().numpy().astype(np.float32)
        wav_data = np.squeeze(wav_data)  # Ensure mono

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(
                tmp.name,
                wav_data,
                24000,
                format="WAV",
                subtype="PCM_16"
            )
            return FileResponse(
                tmp.name,
                media_type="audio/wav",
                filename="speech.wav",
            )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")