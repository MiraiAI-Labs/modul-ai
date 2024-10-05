
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import io
import soundfile as sf
import numpy as np
import webrtcvad

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextSubmission(BaseModel):
    text: str

NOISE_THRESHOLD = 0.02
VAD_AGGRESSIVENESS = 2
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def is_noisy(audio_data, threshold):
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 1 if rms > threshold else 0

def is_speech(audio_data, sample_rate):
    audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
    frame_duration = 30
    frame_size = int(sample_rate * frame_duration / 1000)
    
    for i in range(0, len(audio_bytes), frame_size * 2):
        frame = audio_bytes[i:i + frame_size * 2]
        if len(frame) == frame_size * 2 and vad.is_speech(frame, sample_rate):
            return True
    return False

@app.post("/submit_audio")
async def submit_audio(file: UploadFile = File(...)):
    if file.content_type in ["audio/wav", "audio/mp3"]:
        try:
            contents = await file.read()
            audio_io = io.BytesIO(contents)
            audio_data, sr = sf.read(audio_io)

            if is_speech(audio_data, sr):
                return JSONResponse(status_code=200, content={"message": "Pembicaraan terdeteksi, bukan noise"})
            
            noise_status = is_noisy(audio_data, NOISE_THRESHOLD)
            return JSONResponse(status_code=200, content={"message": "Pindah ke tempat yang lebih tenang" if noise_status else "Tempat sudah kondusif!"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"Terjadi kesalahan saat memproses file audio. Error:\n{str(e)}"})
    
    else:
        return JSONResponse(status_code=400, content={"message": "Format file tidak didukung"})
