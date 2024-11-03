import torch
import numpy as np
import sounddevice as sd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from collections import deque

# Load the Whisper model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-tiny"  # Make sure you’re using a compatible model ID

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Parameters for real-time streaming
SAMPLING_RATE = 16000  # Whisper requires 16 kHz audio
CHUNK_DURATION = 2.0   # Duration of each audio chunk in seconds
OVERLAP_DURATION = 1.0 # Overlap duration with the previous chunk in seconds

CHUNK_SAMPLES = int(SAMPLING_RATE * CHUNK_DURATION)
OVERLAP_SAMPLES = int(SAMPLING_RATE * OVERLAP_DURATION)

# Circular buffer to store the previous chunk for overlap context
audio_buffer = deque(maxlen=CHUNK_SAMPLES + OVERLAP_SAMPLES)

def transcribe_audio(audio_data):
    # Normalize the audio data and prepare it for the pipeline
    audio_data = np.array(audio_data).astype(np.float32) / 32768.0  # Normalize audio
    result = pipe(audio_data)  # Remove sampling_rate argument here
    transcription = result["text"]
    return transcription

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio Status:", status)

    # Extend the buffer with the new audio data and maintain overlap
    audio_data = np.squeeze(indata)
    audio_buffer.extend(audio_data)

    # If buffer is full, transcribe the chunk with overlap
    if len(audio_buffer) >= (CHUNK_SAMPLES + OVERLAP_SAMPLES):
        buffer_data = np.array(audio_buffer)  # Prepare buffer as a numpy array
        transcription = transcribe_audio(buffer_data)
        print("Recognized Speech:", transcription)

# Start streaming with sounddevice
with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SAMPLES):
    print("Listening with Whisper... Press Ctrl+C to stop.")
    sd.sleep(int(CHUNK_DURATION * 10000))  # Listen indefinitely
