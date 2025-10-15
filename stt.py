import whisper
import soundfile as sf
import numpy as np
import io
import uuid
import os

# Load Whisper model once
WHISPER_MODEL = "small"
print("Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded ✅")

def transcribe_wav(file_bytes: bytes, filename_prefix: str = "audio", save_to_disk: bool = True) -> dict:
    """
    Transcribe a WAV file given as bytes using Whisper.
    Returns a dict with:
      - transcript_with_timestamps: text with start/end times
      - Optional saved text file path if save_to_disk=True
    """

    # Read WAV from bytes
    with io.BytesIO(file_bytes) as f:
        audio, sr = sf.read(f)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Whisper transcription
    result = model.transcribe(audio, fp16=False, language="en")
    segments = result.get("segments", [])
    
    # Prepare text with timestamps
    transcript_with_timestamps = "\n".join(
        [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in segments]
    )

    saved_file_path = None
    if save_to_disk:
        unique_id = uuid.uuid4().hex
        saved_file_path = f"{filename_prefix}_{unique_id}_transcript.txt"
        with open(saved_file_path, "w", encoding="utf-8") as f:
            f.write(transcript_with_timestamps)

    return {
        "transcript_with_timestamps": transcript_with_timestamps,
        "text_file": saved_file_path
    }
