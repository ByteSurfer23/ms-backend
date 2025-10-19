import os
import io
import re
import gc
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import dateparser
import whisper
from transformers import pipeline

# ---------- Docker-friendly cache ----------
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"

# ---------- FastAPI setup ----------
app = FastAPI(
    title="Whisper Base + DistilGPT2 Summarizer",
    description="Transcribe WAV audio and generate summary locally",
    docs_url="/"
)

# Allow all origins for external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ---------- Load models ONCE at startup ----------
@app.on_event("startup")
def load_models():
    global whisper_model, generator

    print("🎙️ Loading Whisper small model...")
    whisper_model = whisper.load_model("small")  # Base model
    print("✅ Whisper small loaded")

    print("🤖 Loading Summarizing model model...")
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=-1)
    print("✅ Summarizing model loaded")


# ---------- Helper: Extract deadlines ----------
def extract_deadlines_simple(text: str):
    weekday_pattern = r"\b(?:for|on|by)?\s*(next|previous|following|this|last|coming)?\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
    date_patterns = [
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    deadlines = []
    for sentence in sentences:
        sentence_clean = re.sub(r'[.,;:!?]', '', sentence)
        matches = re.findall(weekday_pattern, sentence_clean, flags=re.IGNORECASE)
        for modifier, weekday in matches:
            mod = modifier.lower() if modifier else "this"
            deadlines.append({"date": f"{mod} {weekday}", "subject": sentence.strip()})
        for pattern in date_patterns:
            matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
            for match in matches:
                parsed_date = dateparser.parse(match, settings={'RELATIVE_BASE': datetime.today()})
                if parsed_date:
                    parsed_date = parsed_date.replace(year=datetime.today().year)
                    deadlines.append({"date": parsed_date.strftime("%Y-%m-%d"), "subject": sentence.strip()})
    if not deadlines:
        deadlines = [{"date": "No deadlines found 🎉", "subject": ""}]
    return deadlines


# ---------- Helper: Generate summary ----------
def generate_summary(transcript_text: str, max_tokens: int = 128):
    prompt = f"Summarize the following meeting transcript clearly and professionally:\n{transcript_text}"
    gen_results = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.1,
        truncation=True,
        num_return_sequences=1
    )
    summary = gen_results[0]["generated_text"].strip()
    summary = re.sub(r"http\S+|www\.\S+", "", summary)
    summary = " ".join(line.strip() for line in summary.split("\n") if line.strip())
    del gen_results
    gc.collect()
    return summary


# ---------- Audio processing ----------
async def process_audio(file_bytes: bytes):
    try:
        with io.BytesIO(file_bytes) as f:
            audio, _ = sf.read(f, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1, dtype='float32')

        result = whisper_model.transcribe(audio, fp16=False, language="en")
        segments = result.get("segments", [])
        timestamped_text = "\n".join([f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text'].strip()}" for seg in segments])
        transcript_text = result["text"].strip()

        deadlines = extract_deadlines_simple(transcript_text)
        summary = generate_summary(transcript_text, max_tokens=128)

        # Clear temporary memory
        del audio, result
        gc.collect()

        return {
            "timestamped_text": timestamped_text,
            "summary": summary,
            "deadlines": deadlines
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- API Routes ----------
@app.post("/upload-wav/")
async def upload_wav(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files allowed")
    file_bytes = await file.read()
    return await process_audio(file_bytes=file_bytes)


# ---------- Run server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
