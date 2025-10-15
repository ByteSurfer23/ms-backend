from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper
import soundfile as sf
import numpy as np
import io
import uuid
import dateparser
# Import transformers for local generation
from transformers import pipeline
from datetime import datetime



import io, uuid, re
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
app = FastAPI(title="Whisper + Local GenAI", description="Transcribe WAV and generate local AI content", docs_url="/")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

print("🎙️ Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("✅ Whisper model loaded")

print("🤖 Loading local GPT-Neo text generation model...")
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
print("✅ Local text generation model loaded")

import io, re, numpy as np, soundfile as sf
from fastapi import HTTPException
from datetime import datetime
import dateparser

async def process_audio(file_bytes: bytes, filename_prefix: str = "audio"):
    try:
        # ----------------------------
        # Step 1: Load and normalize audio
        # ----------------------------
        with io.BytesIO(file_bytes) as f:
            audio, _ = sf.read(f)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # ----------------------------
        # Step 2: Transcribe with Whisper
        # ----------------------------
        result = whisper_model.transcribe(audio, fp16=False, language="en")
        segments = result.get("segments", [])
        timestamped_text = "\n".join(
            [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text'].strip()}" for seg in segments]
        )
        transcript_text = result["text"].strip()

        # ----------------------------
        # Step 3: Generate summary using local GPT-Neo
        # ----------------------------
        prompt = f"""
Summarize the following meeting transcript in a clear, concise, and professional way.
Focus only on the main ideas and discussion points.Do NOT include emails, phone numbers, addresses, telegrams, websites, or any unrelated info.

Transcript:
\"\"\"{transcript_text}\"\"\"
"""
        gen_results = generator(prompt, max_length=500, temperature=0.6, num_return_sequences=1)
        generated_response = gen_results[0]["generated_text"].strip()

        # Clean summary
        generated_response = re.sub(r"http\S+|www\.\S+", "", generated_response)
        generated_response = re.sub(r"(?i)(youtube|reddit|twitter|blog|source).*", "", generated_response).strip()
        # Keep first paragraph only
        summary_lines = generated_response.split("\n")
        clean_summary = " ".join(line.strip() for line in summary_lines if line.strip())

        # ----------------------------
        # Step 4: Extract deadlines (absolute + relative kept as text)
        # ----------------------------
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

                # Relative weekdays: keep text as-is
                matches = re.findall(weekday_pattern, sentence_clean, flags=re.IGNORECASE)
                for modifier, weekday in matches:
                    mod = modifier.lower() if modifier else "this"
                    phrase = f"{mod} {weekday}"
                    deadlines.append({
                        "date": phrase,
                        "subject": sentence.strip()
                    })

                # Absolute dates: convert to ISO
                for pattern in date_patterns:
                    matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
                    for match in matches:
                        parsed_date = dateparser.parse(match, settings={'RELATIVE_BASE': datetime.today()})
                        if parsed_date:
                            parsed_date = parsed_date.replace(year=datetime.today().year)
                            deadlines.append({
                                "date": parsed_date.strftime("%Y-%m-%d"),
                                "subject": sentence.strip()
                            })

            if not deadlines:
                deadlines = [{"date": "No deadlines found 🎉", "subject": ""}]
            return deadlines

        deadlines = extract_deadlines_simple(transcript_text)

        # ----------------------------
        # Step 5: Return JSON
        # ----------------------------
        return {
            "timestamped_text": timestamped_text,
            "summary": clean_summary,
            "deadlines": deadlines
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-wav/")
async def upload_wav(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files allowed")
    file_bytes = await file.read()
    return await process_audio(file_bytes=file_bytes, filename_prefix=file.filename)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
