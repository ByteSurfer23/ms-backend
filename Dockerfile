# ---------- Base Image ----------
    #modification in docker file , more modifications to the docker file
    FROM python:3.12-slim

    # ---------- Working Directory ----------
    WORKDIR /app
    
    # ---------- System Dependencies ----------
    RUN apt-get update && \
        apt-get install -y libsndfile1 git && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- Copy Requirements ----------
    COPY requirements.txt .
    
    # ---------- Install Python Dependencies ----------
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- Copy App ----------
    COPY . .
    
    # ---------- Preload Models ----------
    RUN python -c "import whisper; whisper.load_model('base')" && \
        python -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2')"
    
    # ---------- Expose Port ----------
    EXPOSE 8000
    
    # ---------- Run FastAPI ----------
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    