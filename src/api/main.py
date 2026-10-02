"""API FastAPI para classificação de conversas."""

import os
from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from pydantic import BaseModel

from src.models.predict import TriageClassifier

app = FastAPI(
    title="Conversation Triage API",
    description="Classificador de conversas de atendimento: sentimento + tags (multi-label).",
    version="0.1.0",
)

# Monta arquivos estáticos
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'static')
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Carrega modelos
_classifier = TriageClassifier()


class PredictRequest(BaseModel):
    """Schema de entrada."""
    conversation: str


class PredictResponse(BaseModel):
    """Schema de saída."""
    sentiment: str
    tags: List[str]



@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a interface web."""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.isfile(index_path):
        return HTMLResponse(open(index_path, encoding='utf-8').read())
    return HTMLResponse("<h1>Conversation Triage API</h1><p>POST /predict</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classifica uma conversa: sentimento + tags."""
    result = _classifier.predict(request.conversation)
    return JSONResponse(result)