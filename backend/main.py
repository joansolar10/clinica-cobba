"""
Servidor FastAPI — Clínica Cobba
Endpoint principal: POST /chat

Corre localmente con:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()  # Carga GROQ_API_KEY desde .env

# Importar el agente (después de cargar .env para que tenga la key)
from agent import run_agent
from deep_agent import run_deep_agent

app = FastAPI(title="Clínica Cobba — LangGraph API", version="1.0.0")

# ── CORS: permite peticiones desde el frontend React (localhost:5173) ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:4173",   # Vite preview
        "https://clinica-cobba.vercel.app",  # Producción (ajusta si cambia)
        "*",  # Para desarrollo; restringir en producción
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    state: Optional[dict] = None  # Estado anterior enviado por el frontend

class ChatResponse(BaseModel):
    response: str
    state: dict                    # Nuevo estado que el frontend debe guardar
    new_appointment: Optional[dict] = None  # Si hay cita creada

# ── Endpoint principal ────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    current_state = req.state or {"step": "idle", "intent": None, "extracted": {}}

    try:
        result = run_agent(req.message, current_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del agente: {str(e)}")

    return ChatResponse(
        response=result["response"],
        state={
            "step": result["step"],
            "intent": result["intent"],
            "extracted": result["extracted"],
        },
        new_appointment=result.get("new_appointment"),
    )

# ── Deep Agent ───────────────────────────────────────────────────────────
class DeepAgentRequest(BaseModel):
    appointments: list   # lista de citas actuales
    stats: list          # estadísticas por día

class DeepAgentResponse(BaseModel):
    alerts: list         # lista de recomendaciones generadas

@app.post("/deep-agent", response_model=DeepAgentResponse)
async def deep_agent(req: DeepAgentRequest):
    if not req.appointments and not req.stats:
        raise HTTPException(status_code=400, detail="Se requieren datos de citas o estadísticas.")
    try:
        alerts = run_deep_agent(req.appointments, req.stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del Deep Agent: {str(e)}")
    return DeepAgentResponse(alerts=alerts)

# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "modelo": "llama-3.1-8b-instant (Groq)", "version": "1.0.0"}

# ── Para correr directamente con python main.py ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
