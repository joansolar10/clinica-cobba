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
import traceback

load_dotenv()  # Carga GROQ_API_KEY desde .env

# Importar el agente (después de cargar .env para que tenga la key)
from agent import run_agent
from deep_agent import run_deep_agent
from database import (
    get_all_appointments, create_appointment,
    update_appointment_status, get_all_patients, get_or_create_patient
)

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
    refresh_data: Optional[bool] = False    # NUEVO: Flag para refrescar paneles

# ── Endpoint principal ────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    current_state = req.state or {
        "step": "idle", "intent": None,
        "extracted": {}, "conversation_history": []
    }

    # Guardamos el intent anterior para saber qué acción se estaba realizando
    previous_intent = current_state.get("intent")

    try:
        result = run_agent(req.message, current_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del agente: {str(e)}")

    new_step = result.get("step")
    
    # Lógica para avisar al frontend que recargue los paneles de la base de datos
    # Si estábamos cancelando o modificando y el agente terminó (step == "done")
    refresh_data = False
    if previous_intent in ("cancelar", "modificar") and new_step == "done":
        refresh_data = True

    return ChatResponse(
        response=result["response"],
        state={
            "step":                 new_step,
            "intent":               result.get("intent"),
            "extracted":            result.get("extracted", {}),
            "conversation_history": result.get("conversation_history", []),
        },
        new_appointment=result.get("new_appointment"),
        refresh_data=refresh_data,
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

# ── Supabase — Citas ─────────────────────────────────────────────────────
class AppointmentCreate(BaseModel):
    patient_name: str
    dni: str
    doctor: str
    specialty: str
    date: str
    time: str
    status: Optional[str] = "Confirmada"

class AppointmentStatusUpdate(BaseModel):
    status: str

@app.get("/appointments")
async def list_appointments():
    try:
        return await get_all_appointments()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/appointments")
async def add_appointment(req: AppointmentCreate):
    try:
        # Asegurar que el paciente existe en la tabla patients
        names = req.patient_name.strip().split(" ", 1)
        first = names[0]
        last  = names[1] if len(names) > 1 else ""
        await get_or_create_patient(first, last, req.dni)

        appt = await create_appointment(
            req.patient_name, req.dni, req.doctor,
            req.specialty, req.date, req.time, req.status or "Confirmada"
        )
        return appt
    except Exception as e:
        print("── ERROR creando cita ──")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/appointments/{appointment_id}")
async def patch_appointment(appointment_id: str, req: AppointmentStatusUpdate):
    try:
        return await update_appointment_status(appointment_id, req.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Supabase — Pacientes ──────────────────────────────────────────────────
@app.get("/patients")
async def list_patients():
    try:
        return await get_all_patients()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "modelo": "llama-3.1-8b-instant (Groq)", "version": "1.0.0"}

# ── Para correr directamente con python main.py ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)