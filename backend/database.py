"""
database.py — Clínica Cobba
Módulo de acceso a Supabase (citas y pacientes).
Usa httpx para llamadas async a la REST API de Supabase.
Las funciones sync_* son versiones síncronas para uso dentro del agente LangGraph.
"""

import os
import httpx
from typing import Optional
from datetime import date as date_type

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES SÍNCRONAS — para usar dentro del agente LangGraph
# ══════════════════════════════════════════════════════════════════════════

def sync_get_available_slots(specialty: str) -> list[dict]:
    """
    Devuelve los horarios disponibles para una especialidad,
    excluyendo los que ya están ocupados en Supabase.
    Retorna lista de {doctor, date, time, specialty}.
    """
    # Catálogo de horarios posibles por especialidad
    catalog = {
        "Odontología General": [
            {"doctor": "Dr. Quispe",   "date": "2026-07-08", "time": "09:00"},
            {"doctor": "Dr. Quispe",   "date": "2026-07-08", "time": "11:00"},
            {"doctor": "Dr. Quispe",   "date": "2026-07-09", "time": "09:00"},
            {"doctor": "Dra. Vega",    "date": "2026-07-09", "time": "10:30"},
            {"doctor": "Dra. Vega",    "date": "2026-07-10", "time": "08:00"},
            {"doctor": "Dra. Vega",    "date": "2026-07-11", "time": "12:00"},
        ],
        "Ortodoncia": [
            {"doctor": "Dra. Paz",     "date": "2026-07-08", "time": "10:00"},
            {"doctor": "Dra. Paz",     "date": "2026-07-09", "time": "14:00"},
            {"doctor": "Dr. Salinas",  "date": "2026-07-10", "time": "11:00"},
            {"doctor": "Dr. Salinas",  "date": "2026-07-11", "time": "09:00"},
        ],
        "Endodoncia": [
            {"doctor": "Dr. Flores",   "date": "2026-07-08", "time": "08:00"},
            {"doctor": "Dr. Flores",   "date": "2026-07-09", "time": "15:00"},
            {"doctor": "Dr. Flores",   "date": "2026-07-10", "time": "10:00"},
        ],
        "Periodoncia": [
            {"doctor": "Dra. Torres",  "date": "2026-07-08", "time": "16:00"},
            {"doctor": "Dra. Torres",  "date": "2026-07-10", "time": "09:00"},
            {"doctor": "Dra. Torres",  "date": "2026-07-11", "time": "14:00"},
        ],
        "Implantología": [
            {"doctor": "Dr. Mendoza",  "date": "2026-07-09", "time": "08:00"},
            {"doctor": "Dr. Mendoza",  "date": "2026-07-11", "time": "10:00"},
        ],
        "Odontopediatría": [
            {"doctor": "Dra. Ríos",    "date": "2026-07-08", "time": "09:00"},
            {"doctor": "Dra. Ríos",    "date": "2026-07-09", "time": "11:00"},
            {"doctor": "Dra. Ríos",    "date": "2026-07-10", "time": "16:00"},
        ],
        "Cirugía Oral": [
            {"doctor": "Dr. Silva",    "date": "2026-07-09", "time": "09:00"},
            {"doctor": "Dr. Silva",    "date": "2026-07-10", "time": "14:00"},
            {"doctor": "Dr. Silva",    "date": "2026-07-11", "time": "11:00"},
        ],
    }

    # Buscar la especialidad (case-insensitive, coincidencia parcial)
    specialty_lower = specialty.lower()
    matched_key = next(
        (k for k in catalog if k.lower() in specialty_lower or specialty_lower in k.lower()),
        "Odontología General"
    )
    possible = catalog[matched_key]

    # Consultar citas ya ocupadas en Supabase
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                f"{SUPABASE_URL}/rest/v1/appointments",
                headers=_headers(),
                params={
                    "specialty": f"ilike.%{matched_key}%",
                    "status": "neq.Cancelada",
                    "select": "doctor,date,time",
                },
            )
            occupied = res.json() if res.status_code == 200 else []
    except Exception:
        occupied = []

    # Filtrar los que ya están ocupados
    occupied_set = {
        (o.get("doctor", ""), str(o.get("date", "")), str(o.get("time", "")))
        for o in occupied
    }

    available = [
        {**slot, "specialty": matched_key}
        for slot in possible
        if (slot["doctor"], slot["date"], slot["time"]) not in occupied_set
    ]

    return available


def sync_check_slot(doctor: str, date: str, time: str) -> bool:
    """Verifica si un slot específico está disponible (True = libre)."""
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                f"{SUPABASE_URL}/rest/v1/appointments",
                headers=_headers(),
                params={
                    "doctor": f"ilike.%{doctor}%",
                    "date": f"eq.{date}",
                    "time": f"eq.{time}",
                    "status": "neq.Cancelada",
                    "select": "id",
                },
            )
            data = res.json() if res.status_code == 200 else []
            return len(data) == 0   # True si no hay citas = slot libre
    except Exception:
        return True  # En caso de error, asumir disponible


def sync_get_doctors_by_specialty(specialty: str) -> list[str]:
    """Devuelve lista de médicos que atienden la especialidad dada."""
    slots = sync_get_available_slots(specialty)
    doctors = list({s["doctor"] for s in slots})
    return doctors


def sync_get_slots_for_doctor(doctor: str, specialty: str) -> list[dict]:
    """Devuelve horarios disponibles de un médico específico."""
    slots = sync_get_available_slots(specialty)
    return [s for s in slots if doctor.lower() in s["doctor"].lower()]


# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES ASYNC — para los endpoints FastAPI
# ══════════════════════════════════════════════════════════════════════════

async def get_or_create_patient(first_name: str, last_name: str, dni: str) -> dict:
    """Busca paciente por DNI, lo crea si no existe."""
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            params={"dni": f"eq.{dni}", "select": "*"},
        )
        data = res.json()
        if data:
            return data[0]

        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            json={"first_name": first_name, "last_name": last_name, "dni": dni},
        )
        return res.json()[0]

async def get_all_patients() -> list:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            params={"select": "*", "order": "created_at.desc"},
        )
        return res.json()

async def get_all_appointments() -> list:
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/appointments",
            headers=_headers(),
            params={"select": "*", "order": "date.asc,time.asc"},
        )
        return res.json()

async def create_appointment(
    patient_name: str,
    dni: str,
    doctor: str,
    specialty: str,
    date: str,
    time: str,
    status: str = "Confirmada",
) -> dict:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/appointments",
            headers=_headers(),
            json={
                "patient_name": patient_name,
                "dni": dni,
                "doctor": doctor,
                "specialty": specialty,
                "date": date,
                "time": time,
                "status": status,
            },
        )
        data = res.json()
        return data[0] if isinstance(data, list) else data

async def update_appointment_status(appointment_id: str, status: str) -> dict:
    async with httpx.AsyncClient() as client:
        res = await client.patch(
            f"{SUPABASE_URL}/rest/v1/appointments",
            headers=_headers(),
            params={"id": f"eq.{appointment_id}"},
            json={"status": status},
        )
        data = res.json()
        return data[0] if isinstance(data, list) and data else {}