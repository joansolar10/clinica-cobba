"""
database.py — Clínica Cobba
Módulo de acceso a Supabase (citas y pacientes).
Usa httpx para llamadas async a la REST API de Supabase.
"""

import os
import httpx
from typing import Optional

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

# ── PACIENTES ─────────────────────────────────────────────────────────────

async def get_or_create_patient(first_name: str, last_name: str, dni: str) -> dict:
    """Busca paciente por DNI, lo crea si no existe."""
    async with httpx.AsyncClient() as client:
        # Buscar
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            params={"dni": f"eq.{dni}", "select": "*"},
        )
        data = res.json()
        if data:
            return data[0]

        # Crear
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

# ── CITAS ─────────────────────────────────────────────────────────────────

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
