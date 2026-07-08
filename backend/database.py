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

ALL_SPECIALTIES = ["Cardiología", "Pediatría", "Dermatología", "Medicina General", "Ginecología"]


def match_specialty(text: str) -> Optional[str]:
    """
    Intenta encontrar una especialidad real a partir de texto libre.
    Devuelve el nombre canónico si hay coincidencia, o None si no
    ofrecemos esa especialidad (en vez de asumir Medicina General en silencio).
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    for s in ALL_SPECIALTIES:
        if s.lower() in t or t in s.lower():
            return s
    return None


def sync_get_available_slots(specialty: str) -> list[dict]:
    """
    Devuelve los horarios disponibles (sin ocupar) para una especialidad YA
    VALIDADA (usar match_specialty antes de llamar esta función).
    Lee la tabla schedule_slots y filtra los que ya están reservados
    en la tabla appointments.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                f"{SUPABASE_URL}/rest/v1/schedule_slots",
                headers=_headers(),
                params={
                    "specialty": f"eq.{specialty}",
                    "select": "doctor_name,specialty,date,time",
                    "order": "date.asc,time.asc",
                },
            )
            possible = res.json() if res.status_code == 200 else []
            possible = [
                {"doctor": s["doctor_name"], "date": s["date"], "time": s["time"], "specialty": s["specialty"]}
                for s in possible
            ]

            occ_res = client.get(
                f"{SUPABASE_URL}/rest/v1/appointments",
                headers=_headers(),
                params={
                    "specialty": f"eq.{specialty}",
                    "status": "neq.Cancelada",
                    "select": "doctor,date,time",
                },
            )
            occupied = occ_res.json() if occ_res.status_code == 200 else []
    except Exception:
        return []

    occupied_set = {
        (o.get("doctor", ""), str(o.get("date", "")), str(o.get("time", "")))
        for o in occupied
    }
    return [s for s in possible if (s["doctor"], s["date"], s["time"]) not in occupied_set]


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
    """Devuelve lista de médicos (nombres únicos) que atienden la especialidad dada."""
    slots = sync_get_available_slots(specialty)
    return sorted({s["doctor"] for s in slots})


def sync_get_slots_for_doctor(doctor: str, specialty: str) -> list[dict]:
    """Devuelve horarios disponibles de un médico específico."""
    slots = sync_get_available_slots(specialty)
    return [s for s in slots if doctor.lower() in s["doctor"].lower()]


def nearest_slots(slots: list[dict], date_req: str, time_req: str, n: int = 5) -> list[dict]:
    """
    Ordena una lista de slots por cercanía temporal a la fecha/hora pedida
    por el paciente, y devuelve los `n` más cercanos. Útil para sugerir
    alternativas cuando el horario solicitado no está disponible.
    """
    from datetime import datetime

    try:
        target = datetime.strptime(f"{date_req} {time_req or '09:00'}", "%Y-%m-%d %H:%M")
    except ValueError:
        return slots[:n]

    def _dist(s):
        try:
            dt = datetime.strptime(f"{s['date']} {s['time']}", "%Y-%m-%d %H:%M")
            return abs((dt - target).total_seconds())
        except ValueError:
            return float("inf")

    return sorted(slots, key=_dist)[:n]


# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES ASYNC — para los endpoints FastAPI
# ══════════════════════════════════════════════════════════════════════════

async def get_or_create_patient(first_name: str, last_name: str, dni: str) -> dict:
    """Busca paciente por DNI, lo crea si no existe."""
    dni = (dni or "").strip()
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            params={"dni": f"eq.{dni}", "select": "*"},
        )
        if res.status_code != 200:
            raise Exception(f"Supabase GET /patients falló ({res.status_code}): {res.text}")
        data = res.json()
        if data:
            return data[0]

        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/patients",
            headers=_headers(),
            json={"first_name": first_name, "last_name": last_name, "dni": dni},
        )
        if res.status_code not in (200, 201):
            raise Exception(f"Supabase POST /patients falló ({res.status_code}): {res.text}")
        data = res.json()
        if not data:
            raise Exception(
                "Supabase no devolvió el paciente creado. Revisa que el header "
                "'Prefer: return=representation' funcione y que la policy de INSERT "
                "en 'patients' esté activa."
            )
        return data[0]

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
                "dni": (dni or "").strip(),
                "doctor": doctor,
                "specialty": specialty,
                "date": date,
                "time": time,
                "status": status,
            },
        )
        if res.status_code not in (200, 201):
            # Aquí verás el motivo real: FK violation (dni no existe en patients),
            # CHECK constraint (status inválido), RLS, etc.
            raise Exception(f"Supabase POST /appointments falló ({res.status_code}): {res.text}")
        data = res.json()
        if not data:
            raise Exception(
                "Supabase no devolvió la cita creada. Revisa la policy de INSERT "
                "en 'appointments' y el header 'Prefer: return=representation'."
            )
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