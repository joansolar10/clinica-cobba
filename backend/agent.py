"""
Agente LangGraph para Clínica Cobba — v4
Grafo: Router → Scheduler / Escalation / Fallback

Cambios:
- Se añadió flujo para consultar, cancelar y modificar citas usando DNI.
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import os, json, re
from database import (
    sync_get_available_slots,
    sync_check_slot,
    sync_get_doctors_by_specialty,
    sync_get_slots_for_doctor,
    match_specialty,
    nearest_slots,
    ALL_SPECIALTIES,
    sync_get_appointments_by_dni,
    sync_update_appointment
)
from knowledge_base import retrieve_context, format_context_for_prompt

# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"],
)

# ── Estado ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    step: str
    extracted: dict
    response: str
    new_appointment: Optional[dict]
    conversation_history: list   # historial para que el LLM tenga contexto

# ── Helpers ────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    """Limpia y parsea JSON del LLM de forma segura."""
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        return {}

def _slots_to_text(slots: list[dict], show_doctor: bool = True) -> str:
    """Convierte lista de slots a texto legible para el LLM/paciente."""
    if not slots:
        return "No hay horarios disponibles."
    lines = []
    for i, s in enumerate(slots, 1):
        if show_doctor:
            lines.append(f"{i}. {s['doctor']} — {s['date']} a las {s['time']}")
        else:
            lines.append(f"{i}. {s['date']} a las {s['time']}")
    return "\n".join(lines)

def _history_text(history: list, n: int = 6) -> str:
    """Últimos n mensajes de la conversación, en texto plano para dar contexto al LLM."""
    recent = history[-n:]
    if not recent:
        return "(sin conversación previa)"
    return "\n".join(
        f"{'Paciente' if m['role']=='user' else 'Asistente'}: {m['content']}"
        for m in recent
    )

def _try_parse_plain_number(user_input: str, n_options: int):
    """
    Si el paciente escribió solo un número (ej. "3", "el 3", "opción 3."),
    lo interpretamos directo con código, SIN pasar por el LLM.

    Por qué: dejamos que el LLM interprete el número antes, y para casos
    fuera de rango (ej. elegir "12" habiendo solo 10 opciones) a veces no
    reportaba fielmente ese valor — "adivinaba" un índice válido en vez de
    decir honestamente que no existía. Para el caso simple y más común
    (un número puro) no hace falta el LLM: se valida con código.

    Devuelve (es_numero_simple, indice_0_based):
      - (False, None)  -> el mensaje no es un número simple, seguir con el LLM
      - (True, idx)     -> número válido, idx es el índice 0-based
      - (True, -1)      -> es un número simple pero está fuera de rango
    """
    m = re.match(r"^\s*(?:opci[oó]n\s*)?#?\s*(\d+)\s*[\.\)]?\s*$", (user_input or "").strip().lower())
    if not m:
        return False, None
    idx = int(m.group(1)) - 1
    if 0 <= idx < n_options:
        return True, idx
    return True, -1

_NAME_RE = re.compile(r"^[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+(?:[\s'-][A-Za-zÁÉÍÓÚÑÜáéíóúñü]+)*$")

def _is_valid_name(text: str) -> bool:
    """
    Valida que un nombre/apellido contenga solo letras (incluye tildes, ñ),
    espacios, apóstrofes o guiones para nombres compuestos (ej. "María José",
    "O'Brien", "Pérez-Gómez"). Rechaza números y símbolos.
    """
    text = (text or "").strip()
    if not text or len(text) > 60:
        return False
    return bool(_NAME_RE.match(text))

# ── Nodo 1: ROUTER ─────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    if state["step"] not in ("idle", "done"):
        return state

    history = state.get("conversation_history", [])
    prompt = f"""# Rol
Eres el clasificador de intenciones del chat de Clínica Cobba, una clínica
odontológica peruana.

# Contexto
Ten en cuenta la conversación reciente para no malinterpretar respuestas
cortas o ambiguas (ej. "sí", "el segundo", "mañana a las 3") que solo tienen
sentido a la luz del turno anterior.

Conversación reciente:
{_history_text(history)}

Nuevo mensaje del paciente: "{state['user_input']}"

# Instrucciones
Clasifica ÚNICAMENTE el NUEVO mensaje en una de estas categorías:
- agendar: quiere programar una cita nueva.
- cancelar: quiere anular UNA CITA PROPIA que ya tiene agendada.
- consultar: quiere ver el estado o detalle de UNA CITA PROPIA que ya
  tiene agendada (no información general de la clínica).
- modificar: quiere reprogramar/cambiar UNA CITA PROPIA que ya tiene.
- humano: pide hablar con una persona, tiene una queja o una emergencia.
- desconocido: cualquier otro caso, incluyendo saludos, y preguntas
  GENERALES sobre la clínica (horarios de atención, ubicación, precios,
  seguros/convenios, qué llevar a la primera cita, especialidades
  disponibles, políticas). Estas NO son "consultar" porque no se refieren
  a una cita que el paciente ya tiene.

# Distinción clave (la fuente más común de error)
"consultar/cancelar/modificar" requieren que el paciente esté hablando de
UNA CITA PROPIA YA EXISTENTE. Si la pregunta es información general sobre
la clínica y no sobre una cita específica del paciente, es "desconocido",
aunque use palabras como "cita", "horario" o "disponible".

Pista práctica: si el mensaje empieza con "qué debo", "qué necesito",
"cómo funciona", "cuánto cuesta", "a qué hora", o pregunta por un
REQUISITO o POLÍTICA en general (no por el estado de una cita que el
paciente ya tiene), es "desconocido" — aunque mencione la palabra "cita".

Ejemplos:
- "¿A qué hora abren los sábados?" → desconocido (horario general, no una cita propia)
- "¿Tienen atención los domingos?" → desconocido
- "¿Aceptan mi seguro Rimac?" → desconocido
- "¿Cuánto cuesta una limpieza?" → desconocido
- "¿Qué debo llevar a mi primera cita?" → desconocido (información general de requisitos, no pregunta por SU cita)
- "¿Cuándo es mi próxima cita?" → consultar (es SU cita)
- "Quiero ver mis citas" → consultar
- "Ya no puedo ir a mi cita del jueves" → cancelar
- "Quiero cambiar mi cita para otro día" → modificar
- "Quiero una cita con el ortodoncista" → agendar
- "me duele mucho una muela, es urgente" → humano (emergencia, no es información general)
- "tengo una queja sobre mi última visita" → humano
- "quiero hablar con alguien de recepción" → humano

# Manejo de errores
Si el mensaje es ambiguo o no tienes suficiente certeza, clasifica como
"desconocido" en vez de adivinar una acción sensible (agendar/cancelar/
modificar).

# Formato de salida
Responde SOLO con una palabra exacta de la lista anterior, en minúsculas,
sin puntuación, sin comillas y sin explicación.
"""
    result = llm.invoke([SystemMessage(content=prompt)])
    raw = (result.content or "").strip().lower()
    intent = raw.split()[0] if raw else "desconocido"
    valid = {"agendar", "cancelar", "consultar", "modificar", "humano"}
    intent = intent if intent in valid else "desconocido"

    new_step = state["step"]
    if intent == "agendar":
        new_step = "specialty_prompt"
    elif intent in ("consultar", "cancelar", "modificar"):
        new_step = "asking_dni_for_action"
    elif intent == "humano":
        new_step = "handoff"

    return {**state, "intent": intent, "step": new_step}

# ── Nodo 2: SCHEDULER — cerebro conversacional con contexto de BD ──────────
def scheduler_node(state: AgentState) -> AgentState:
    step       = state["step"]
    user_input = state["user_input"]
    extracted  = dict(state["extracted"])
    history    = list(state.get("conversation_history", []))

    # Agregar mensaje del usuario al historial
    history.append({"role": "user", "content": user_input})

    def _reply(text: str, new_step: str, **extra_state):
        history.append({"role": "assistant", "content": text})
        return {**state, "extracted": extracted, "step": new_step,
                "response": text, "conversation_history": history, **extra_state}

    # ── PASOS PARA CONSULTAR / CANCELAR / MODIFICAR (NUEVO) ────────────────
    if step == "asking_dni_for_action":
        response = "Para buscar tus citas, por favor ingresa tu **número de DNI** (8 dígitos):"
        return _reply(response, "processing_dni_for_action")

    if step == "processing_dni_for_action":
        dni = re.sub(r"[^0-9]", "", user_input)
        if len(dni) != 8:
            response = "❌ El DNI debe tener exactamente 8 dígitos. Por favor, ingrésalo de nuevo:"
            return _reply(response, "processing_dni_for_action")
        
        citas = sync_get_appointments_by_dni(dni)
        citas_activas = [c for c in citas if c["status"] not in ("Cancelada", "No-Show")]
        
        if not citas_activas:
            response = f"No encontré citas pendientes asociadas al DNI {dni}. ¿En qué más te puedo ayudar?"
            return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}

        extracted["citas_activas"] = citas_activas
        texto_citas = "\n".join(
            f"{i+1}. {c['specialty']} con {c['doctor']} - {c['date']} a las {c['time']}"
            for i, c in enumerate(citas_activas)
        )
        
        if state["intent"] == "consultar":
            response = f"Aquí tienes tus citas activas:\n\n{texto_citas}\n\n¿Hay algo más que pueda hacer por ti?"
            return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}
        
        elif state["intent"] == "cancelar":
            response = f"Encontré estas citas:\n\n{texto_citas}\n\nPor favor, dime el **número** de la cita que deseas cancelar."
            return _reply(response, "select_appointment_for_cancel")
            
        elif state["intent"] == "modificar":
            response = f"Encontré estas citas:\n\n{texto_citas}\n\nPor favor, dime el **número** de la cita que deseas reprogramar."
            return _reply(response, "select_appointment_for_modify")

    if step == "select_appointment_for_cancel":
        citas_activas = extracted.get("citas_activas", [])
        m = re.search(r"\d+", user_input)
        if m:
            idx = int(m.group(0)) - 1
            if 0 <= idx < len(citas_activas):
                cita_elegida = citas_activas[idx]
                sync_update_appointment(cita_elegida["id"], status="Cancelada")
                response = f"✅ Tu cita de {cita_elegida['specialty']} ha sido cancelada exitosamente.\n\n¿Puedo ayudarte con algo más?"
                return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}
                
        response = "No entendí a qué cita te refieres. Por favor responde con el **número** de la cita que deseas cancelar."
        return _reply(response, "select_appointment_for_cancel")

    if step == "select_appointment_for_modify":
        citas_activas = extracted.get("citas_activas", [])
        m = re.search(r"\d+", user_input)
        if m:
            idx = int(m.group(0)) - 1
            if 0 <= idx < len(citas_activas):
                cita_elegida = citas_activas[idx]
                extracted["cita_a_modificar"] = cita_elegida
                extracted["specialty"] = cita_elegida["specialty"]
                
                slots = sync_get_available_slots(cita_elegida["specialty"])
                extracted["available_slots"] = slots
                
                if not slots:
                    response = f"Lo siento, no hay otros horarios disponibles para {cita_elegida['specialty']} en este momento."
                    return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}

                slots_text = _slots_to_text(slots, show_doctor=True)
                response = (f"Vamos a reprogramar tu cita de {cita_elegida['specialty']} (actualmente {cita_elegida['date']} {cita_elegida['time']}).\n\n"
                            f"Estos son los nuevos horarios disponibles:\n\n{slots_text}\n\n"
                            f"¿Qué número de horario prefieres? O dime una fecha/hora distinta.")
                return _reply(response, "slot_pick_for_modify")
                
        response = "No entendí. Por favor responde con el **número** de la cita que deseas reprogramar."
        return _reply(response, "select_appointment_for_modify")

    if step == "slot_pick_for_modify":
        slots = extracted.get("available_slots", [])
        specialty = extracted.get("specialty", "")
        cita_a_modificar = extracted.get("cita_a_modificar")
        slots_text = _slots_to_text(slots, show_doctor=True)

        # Atajo determinístico: si el paciente escribió solo un número,
        # lo resolvemos directo con código (ver _try_parse_plain_number),
        # sin pasar por el LLM.
        is_plain_number, idx = _try_parse_plain_number(user_input, len(slots))
        if is_plain_number:
            if idx is not None and idx >= 0:
                slot = slots[idx]
                sync_update_appointment(cita_a_modificar["id"], doctor=slot["doctor"], date=slot["date"], time=slot["time"])
                response = (
                    f"✅ ¡Perfecto! Tu cita ha sido reprogramada con **{slot['doctor']}** "
                    f"para el **{slot['date']}** a las **{slot['time']}**.\n\n"
                    f"¿Hay algo más en lo que pueda ayudarte?"
                )
                return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}
            else:
                response = (
                    f"Ese número no está en la lista — tengo {len(slots)} horario"
                    f"{'s' if len(slots) != 1 else ''} disponible{'s' if len(slots) != 1 else ''} "
                    f"(del 1 al {len(slots)}). Por favor elige un número válido:\n\n{slots_text}"
                )
                return _reply(response, "slot_pick_for_modify")

        prompt = f"""# Rol
Eres el asistente de agenda de Clínica Cobba.

# Tarea
El paciente está eligiendo un NUEVO horario para reprogramar su cita de {specialty}.

Horarios disponibles en la BD (única fuente de horarios válida):
{slots_text}

Mensaje del paciente: "{user_input}"

# Restricciones
No inventes doctores, fechas ni horas que no estén en la lista anterior o
que el paciente no haya mencionado explícitamente.

# Manejo de errores
Si el mensaje es ambiguo y no puedes determinar la elección con certeza,
usa "pedir_mas_opciones" en vez de adivinar un índice.

# Formato de salida
Responde SOLO con este JSON (sin texto adicional, sin markdown):
{{
  "accion": "elegir_disponible" | "pedir_alternativo" | "pedir_mas_opciones" | "cancelar",
  "indice": <número del slot elegido (1-based), o null>,
  "doctor_solicitado": "<nombre del doctor si lo mencionó, o null>",
  "fecha_solicitada": "<fecha en formato YYYY-MM-DD si la mencionó, o null>",
  "hora_solicitada": "<hora en formato HH:MM si la mencionó, o null>"
}}
"""
        parsed = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
        accion = parsed.get("accion", "pedir_mas_opciones")

        if accion == "elegir_disponible" and parsed.get("indice"):
            idx = int(parsed["indice"]) - 1
            if 0 <= idx < len(slots):
                slot = slots[idx]
                sync_update_appointment(cita_a_modificar["id"], doctor=slot["doctor"], date=slot["date"], time=slot["time"])
                response = (
                    f"✅ ¡Perfecto! Tu cita ha sido reprogramada con **{slot['doctor']}** "
                    f"para el **{slot['date']}** a las **{slot['time']}**.\n\n"
                    f"¿Hay algo más en lo que pueda ayudarte?"
                )
                return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}

        if accion == "pedir_alternativo":
            doc_req  = parsed.get("doctor_solicitado") or ""
            date_req = parsed.get("fecha_solicitada") or ""
            time_req = parsed.get("hora_solicitada") or ""

            if doc_req and not date_req:
                doc_slots = sync_get_slots_for_doctor(doc_req, specialty)
                if doc_slots:
                    extracted["available_slots"] = doc_slots
                    response = f"Para **{doc_req}** tengo estos horarios:\n\n{_slots_to_text(doc_slots, show_doctor=False)}\n\n¿Cuál te conviene?"
                    return _reply(response, "slot_pick_for_modify")
                else:
                    response = f"Lo siento, **{doc_req}** no tiene horarios libres. ¿Te gustaría ver otras opciones?\n\n{slots_text}"
                    return _reply(response, "slot_pick_for_modify")

            if date_req and time_req:
                exact_match = next((s for s in slots if s["date"] == date_req and s["time"] == time_req), None)
                is_free = exact_match and sync_check_slot(exact_match["doctor"], date_req, time_req)
                if exact_match and is_free:
                    sync_update_appointment(cita_a_modificar["id"], doctor=exact_match["doctor"], date=date_req, time=time_req)
                    response = f"✅ ¡Perfecto! Tu cita ha sido reprogramada con **{exact_match['doctor']}** para el **{date_req}** a las **{time_req}**.\n\n¿Algo más?"
                    return {**state, "step": "done", "extracted": {}, "intent": None, "response": response, "conversation_history": history}

                all_specialty_slots = sync_get_available_slots(specialty)
                alternatives = nearest_slots(all_specialty_slots, date_req, time_req, n=5)
                if not alternatives:
                    response = f"Lo siento, no hay horarios cercanos a {date_req} {time_req}. ¿Deseas elegir de la lista?\n\n{slots_text}"
                    return _reply(response, "slot_pick_for_modify")

                extracted["available_slots"] = alternatives
                response = f"Ese horario no está disponible. Las opciones más cercanas son:\n\n{_slots_to_text(alternatives, show_doctor=True)}\n\n¿Cuál prefieres?"
                return _reply(response, "slot_pick_for_modify")

        if accion == "cancelar":
            return {**state, "step": "done", "extracted": {}, "response": "Entendido, no modifiqué tu cita. ¿En qué más te ayudo?", "conversation_history": history}

        response = f"No logré identificar tu elección. Los horarios disponibles son:\n\n{slots_text}\n\nElige un número o dime fecha y hora."
        return _reply(response, "slot_pick_for_modify")

    # ── PASO 0: Mostrar menú de especialidades ────────────────────────────────
    if step == "specialty_prompt":
        opciones = "\n".join(f"{i}. {s}" for i, s in enumerate(ALL_SPECIALTIES, 1))
        response = (
            f"¡Con gusto! ¿Para qué especialidad necesitas la cita?\n\n{opciones}\n\n"
            f"Dime el nombre o el número."
        )
        return _reply(response, "specialty_pick")

    # ── PASO 1: Extraer y validar la especialidad elegida ───────────────────
    if step == "specialty_pick":
        specialty = None
        m = re.match(r"^\s*(\d+)\s*$", user_input)
        if m and 1 <= int(m.group(1)) <= len(ALL_SPECIALTIES):
            specialty = ALL_SPECIALTIES[int(m.group(1)) - 1]
        else:
            specialty = match_specialty(user_input)

        if not specialty:
            opciones = "\n".join(f"{i}. {s}" for i, s in enumerate(ALL_SPECIALTIES, 1))
            response = (
                f"No tenemos esa especialidad disponible. Estas son las que "
                f"ofrecemos actualmente:\n\n{opciones}\n\nDime el nombre o el número."
            )
            return _reply(response, "specialty_pick")

        extracted["specialty"] = specialty
        doctors = sync_get_doctors_by_specialty(specialty)

        if not doctors:
            response = (
                f"Lo siento, en este momento no tenemos horarios disponibles para "
                f"**{specialty}**. ¿Te gustaría elegir otra especialidad?"
            )
            return _reply(response, "specialty_prompt")

        extracted["doctors_list"] = doctors
        opciones = "\n".join(f"{i}. {d}" for i, d in enumerate(doctors, 1))
        response = (
            f"Perfecto, **{specialty}**. ¿Con qué doctor prefieres tu cita?\n\n"
            f"{opciones}\n\nDime el nombre, el número, o escribe **\"cualquiera\"** "
            f"si no tienes preferencia."
        )
        return _reply(response, "doctor_pick")

    # ── PASO 2: Elegir doctor (o "cualquiera") ──────────────────────────────
    if step == "doctor_pick":
        doctors = extracted.get("doctors_list", [])
        specialty = extracted.get("specialty", "")
        chosen_doctor = None

        m = re.match(r"^\s*(\d+)\s*$", user_input)
        if m and 1 <= int(m.group(1)) <= len(doctors):
            chosen_doctor = doctors[int(m.group(1)) - 1]
        else:
            low = user_input.strip().lower()
            if low in ("cualquiera", "cualquier doctor", "no tengo preferencia", "me da igual", "el que sea"):
                chosen_doctor = None
            else:
                for d in doctors:
                    if d.lower() in low or low in d.lower():
                        chosen_doctor = d
                        break
                else:
                    opciones = "\n".join(f"{i}. {d}" for i, d in enumerate(doctors, 1))
                    response = (
                        f"No identifiqué ese doctor. Las opciones para **{specialty}** son:\n\n"
                        f"{opciones}\n\nDime el nombre, el número, o \"cualquiera\"."
                    )
                    return _reply(response, "doctor_pick")

        if chosen_doctor:
            slots = sync_get_slots_for_doctor(chosen_doctor, specialty)
        else:
            slots = sync_get_available_slots(specialty)

        if not slots:
            quien = chosen_doctor or "esa especialidad"
            response = (
                f"Lo siento, no hay horarios libres con {quien} en este momento. "
                f"¿Quieres intentar con otro doctor o especialidad?"
            )
            return _reply(response, "doctor_pick")

        extracted["doctor_filter"] = chosen_doctor
        extracted["available_slots"] = slots
        slots_text = _slots_to_text(slots, show_doctor=(chosen_doctor is None))
        response = (
            f"Estos son los horarios disponibles"
            f"{f' con {chosen_doctor}' if chosen_doctor else ''} para **{specialty}**:\n\n"
            f"{slots_text}\n\n¿Cuál prefieres? Puedes elegir un número o pedirme otro "
            f"día/hora y verifico si está disponible."
        )
        return _reply(response, "slot_pick")

    # ── PASO 3: Elegir o negociar horario (AGENDAR NUEVA CITA) ──────────────
    if step == "slot_pick":
        slots = extracted.get("available_slots", [])
        specialty = extracted.get("specialty", "")
        doctor_filter = extracted.get("doctor_filter")
        slots_text = _slots_to_text(slots, show_doctor=(doctor_filter is None))

        # Atajo determinístico: si el paciente escribió solo un número,
        # lo resolvemos directo con código, sin pasar por el LLM.
        is_plain_number, idx = _try_parse_plain_number(user_input, len(slots))
        if is_plain_number:
            if idx is not None and idx >= 0:
                slot = slots[idx]
                extracted.update({"doctor": slot["doctor"], "date": slot["date"], "time": slot["time"]})
                response = (
                    f"¡Perfecto! Reservé el espacio con **{slot['doctor']}** "
                    f"para el **{slot['date']}** a las **{slot['time']}**.\n\n"
                    f"Para registrarte, ¿cuál es tu **primer nombre**?"
                )
                return _reply(response, "asking_first_name")
            else:
                response = (
                    f"Ese número no está en la lista — tengo {len(slots)} horario"
                    f"{'s' if len(slots) != 1 else ''} disponible{'s' if len(slots) != 1 else ''} "
                    f"(del 1 al {len(slots)}). Por favor elige un número válido:\n\n{slots_text}"
                )
                return _reply(response, "slot_pick")

        prompt = f"""# Rol
Eres el asistente de agenda de Clínica Cobba.

# Tarea
El paciente está eligiendo un horario para {specialty}.

Horarios disponibles en la BD (única fuente de horarios válida):
{slots_text}

Mensaje del paciente: "{user_input}"

# Restricciones
No inventes doctores, fechas ni horas que no estén en la lista anterior o
que el paciente no haya mencionado explícitamente.

# Manejo de errores
Si el mensaje es ambiguo y no puedes determinar la elección con certeza,
usa "pedir_mas_opciones" en vez de adivinar un índice.

# Formato de salida
Responde SOLO con este JSON (sin texto adicional, sin markdown):
{{
  "accion": "elegir_disponible" | "pedir_alternativo" | "pedir_mas_opciones" | "cancelar",
  "indice": <número del slot elegido (1-based), o null>,
  "doctor_solicitado": "<nombre del doctor si lo mencionó, o null>",
  "fecha_solicitada": "<fecha en formato YYYY-MM-DD si la mencionó, o null>",
  "hora_solicitada": "<hora en formato HH:MM si la mencionó, o null>"
}}
"""
        parsed = _parse_json(llm.invoke([SystemMessage(content=prompt)]).content)
        accion = parsed.get("accion", "pedir_mas_opciones")

        if accion == "elegir_disponible" and parsed.get("indice"):
            idx = int(parsed["indice"]) - 1
            if 0 <= idx < len(slots):
                slot = slots[idx]
                extracted.update({"doctor": slot["doctor"], "date": slot["date"], "time": slot["time"]})
                response = (
                    f"¡Perfecto! Reservé el espacio con **{slot['doctor']}** "
                    f"para el **{slot['date']}** a las **{slot['time']}**.\n\n"
                    f"Para registrarte, ¿cuál es tu **primer nombre**?"
                )
                return _reply(response, "asking_first_name")

        if accion == "pedir_alternativo":
            doc_req  = parsed.get("doctor_solicitado") or ""
            date_req = parsed.get("fecha_solicitada") or ""
            time_req = parsed.get("hora_solicitada") or ""

            if doc_req and not date_req:
                doc_slots = sync_get_slots_for_doctor(doc_req, specialty)
                if doc_slots:
                    extracted["available_slots"] = doc_slots
                    extracted["doctor_filter"] = doc_req
                    response = (
                        f"Para **{doc_req}** tengo estos horarios disponibles:\n\n"
                        f"{_slots_to_text(doc_slots, show_doctor=False)}\n\n¿Cuál te conviene?"
                    )
                    return _reply(response, "slot_pick")
                else:
                    response = (
                        f"Lo siento, **{doc_req}** no tiene horarios libres en este momento. "
                        f"¿Te gustaría ver opciones con otro médico?\n\n{slots_text}"
                    )
                    return _reply(response, "slot_pick")

            if date_req and time_req:
                exact_match = next(
                    (s for s in slots if s["date"] == date_req and s["time"] == time_req), None
                )
                is_free = exact_match and sync_check_slot(exact_match["doctor"], date_req, time_req)

                if exact_match and is_free:
                    extracted.update({"doctor": exact_match["doctor"], "date": date_req, "time": time_req})
                    response = (
                        f"¡Perfecto! Ese horario está disponible con "
                        f"**{exact_match['doctor']}** el **{date_req}** a las **{time_req}**.\n\n"
                        f"Para registrarte, ¿cuál es tu **primer nombre**?"
                    )
                    return _reply(response, "asking_first_name")

                all_specialty_slots = sync_get_available_slots(specialty)
                alternatives = nearest_slots(all_specialty_slots, date_req, time_req, n=5)

                if not alternatives:
                    response = (
                        f"Lo siento, no encontré horarios disponibles cercanos a "
                        f"{date_req} {time_req} para {specialty}. ¿Quieres intentar "
                        f"con otra especialidad?"
                    )
                    return _reply(response, "specialty_prompt")

                extracted["available_slots"] = alternatives
                extracted["doctor_filter"] = None
                response = (
                    f"Ese horario ({date_req} a las {time_req}) no está disponible. "
                    f"Estas son las 5 opciones más cercanas:\n\n"
                    f"{_slots_to_text(alternatives, show_doctor=True)}\n\n"
                    f"¿Cuál prefieres?"
                )
                return _reply(response, "slot_pick")

        if accion == "cancelar":
            return {**state, "step": "done", "extracted": {},
                    "response": "Entendido, cancelé el proceso. ¿En qué más puedo ayudarte?",
                    "conversation_history": history}

        response = (
            f"No logré identificar tu elección. Los horarios disponibles son:\n\n"
            f"{slots_text}\n\nElige un número o dime el día y hora que prefieres."
        )
        return _reply(response, "slot_pick")

    # ── PASO 4-6: Recolección de datos del paciente ────────────────────────
    if step == "asking_first_name":
        candidate = user_input.strip()
        if not _is_valid_name(candidate):
            response = (
                "❌ El nombre solo debe contener letras (sin números ni símbolos). "
                "Por favor, ingresa tu **primer nombre** de nuevo:"
            )
            return _reply(response, "asking_first_name")
        name = candidate.title()
        extracted["firstName"] = name
        response = f"Gracias, {name}. ¿Cuáles son tus **apellidos**?"
        return _reply(response, "asking_last_name")

    if step == "asking_last_name":
        candidate = user_input.strip()
        if not _is_valid_name(candidate):
            response = (
                "❌ Los apellidos solo deben contener letras (sin números ni símbolos). "
                "Por favor, ingrésalos de nuevo:"
            )
            return _reply(response, "asking_last_name")
        extracted["lastName"] = candidate.title()
        response = "Perfecto. Por último, ingresa tu **número de DNI** (8 dígitos):"
        return _reply(response, "asking_dni")

    if step == "asking_dni":
        dni = re.sub(r"[^0-9]", "", user_input)
        if len(dni) != 8:
            response = "❌ El DNI debe tener exactamente 8 dígitos. Inténtalo de nuevo:"
            return _reply(response, "asking_dni")

        extracted["dni"] = dni
        resumen = (
            f"✅ **Resumen de tu cita:**\n"
            f"- Paciente: {extracted.get('firstName')} {extracted.get('lastName')}\n"
            f"- Especialidad: {extracted.get('specialty')}\n"
            f"- Médico: {extracted.get('doctor')}\n"
            f"- Fecha/Hora: {extracted.get('date')} a las {extracted.get('time')}\n\n"
            f"¿Confirmas? Responde **Sí** o **No**."
        )
        return _reply(resumen, "ready_to_schedule")

    # ── PASO 7: Confirmación final ─────────────────────────────────────────
    if step == "ready_to_schedule":
        prompt = f"""# Rol
Eres el asistente de agenda de Clínica Cobba, interpretando la respuesta a
una confirmación de cita médica.

# Manejo de errores
Si el mensaje es ambiguo o no expresa una confirmación clara, responde "no"
para evitar agendar una cita por error.

# Formato de salida
Responde SOLO con una palabra: si | no

Mensaje del paciente: "{user_input}"
"""
        decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
        confirmed = "si" in decision or "sí" in decision

        if confirmed:
            appointment = {
                "patientName": f"{extracted.get('firstName','')} {extracted.get('lastName','')}".strip(),
                "dni": extracted.get("dni"),
                "doctor": extracted.get("doctor", "Por asignar"),
                "specialty": extracted.get("specialty", "General"),
                "date": extracted.get("date", ""),
                "time": extracted.get("time", ""),
            }
            response = (
                "🎉 **¡Cita confirmada exitosamente!**\n"
                "Te enviaremos un recordatorio por WhatsApp 24 horas antes.\n\n"
                "¿Hay algo más en lo que pueda ayudarte?"
            )
            history.append({"role": "assistant", "content": response})
            return {**state, "step": "done", "new_appointment": appointment,
                    "extracted": {}, "intent": None,
                    "response": response, "conversation_history": history}
        else:
            response = "Entendido, he cancelado el agendamiento. ¿En qué más puedo ayudarte?"
            history.append({"role": "assistant", "content": response})
            return {**state, "step": "done", "extracted": {}, "intent": None,
                    "response": response, "conversation_history": history}

    return state

# ── Nodo 3: ESCALATION ─────────────────────────────────────────────────────
def escalation_node(state: AgentState) -> AgentState:
    response = (
        "He pausado el bot y **notificado a recepción**. "
        "Un agente humano revisará tu historial y tomará el chat en breve. "
        "Por favor, espera en línea. 🧑‍💼"
    )
    return {**state, "step": "handoff", "response": response}

# ── Nodo 4: FALLBACK (con RAG sobre la base de conocimiento de la clínica) ──
def fallback_node(state: AgentState) -> AgentState:
    history = state.get("conversation_history", [])
    history_text = _history_text(history)
    user_input = state["user_input"]

    # RAG: recuperamos los fragmentos de la base de conocimiento (horarios,
    # seguros, precios, políticas, etc.) más relevantes para la pregunta,
    # en vez de dejar que el LLM "recuerde"/invente esos datos.
    relevant_docs = retrieve_context(user_input, k=3)
    kb_context = format_context_for_prompt(relevant_docs)

    prompt = f"""# Rol y contexto
Eres el asistente virtual de Clínica Cobba, una clínica odontológica
peruana. Atiendes pacientes por chat: resuelves dudas administrativas
generales y los derivas al flujo correspondiente cuando corresponde.

# Base de conocimiento recuperada (usar como única fuente de verdad)
{kb_context}

# Instrucciones
1. Responde en español, de forma cálida, profesional y concisa (máx. 3-4
   líneas).
2. Si la base de conocimiento anterior contiene la respuesta, básate
   ÚNICAMENTE en ella.
3. Si el paciente quiere agendar una cita, dile que escriba
   "quiero agendar una cita".
4. Si quiere consultar, modificar o cancelar una cita, dile que te lo pida
   directamente (ej. "quiero cancelar mi cita").
5. Si quiere hablar con una persona, o describe una emergencia, dile que
   escriba "hablar con recepción".

# Manejo de errores
Si la base de conocimiento no cubre lo que el paciente pregunta, dilo con
honestidad ("no tengo ese dato a la mano") y ofrece que recepción se lo
confirme. NO inventes horarios, precios, direcciones ni políticas.

# Restricciones
- NO propongas agendar una cita a menos que el paciente lo haya pedido
  explícitamente.
- NO des consejos médicos ni diagnósticos; solo información administrativa
  de la clínica.
- NO inventes datos que no estén en la base de conocimiento recuperada.
- Si el paciente menciona un nombre propio (una aseguradora, un doctor,
  una marca, etc.) y ese nombre específico NO aparece mencionado en la
  base de conocimiento recuperada, NO confirmes ni niegues que aplica a
  él. Responde de forma genérica con lo que sí dice la base de
  conocimiento y aclara que ese dato puntual debe confirmarlo recepción.
  Ejemplo: si preguntan "¿aceptan mi seguro Rimac?" y la base de
  conocimiento no menciona "Rimac" explícitamente, NO respondas "sí,
  aceptamos Rimac"; responde explicando la política general de reembolso
  y que recepción debe confirmar si esa aseguradora específica aplica.

# Historial reciente
{history_text}

# Nuevo mensaje del paciente
"{user_input}"
"""
    response = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return {**state, "response": response, "step": "idle",
            "conversation_history": history}

# ── Enrutamiento condicional ────────────────────────────────────────────────
SCHEDULER_STEPS = (
    "specialty_prompt", "specialty_pick", "doctor_pick", "slot_pick",
    "asking_first_name", "asking_last_name", "asking_dni", "ready_to_schedule",
    "asking_dni_for_action", "processing_dni_for_action", 
    "select_appointment_for_cancel", "select_appointment_for_modify", "slot_pick_for_modify"
)

def route_main(state: AgentState) -> str:
    step = state["step"]
    if step == "handoff":
        return "escalation"
    if step in SCHEDULER_STEPS:
        return "scheduler"
    return "router"

def route_after_router(state: AgentState) -> str:
    if state["step"] == "handoff":
        return "escalation"
    if state["step"] in ("specialty_prompt", "asking_dni_for_action"):
        return "scheduler"
    return "fallback"

# ── Construcción del grafo ──────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",     router_node)
    g.add_node("scheduler",  scheduler_node)
    g.add_node("escalation", escalation_node)
    g.add_node("fallback",   fallback_node)

    g.set_conditional_entry_point(route_main)

    g.add_conditional_edges("router", route_after_router, {
        "escalation": "escalation",
        "scheduler":  "scheduler",
        "fallback":   "fallback",
    })
    g.add_edge("scheduler",  END)
    g.add_edge("escalation", END)
    g.add_edge("fallback",   END)

    return g.compile()

graph = build_graph()

# ── Función pública ─────────────────────────────────────────────────────────
def run_agent(user_input: str, current_state: dict) -> dict:
    input_state: AgentState = {
        "user_input":            user_input,
        "intent":                current_state.get("intent"),
        "step":                  current_state.get("step", "idle"),
        "extracted":             current_state.get("extracted", {}),
        "response":              "",
        "new_appointment":       None,
        "conversation_history":  current_state.get("conversation_history", []),
    }
    return graph.invoke(input_state)