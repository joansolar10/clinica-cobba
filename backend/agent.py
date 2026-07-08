"""
Agente LangGraph para Clínica Cobba — v3 (flujo especialidad → doctor → horario)
Grafo: Router → Scheduler / Escalation / Fallback

Cambios respecto a v2:
- Flujo de agendamiento en 3 pasos reales: primero especialidad, luego
  doctor, luego horario (antes saltaba directo a "Medicina General")
- Validación explícita de especialidad: si el paciente pide una que no
  existe, se le dice claramente y se le listan las que sí hay (antes
  caía en silencio a Medicina General)
- Cuando un horario pedido no está disponible, se calculan (con fechas
  reales, no con el LLM) las 5 alternativas más cercanas en el tiempo,
  cruzando todos los doctores de la especialidad
- El clasificador de intención ahora recibe los últimos mensajes de la
  conversación como contexto, para no reinterpretar un "no, gracias"
  como un nuevo pedido de cita
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
)

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

# ── Nodo 1: ROUTER ─────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    if state["step"] not in ("idle", "done"):
        return state

    history = state.get("conversation_history", [])
    prompt = f"""Eres un clasificador de intenciones para el chat de una clínica médica peruana.
Ten en cuenta la conversación reciente para no malinterpretar respuestas cortas
(por ejemplo "no, gracias" o "eso es todo" después de que el asistente preguntó
"¿algo más en lo que pueda ayudarte?" NO es un pedido de agendar cita, es un cierre).

Conversación reciente:
{_history_text(history)}

Nuevo mensaje del paciente: "{state['user_input']}"

Responde SOLO con una palabra (sin explicación), la intención del NUEVO mensaje:
agendar | cancelar | consultar | humano | desconocido
"""
    result = llm.invoke([SystemMessage(content=prompt)])
    intent = result.content.strip().lower().split()[0]
    valid = {"agendar", "cancelar", "consultar", "humano"}
    intent = intent if intent in valid else "desconocido"

    new_step = state["step"]
    if intent == "agendar":
        new_step = "specialty_prompt"
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

    # ── PASO 0: Mostrar menú de especialidades (no interpreta el mensaje
    #            que disparó "agendar", solo presenta el menú) ─────────────
    if step == "specialty_prompt":
        opciones = "\n".join(f"{i}. {s}" for i, s in enumerate(ALL_SPECIALTIES, 1))
        response = (
            f"¡Con gusto! ¿Para qué especialidad necesitas la cita?\n\n{opciones}\n\n"
            f"Dime el nombre o el número."
        )
        return _reply(response, "specialty_pick")

    # ── PASO 1: Extraer y validar la especialidad elegida ───────────────────
    if step == "specialty_pick":
        # Si respondió con un número, mapearlo directo
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
                    # No se entendió — volver a preguntar
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

    # ── PASO 3: Elegir o negociar horario ───────────────────────────────────
    if step == "slot_pick":
        slots = extracted.get("available_slots", [])
        specialty = extracted.get("specialty", "")
        doctor_filter = extracted.get("doctor_filter")
        slots_text = _slots_to_text(slots, show_doctor=(doctor_filter is None))

        prompt = f"""Eres el asistente de agenda de Clínica Cobba.
El paciente está eligiendo un horario para {specialty}.

Horarios disponibles en la BD:
{slots_text}

Mensaje del paciente: "{user_input}"

Analiza el mensaje y responde SOLO con JSON:
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

            # Pidió un doctor específico distinto al filtro actual
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

            # Pidió fecha y hora específica
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

                # No disponible: calcular las 5 alternativas más cercanas de
                # TODA la especialidad (cruzando doctores), no solo del filtro actual
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
                extracted["doctor_filter"] = None  # las alternativas cruzan doctores
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

        # Fallback — mostrar opciones de nuevo
        response = (
            f"No logré identificar tu elección. Los horarios disponibles son:\n\n"
            f"{slots_text}\n\nElige un número o dime el día y hora que prefieres."
        )
        return _reply(response, "slot_pick")

    # ── PASO 4-6: Recolección de datos del paciente ────────────────────────
    if step == "asking_first_name":
        name = user_input.strip().title()
        extracted["firstName"] = name
        response = f"Gracias, {name}. ¿Cuáles son tus **apellidos**?"
        return _reply(response, "asking_last_name")

    if step == "asking_last_name":
        extracted["lastName"] = user_input.strip().title()
        response = "Perfecto. Por último, ingresa tu **número de DNI** (8 dígitos):"
        return _reply(response, "asking_dni")

    if step == "asking_dni":
        dni = re.sub(r"[^0-9]", "", user_input)
        if len(dni) < 8:
            response = "❌ El DNI parece inválido (mínimo 8 dígitos). Inténtalo de nuevo:"
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
        prompt = f"""El usuario responde a una confirmación de cita médica.
¿Confirma o cancela? Responde SOLO: si | no
Mensaje: "{user_input}"
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

# ── Nodo 4: FALLBACK ────────────────────────────────────────────────────────
def fallback_node(state: AgentState) -> AgentState:
    history = state.get("conversation_history", [])
    history_text = _history_text(history)

    prompt = f"""Eres el asistente virtual de la Clínica Cobba, una clínica médica peruana.
Responde de forma amable y concisa en español.
Si el usuario quiere agendar una cita, dile que escriba "quiero agendar una cita".
Si quiere hablar con un humano, dile que escriba "hablar con recepción".
NO propongas agendar una cita a menos que el paciente lo haya pedido explícitamente
en su nuevo mensaje.

Historial reciente:
{history_text}

Nuevo mensaje: "{state['user_input']}"
"""
    response = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    history.append({"role": "user", "content": state["user_input"]})
    history.append({"role": "assistant", "content": response})
    return {**state, "response": response, "step": "idle",
            "conversation_history": history}

# ── Enrutamiento condicional ────────────────────────────────────────────────
SCHEDULER_STEPS = (
    "specialty_prompt", "specialty_pick", "doctor_pick", "slot_pick",
    "asking_first_name", "asking_last_name", "asking_dni", "ready_to_schedule",
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
    if state["step"] == "specialty_prompt":
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