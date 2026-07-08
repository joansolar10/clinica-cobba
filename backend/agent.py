"""
Agente LangGraph para Clínica Cobba — v2 (con contexto real de BD)
Grafo: Router → Scheduler / Escalation / Fallback

Cambios respecto a v1:
- El agente consulta Supabase ANTES de responder para tener horarios reales
- El LLM razona sobre disponibilidad real, no sobre datos hardcodeados
- Si el paciente pide un horario alternativo, el LLM verifica en la BD
  si ese slot existe y está libre, y responde en consecuencia
- Un único nodo "scheduler" maneja todo el flujo conversacional con
  el LLM como cerebro, en vez de if/else por step
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

def _slots_to_text(slots: list[dict]) -> str:
    """Convierte lista de slots a texto legible para el LLM."""
    if not slots:
        return "No hay horarios disponibles para esta especialidad."
    lines = []
    for i, s in enumerate(slots, 1):
        lines.append(f"{i}. {s['doctor']} — {s['date']} a las {s['time']}")
    return "\n".join(lines)

# ── Nodo 1: ROUTER ─────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    if state["step"] not in ("idle", "done"):
        return state

    prompt = f"""Eres un clasificador de intenciones para una clínica médica peruana.
Responde SOLO con una palabra (sin explicación):
agendar | cancelar | consultar | humano | desconocido

Mensaje: "{state['user_input']}"
"""
    result = llm.invoke([SystemMessage(content=prompt)])
    intent = result.content.strip().lower().split()[0]
    valid = {"agendar", "cancelar", "consultar", "humano"}
    intent = intent if intent in valid else "desconocido"

    new_step = state["step"]
    if intent == "agendar":
        new_step = "asking_specialty"
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

    # ── PASO 1: Detectar especialidad ──────────────────────────────────────
    if step == "asking_specialty":
        prompt = f"""Extrae la especialidad médica del mensaje. 
Si no se menciona ninguna responde exactamente: Medicina General
Responde SOLO con el nombre de la especialidad.
Mensaje: "{user_input}"
"""
        specialty = llm.invoke([SystemMessage(content=prompt)]).content.strip()
        extracted["specialty"] = specialty

        # Consultar BD real
        slots = sync_get_available_slots(specialty)
        slots_text = _slots_to_text(slots)
        extracted["available_slots"] = slots  # guardar para uso posterior

        if not slots:
            response = (
                f"Lo siento, actualmente no tenemos horarios disponibles para "
                f"**{specialty}**. ¿Te puedo ayudar con otra especialidad o "
                f"prefiere hablar con recepción?"
            )
            history.append({"role": "assistant", "content": response})
            return {**state, "extracted": extracted, "step": "asking_specialty",
                    "response": response, "conversation_history": history}

        response = (
            f"Consulté nuestra agenda para **{specialty}** y estos son los "
            f"horarios disponibles:\n\n{slots_text}\n\n"
            f"¿Cuál prefieres? Puedes elegir un número o pedirme otro día/hora "
            f"y verifico si está disponible."
        )
        history.append({"role": "assistant", "content": response})
        return {**state, "extracted": extracted, "step": "choosing_slot",
                "response": response, "conversation_history": history}

    # ── PASO 2: Elegir o negociar horario ─────────────────────────────────
    if step == "choosing_slot":
        slots = extracted.get("available_slots", [])
        slots_text = _slots_to_text(slots)

        # El LLM interpreta la respuesta del usuario y decide qué hacer
        prompt = f"""Eres el asistente de agenda de Clínica Cobba.
El paciente está eligiendo un horario para {extracted.get('specialty', 'su cita')}.

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
                extracted.update({
                    "doctor": slot["doctor"],
                    "date": slot["date"],
                    "time": slot["time"],
                })
                response = (
                    f"¡Perfecto! Reservé el espacio con **{slot['doctor']}** "
                    f"para el **{slot['date']}** a las **{slot['time']}**.\n\n"
                    f"Para registrarte, ¿cuál es tu **primer nombre**?"
                )
                history.append({"role": "assistant", "content": response})
                return {**state, "extracted": extracted, "step": "asking_first_name",
                        "response": response, "conversation_history": history}

        if accion == "pedir_alternativo":
            # Verificar el slot alternativo en la BD
            doc_req  = parsed.get("doctor_solicitado") or ""
            date_req = parsed.get("fecha_solicitada") or ""
            time_req = parsed.get("hora_solicitada") or ""

            # Si pide un doctor específico, buscar sus slots disponibles
            if doc_req and not date_req:
                doc_slots = sync_get_slots_for_doctor(doc_req, extracted.get("specialty", ""))
                if doc_slots:
                    doc_slots_text = _slots_to_text(doc_slots)
                    response = (
                        f"Para **{doc_req}** tengo estos horarios disponibles:\n\n"
                        f"{doc_slots_text}\n\n¿Cuál te conviene?"
                    )
                else:
                    response = (
                        f"Lo siento, **{doc_req}** no tiene horarios libres en este momento. "
                        f"¿Te gustaría ver opciones con otro médico?\n\n{slots_text}"
                    )
                history.append({"role": "assistant", "content": response})
                return {**state, "extracted": extracted, "step": "choosing_slot",
                        "response": response, "conversation_history": history}

            # Si pide fecha y hora específica, verificar en BD
            if date_req and time_req:
                # Buscar qué doctor podría atender ese slot
                available_for_slot = [
                    s for s in slots
                    if s["date"] == date_req and s["time"] == time_req
                ]
                if available_for_slot:
                    slot = available_for_slot[0]
                    is_free = sync_check_slot(slot["doctor"], date_req, time_req)
                    if is_free:
                        extracted.update({
                            "doctor": slot["doctor"],
                            "date": date_req,
                            "time": time_req,
                        })
                        response = (
                            f"¡Perfecto! Ese horario está disponible con "
                            f"**{slot['doctor']}** el **{date_req}** a las **{time_req}**.\n\n"
                            f"Para registrarte, ¿cuál es tu **primer nombre**?"
                        )
                        history.append({"role": "assistant", "content": response})
                        return {**state, "extracted": extracted,
                                "step": "asking_first_name",
                                "response": response, "conversation_history": history}

                # No disponible — el LLM sugiere alternativas cercanas
                prompt_alt = f"""Eres el asistente de Clínica Cobba.
El paciente pidió el {date_req} a las {time_req} pero ese horario no está disponible.
Horarios reales disponibles:
{slots_text}

Explica amablemente que ese horario no está libre y sugiere las 2 opciones más cercanas.
Sé breve y directo. Responde en español."""
                response = llm.invoke([SystemMessage(content=prompt_alt)]).content.strip()
                history.append({"role": "assistant", "content": response})
                return {**state, "extracted": extracted, "step": "choosing_slot",
                        "response": response, "conversation_history": history}

        if accion == "cancelar":
            return {**state, "step": "done", "extracted": {},
                    "response": "Entendido, cancelé el proceso. ¿En qué más puedo ayudarte?",
                    "conversation_history": history}

        # Fallback — mostrar opciones de nuevo
        response = (
            f"No logré identificar tu elección. Los horarios disponibles son:\n\n"
            f"{slots_text}\n\nElige un número o dime el día y hora que prefieres."
        )
        history.append({"role": "assistant", "content": response})
        return {**state, "extracted": extracted, "step": "choosing_slot",
                "response": response, "conversation_history": history}

    # ── PASO 3-5: Recolección de datos del paciente ────────────────────────
    if step == "asking_first_name":
        name = user_input.strip().title()
        extracted["firstName"] = name
        response = f"Gracias, {name}. ¿Cuáles son tus **apellidos**?"
        history.append({"role": "assistant", "content": response})
        return {**state, "extracted": extracted, "step": "asking_last_name",
                "response": response, "conversation_history": history}

    if step == "asking_last_name":
        extracted["lastName"] = user_input.strip().title()
        response = "Perfecto. Por último, ingresa tu **número de DNI** (8 dígitos):"
        history.append({"role": "assistant", "content": response})
        return {**state, "extracted": extracted, "step": "asking_dni",
                "response": response, "conversation_history": history}

    if step == "asking_dni":
        dni = re.sub(r"[^0-9]", "", user_input)
        if len(dni) < 8:
            response = "❌ El DNI parece inválido (mínimo 8 dígitos). Inténtalo de nuevo:"
            history.append({"role": "assistant", "content": response})
            return {**state, "response": response, "conversation_history": history}

        extracted["dni"] = dni
        resumen = (
            f"✅ **Resumen de tu cita:**\n"
            f"- Paciente: {extracted.get('firstName')} {extracted.get('lastName')}\n"
            f"- Especialidad: {extracted.get('specialty')}\n"
            f"- Médico: {extracted.get('doctor')}\n"
            f"- Fecha/Hora: {extracted.get('date')} a las {extracted.get('time')}\n\n"
            f"¿Confirmas? Responde **Sí** o **No**."
        )
        history.append({"role": "assistant", "content": resumen})
        return {**state, "extracted": extracted, "step": "ready_to_schedule",
                "response": resumen, "conversation_history": history}

    # ── PASO 6: Confirmación final ─────────────────────────────────────────
    if step == "ready_to_schedule":
        # LLM interpreta si el usuario confirma o no
        prompt = f"""El usuario responde a una confirmación de cita médica.
¿Confirma o cancela? Responde SOLO: si | no
Mensaje: "{user_input}"
"""
        decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
        confirmed = "si" in decision or "sí" in decision

        if confirmed:
            extracted_clean = {k: v for k, v in extracted.items()
                               if k != "available_slots"}
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

    # Construir contexto del historial para que el LLM recuerde la conversación
    history_text = "\n".join(
        f"{'Paciente' if m['role']=='user' else 'Asistente'}: {m['content']}"
        for m in history[-6:]  # últimos 6 mensajes
    )

    prompt = f"""Eres el asistente virtual de la Clínica Cobba, una clínica médica peruana.
Responde de forma amable y concisa en español.
Si el usuario quiere agendar una cita, dile que escriba "quiero agendar una cita".
Si quiere hablar con un humano, dile que escriba "hablar con recepción".

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
def route_main(state: AgentState) -> str:
    step = state["step"]
    if step == "handoff":
        return "escalation"
    if step in ("asking_specialty", "choosing_slot", "asking_first_name",
                "asking_last_name", "asking_dni", "ready_to_schedule"):
        return "scheduler"
    return "router"

def route_after_router(state: AgentState) -> str:
    if state["step"] == "handoff":
        return "escalation"
    if state["step"] == "asking_specialty":
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
