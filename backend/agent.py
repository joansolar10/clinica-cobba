"""
Agente LangGraph para Clínica Cobba
Grafo: Router → Validator → Scheduler / Escalation

Nodos:
  router_node    → clasifica la intención del usuario
  validator_node → extrae y valida datos de la cita paso a paso
  scheduler_node → confirma y registra la cita
  escalation_node→ transfiere a un humano
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os, json, re

# ── LLM (Groq - gratuito) ──────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"],
)

# ── Estado compartido del grafo ────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    intent: Optional[Literal["agendar", "cancelar", "consultar", "humano", "desconocido"]]
    step: Literal[
        "idle", "asking_specialty", "choosing_option",
        "asking_first_name", "asking_last_name", "asking_dni",
        "ready_to_schedule", "handoff", "done"
    ]
    extracted: dict          # specialty, doctor, date, time, firstName, lastName, dni
    response: str            # texto que se devuelve al frontend
    new_appointment: Optional[dict]  # si hay cita a registrar

# ── Nodo 1: ROUTER — clasifica intención ──────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    # Si ya estamos en medio de un flujo, no re-clasificar
    if state["step"] not in ("idle", "done"):
        return state

    prompt = f"""Eres un clasificador de intenciones para una clínica médica.
Responde SOLO con una de estas palabras (sin explicación):
agendar | cancelar | consultar | humano | desconocido

Mensaje del usuario: "{state['user_input']}"
"""
    result = llm.invoke([SystemMessage(content=prompt)])
    raw = result.content.strip().lower()

    intent_map = {
        "agendar": "agendar", "cancelar": "cancelar",
        "consultar": "consultar", "humano": "humano",
    }
    intent = intent_map.get(raw, "desconocido")

    new_step = state["step"]
    if intent == "agendar":
        new_step = "asking_specialty"
    elif intent == "humano":
        new_step = "handoff"

    return {**state, "intent": intent, "step": new_step}

# ── Nodo 2: VALIDATOR — extrae datos paso a paso ──────────────────────────
def validator_node(state: AgentState) -> AgentState:
    user_input = state["user_input"]
    step = state["step"]
    extracted = dict(state["extracted"])

    # ── asking_specialty ──
    if step == "asking_specialty":
        prompt = f"""El usuario quiere reservar una cita médica.
Extrae la especialidad mencionada. Si no se menciona ninguna, responde "Medicina General".
Responde SOLO con el nombre de la especialidad (ej: Cardiología, Pediatría, Dermatología, Medicina General).

Mensaje: "{user_input}"
"""
        result = llm.invoke([SystemMessage(content=prompt)])
        specialty = result.content.strip()
        extracted["specialty"] = specialty

        response = (
            f"He consultado la base de datos para **{specialty}**. "
            f"Tengo estas opciones disponibles:\n\n"
            f"1) Dr. Silva — Mañana a las 09:00 AM\n"
            f"2) Dra. Paz — Jueves a las 11:30 AM\n\n"
            f"Responde **1** o **2**, o dime otro día/hora si prefieres."
        )
        return {**state, "extracted": extracted, "step": "choosing_option", "response": response}

    # ── choosing_option ──
    if step == "choosing_option":
        prompt = f"""El usuario está eligiendo entre opciones de cita médica.
Responde SOLO con un JSON con las claves: "opcion" (1, 2, o "otra"), "dia" (string o null), "hora" (string o null).

Mensaje: "{user_input}"
"""
        result = llm.invoke([SystemMessage(content=prompt)])
        raw = result.content.strip()
        # Limpia posibles ```json
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"opcion": "otra", "dia": None, "hora": None}

        opcion = str(parsed.get("opcion", "")).strip()

        if opcion == "1":
            extracted.update({"doctor": "Dr. Silva", "date": "Mañana", "time": "09:00 AM"})
        elif opcion == "2":
            extracted.update({"doctor": "Dra. Paz", "date": "Jueves", "time": "11:30 AM"})
        else:
            dia = parsed.get("dia") or "Miércoles"
            hora = parsed.get("hora") or "10:00 AM"
            extracted.update({"doctor": "Dr. López", "date": dia, "time": hora})

        response = (
            f"¡Perfecto! Reservé el espacio con **{extracted['doctor']}** "
            f"para el {extracted['date']} a las {extracted['time']}.\n\n"
            f"Para registrarte, ¿cuál es tu **primer nombre**?"
        )
        return {**state, "extracted": extracted, "step": "asking_first_name", "response": response}

    # ── asking_first_name ──
    if step == "asking_first_name":
        name = user_input.strip().title()
        extracted["firstName"] = name
        return {
            **state,
            "extracted": extracted,
            "step": "asking_last_name",
            "response": f"Gracias, {name}. ¿Cuáles son tus **apellidos**?"
        }

    # ── asking_last_name ──
    if step == "asking_last_name":
        extracted["lastName"] = user_input.strip().title()
        return {
            **state,
            "extracted": extracted,
            "step": "asking_dni",
            "response": "Perfecto. Por último, ingresa tu **número de DNI** (8 dígitos):"
        }

    # ── asking_dni ──
    if step == "asking_dni":
        dni = re.sub(r"[^0-9]", "", user_input)
        if len(dni) < 8:
            return {
                **state,
                "response": "❌ El DNI parece inválido (mínimo 8 dígitos). Inténtalo de nuevo:"
            }
        extracted["dni"] = dni
        resumen = (
            f"✅ **Resumen de tu cita:**\n"
            f"- Paciente: {extracted.get('firstName')} {extracted.get('lastName')}\n"
            f"- Especialidad: {extracted.get('specialty')}\n"
            f"- Médico: {extracted.get('doctor')}\n"
            f"- Fecha/Hora: {extracted.get('date')} a las {extracted.get('time')}\n\n"
            f"¿Confirmas? Responde **Sí** o **No**."
        )
        return {**state, "extracted": extracted, "step": "ready_to_schedule", "response": resumen}

    return state

# ── Nodo 3: SCHEDULER — confirma y registra la cita ───────────────────────
def scheduler_node(state: AgentState) -> AgentState:
    user_input = state["user_input"].lower()

    confirmations = ["si", "sí", "yes", "ok", "confirmo", "dale", "acepto"]
    confirmed = any(c in user_input for c in confirmations)

    if confirmed:
        extracted = state["extracted"]
        appointment = {
            "patientName": f"{extracted.get('firstName', '')} {extracted.get('lastName', '')}".strip(),
            "dni": extracted.get("dni"),
            "doctor": extracted.get("doctor", "Dr. Asignado"),
            "specialty": extracted.get("specialty", "General"),
            "date": extracted.get("date", "Pendiente"),
            "time": extracted.get("time", "Pendiente"),
        }
        return {
            **state,
            "step": "done",
            "new_appointment": appointment,
            "extracted": {},
            "intent": None,
            "response": (
                "🎉 **¡Cita confirmada exitosamente!**\n"
                "Te enviaremos un recordatorio por WhatsApp 24 horas antes.\n\n"
                "¿Hay algo más en lo que pueda ayudarte?"
            )
        }
    else:
        return {
            **state,
            "step": "done",
            "extracted": {},
            "intent": None,
            "response": "Entendido, he cancelado el agendamiento. ¿En qué más puedo ayudarte?"
        }

# ── Nodo 4: ESCALATION — transfiere a humano ─────────────────────────────
def escalation_node(state: AgentState) -> AgentState:
    return {
        **state,
        "step": "handoff",
        "response": (
            "He pausado el bot y **notificado a recepción**. "
            "Un agente humano revisará tu historial y tomará el chat en breve. "
            "Por favor, espera en línea. 🧑‍💼"
        )
    }

# ── Nodo 5: FALLBACK — respuesta general del LLM ─────────────────────────
def fallback_node(state: AgentState) -> AgentState:
    prompt = f"""Eres el asistente virtual de la Clínica Cobba, una clínica médica peruana.
Responde de forma amable y concisa. Si el usuario quiere agendar una cita, sugiere que escriba "agendar cita".
Si quiere hablar con un humano, sugiere que escriba "hablar con recepción".

Pregunta del usuario: "{state['user_input']}"
"""
    result = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "response": result.content.strip(), "step": "idle"}

# ── Enrutador condicional ─────────────────────────────────────────────────
def route_after_router(state: AgentState) -> str:
    step = state["step"]
    if step == "handoff":
        return "escalation"
    if step == "asking_specialty":
        return "validator"
    if state["intent"] == "desconocido":
        return "fallback"
    return "fallback"

def route_after_validator(state: AgentState) -> str:
    if state["step"] == "ready_to_schedule":
        return "scheduler"
    return END

def route_main(state: AgentState) -> str:
    step = state["step"]
    if step == "handoff":
        return "escalation"
    if step in ("asking_specialty", "choosing_option", "asking_first_name",
                "asking_last_name", "asking_dni"):
        return "validator"
    if step == "ready_to_schedule":
        return "scheduler"
    return "router"

# ── Construcción del grafo ────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("validator", validator_node)
    g.add_node("scheduler", scheduler_node)
    g.add_node("escalation", escalation_node)
    g.add_node("fallback", fallback_node)

    g.set_conditional_entry_point(route_main)

    g.add_conditional_edges("router", route_after_router, {
        "escalation": "escalation",
        "validator": "validator",
        "fallback": "fallback",
    })
    g.add_conditional_edges("validator", route_after_validator, {
        "scheduler": "scheduler",
        END: END,
    })
    g.add_edge("scheduler", END)
    g.add_edge("escalation", END)
    g.add_edge("fallback", END)

    return g.compile()

graph = build_graph()

# ── Función pública que usa main.py ──────────────────────────────────────
def run_agent(user_input: str, current_state: dict) -> dict:
    """
    Recibe el mensaje del usuario y el estado anterior (serializado como dict).
    Devuelve el nuevo estado con 'response' y opcionalmente 'new_appointment'.
    """
    input_state: AgentState = {
        "user_input": user_input,
        "intent": current_state.get("intent"),
        "step": current_state.get("step", "idle"),
        "extracted": current_state.get("extracted", {}),
        "response": "",
        "new_appointment": None,
    }

    result = graph.invoke(input_state)
    return result
