"""
Deep Agent — Clínica Cobba
Worker asíncrono que analiza datos reales de citas y genera
recomendaciones únicas usando LangGraph + Groq.

Grafo:
  data_collector → pattern_analyzer → recommendation_generator
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from datetime import datetime
import os, json, random

# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,          # más alto = recomendaciones más variadas cada día
    api_key=os.environ["GROQ_API_KEY"],
)

# ── Estado del grafo ───────────────────────────────────────────────────────
class DeepAgentState(TypedDict):
    appointments: List[dict]   # datos crudos de citas
    stats: List[dict]          # estadísticas por día
    patterns: str              # análisis de patrones detectados
    alerts: List[str]          # recomendaciones finales generadas

# ── Nodo 1: DATA COLLECTOR — prepara resumen estadístico ─────────────────
def data_collector_node(state: DeepAgentState) -> DeepAgentState:
    appointments = state["appointments"]
    stats = state["stats"]

    # Calcular métricas por doctor
    doctor_stats: dict = {}
    for appt in appointments:
        doc = appt.get("doctor", "Desconocido")
        if doc not in doctor_stats:
            doctor_stats[doc] = {
                "total": 0, "no_show": 0,
                "specialty": appt.get("specialty", "General"),
                "days": []
            }
        doctor_stats[doc]["total"] += 1
        if appt.get("status") == "No-Show":
            doctor_stats[doc]["no_show"] += 1
        doctor_stats[doc]["days"].append(appt.get("date", ""))

    # Calcular métricas por día de la semana
    day_stats: dict = {}
    for s in stats:
        day = s.get("name", "?")
        citas = s.get("citas", 0)
        no_shows = s.get("noShows", 0)
        tasa = round((no_shows / citas * 100), 1) if citas > 0 else 0
        day_stats[day] = {"citas": citas, "no_shows": no_shows, "tasa_noshow": tasa}

    # Encontrar día crítico y doctor crítico
    peor_dia = max(day_stats, key=lambda d: day_stats[d]["tasa_noshow"]) if day_stats else "N/A"
    peor_doctor = max(
        doctor_stats,
        key=lambda d: doctor_stats[d]["no_show"] / max(doctor_stats[d]["total"], 1)
    ) if doctor_stats else "N/A"

    summary = {
        "fecha_analisis": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_citas": len(appointments),
        "doctor_stats": doctor_stats,
        "day_stats": day_stats,
        "peor_dia": peor_dia,
        "peor_doctor": peor_doctor,
        "peor_doctor_especialidad": doctor_stats.get(peor_doctor, {}).get("specialty", ""),
        "tasa_noshow_global": round(
            sum(d["no_shows"] for d in day_stats.values()) /
            max(sum(d["citas"] for d in day_stats.values()), 1) * 100, 1
        )
    }

    return {**state, "patterns": json.dumps(summary, ensure_ascii=False)}

# ── Nodo 2: PATTERN ANALYZER — LLM detecta patrones críticos ─────────────
def pattern_analyzer_node(state: DeepAgentState) -> DeepAgentState:
    summary = json.loads(state["patterns"])

    prompt = f"""# Rol y contexto
Eres un analista de datos clínicos experto que apoya a la gerencia de una
clínica odontológica peruana (Clínica Cobba) a detectar problemas
operativos a partir de datos reales de citas.

# Datos del análisis ({summary['fecha_analisis']})
- Total de citas analizadas: {summary['total_citas']}
- Tasa de No-Show global: {summary['tasa_noshow_global']}%
- Día con mayor tasa de No-Show: {summary['peor_dia']} ({summary['day_stats'].get(summary['peor_dia'], {}).get('tasa_noshow', 0)}%)
- Doctor con más No-Shows: {summary['peor_doctor']} ({summary['peor_doctor_especialidad']})
- Estadísticas por doctor: {json.dumps(summary['doctor_stats'], ensure_ascii=False)}
- Estadísticas por día: {json.dumps(summary['day_stats'], ensure_ascii=False)}

# Instrucciones
Detecta los 2 patrones más críticos que requieren acción inmediata de la
gerencia, basándote ÚNICAMENTE en los datos anteriores.

# Restricciones
- No inventes cifras ni doctores que no aparezcan en los datos.
- Si los datos no alcanzan para dos patrones distintos, repite el patrón
  más relevante en ambos campos en vez de inventar uno nuevo.

# Manejo de errores
Si algún dato viene vacío o en "N/A", indícalo como impacto "bajo" en vez
de fabricar una conclusión.

# Formato de salida
Responde SOLO con un JSON con esta estructura exacta (sin explicación, sin markdown):
{{
  "patron_1": {{
    "tipo": "no_show|sobrecarga|cancelacion|eficiencia",
    "descripcion": "descripción breve del patrón detectado",
    "impacto": "alto|medio|bajo",
    "datos_clave": "cifra o dato específico que evidencia el patrón"
  }},
  "patron_2": {{
    "tipo": "no_show|sobrecarga|cancelacion|eficiencia",
    "descripcion": "descripción breve del patrón detectado",
    "impacto": "alto|medio|bajo",
    "datos_clave": "cifra o dato específico que evidencia el patrón"
  }}
}}"""

    result = llm.invoke([SystemMessage(content=prompt)])
    raw = result.content.strip()

    # Limpiar posibles bloques markdown
    import re
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        patterns = json.loads(raw)
    except Exception:
        # Fallback si el LLM no devuelve JSON válido
        patterns = {
            "patron_1": {
                "tipo": "no_show",
                "descripcion": f"Alta tasa de No-Show los días {summary['peor_dia']}",
                "impacto": "alto",
                "datos_clave": f"{summary['tasa_noshow_global']}% tasa global"
            },
            "patron_2": {
                "tipo": "eficiencia",
                "descripcion": f"Concentración de inasistencias en {summary['peor_doctor']}",
                "impacto": "medio",
                "datos_clave": f"Especialidad: {summary['peor_doctor_especialidad']}"
            }
        }

    # Combinar el resumen original con los patrones detectados
    enriched = {**json.loads(state["patterns"]), "patrones_detectados": patterns}
    return {**state, "patterns": json.dumps(enriched, ensure_ascii=False)}

# ── Nodo 3: RECOMMENDATION GENERATOR — genera alertas accionables ────────
def recommendation_generator_node(state: DeepAgentState) -> DeepAgentState:
    data = json.loads(state["patterns"])
    patrones = data.get("patrones_detectados", {})
    fecha = data.get("fecha_analisis", datetime.now().strftime("%Y-%m-%d"))

    # Semilla del día para que las recomendaciones varíen cada día pero sean
    # consistentes dentro del mismo día
    day_seed = int(datetime.now().strftime("%Y%m%d"))
    random.seed(day_seed)

    alerts = []
    for key in ["patron_1", "patron_2"]:
        patron = patrones.get(key, {})
        if not patron:
            continue

        prompt = f"""# Rol
Eres el Deep Agent de Clínica Cobba, un sistema de IA analítica médica que
asesora a la gerencia con recomendaciones operativas.

# Tarea
Genera UNA recomendación ejecutiva concisa (máximo 2 oraciones) para el
siguiente patrón detectado.

Patrón detectado: {patron.get('descripcion')}
Tipo: {patron.get('tipo')}
Impacto: {patron.get('impacto')}
Dato clave: {patron.get('datos_clave')}
Contexto adicional: Análisis del {fecha}, {data.get('total_citas')} citas analizadas.

# Restricciones
- Usa solo el dato clave proporcionado; no inventes cifras adicionales.
- No des indicaciones médicas, solo recomendaciones operativas/administrativas.
- La recomendación debe ser específica, accionable y profesional.

# Formato de salida
Responde SOLO con la recomendación en español, sin introducción ni listas.
Una o dos oraciones directas."""

        result = llm.invoke([SystemMessage(content=prompt)])
        recommendation = result.content.strip()

        # Prefijo según impacto
        impacto = patron.get("impacto", "medio")
        tipo = patron.get("tipo", "general")

        prefijos = {
            "no_show": "⚠️ Alerta No-Show",
            "sobrecarga": "📊 Alerta Sobrecarga",
            "cancelacion": "🔔 Alerta Cancelaciones",
            "eficiencia": "💡 Optimización",
        }
        prefijo = prefijos.get(tipo, "📋 Recomendación")
        badge = "🔴" if impacto == "alto" else "🟡" if impacto == "medio" else "🟢"

        alerts.append(f"{badge} {prefijo}: {recommendation}")

    return {**state, "alerts": alerts}

# ── Construcción del grafo ────────────────────────────────────────────────
def build_deep_agent():
    g = StateGraph(DeepAgentState)

    g.add_node("data_collector", data_collector_node)
    g.add_node("pattern_analyzer", pattern_analyzer_node)
    g.add_node("recommendation_generator", recommendation_generator_node)

    g.set_entry_point("data_collector")
    g.add_edge("data_collector", "pattern_analyzer")
    g.add_edge("pattern_analyzer", "recommendation_generator")
    g.add_edge("recommendation_generator", END)

    return g.compile()

deep_agent_graph = build_deep_agent()

# ── Función pública que usa main.py ──────────────────────────────────────
def run_deep_agent(appointments: list, stats: list) -> List[str]:
    """
    Recibe los datos actuales de citas y estadísticas.
    Devuelve una lista de alertas/recomendaciones generadas por el LLM.
    """
    result = deep_agent_graph.invoke({
        "appointments": appointments,
        "stats": stats,
        "patterns": "",
        "alerts": [],
    })
    return result["alerts"]