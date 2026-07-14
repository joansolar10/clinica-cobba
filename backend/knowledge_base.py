"""
knowledge_base.py — Clínica Cobba
Base de conocimiento local para RAG (Retrieval-Augmented Generation).

¿Por qué existe este archivo?
El agente conversacional (agent.py) respondía preguntas generales del
paciente (horarios, seguros, precios, políticas, etc.) usando SOLO lo que
el LLM "recordaba" de su entrenamiento, es decir, podía inventar (alucinar)
datos de la clínica. Este módulo agrega una capa de recuperación: antes de
responder, buscamos en documentos reales de la clínica los fragmentos más
relevantes a la pregunta y se los damos al LLM como contexto obligatorio.

Cómo funciona:
1. KB_DOCUMENTS contiene los documentos/fragmentos de conocimiento de la
   clínica (horarios, políticas, seguros, precios referenciales, etc.)
2. retrieve_context() calcula similitud TF-IDF entre la pregunta del
   paciente y cada documento, y devuelve los k fragmentos más relevantes.
3. format_context_for_prompt() convierte esos fragmentos en texto listo
   para insertar en el prompt del LLM (ver agent.py -> fallback_node).

⚠️ IMPORTANTE PARA EL EQUIPO:
Los textos de KB_DOCUMENTS son una PLANTILLA inicial con marcadores
[COMPLETAR: ...]. Deben reemplazarse con la información real y verificada
de la clínica (horarios reales, tarifas reales, convenios con seguros,
dirección, teléfono, etc.) antes de pasar a producción. No se debe dejar
información inventada de cara al paciente.

Cómo agregar nueva información:
Simplemente añade un nuevo dict a KB_DOCUMENTS con "id", "categoria" y
"contenido". No se requiere reentrenar nada: la búsqueda es en tiempo real.
"""

from __future__ import annotations
import re

# ── Base de conocimiento (documentos fuente para RAG) ──────────────────────
KB_DOCUMENTS: list[dict] = [
    {
        "id": "horarios_ubicacion",
        "categoria": "Horarios y ubicación",
        "contenido": (
            "Clínica Cobba atiende de lunes a sábado de 9:00 a.m. a 7:00 p.m. "
            "Domingos y feriados permanecemos cerrados. "
            "[COMPLETAR: dirección exacta, referencia y teléfono de recepción]."
        ),
    },
    {
        "id": "seguros_convenios",
        "categoria": "Seguros y convenios",
        "contenido": (
            "Aceptamos reembolso de gastos médicos con la mayoría de seguros "
            "privados; el paciente paga la consulta y luego tramita el "
            "reembolso con su aseguradora. "
            "[COMPLETAR: lista real de aseguradoras/convenios y el "
            "procedimiento exacto para usarlos en esta clínica]."
        ),
    },
    {
        "id": "precios_referenciales",
        "categoria": "Precios referenciales",
        "contenido": (
            "El costo final de cada tratamiento se confirma tras la "
            "evaluación del especialista, ya que depende del diagnóstico. "
            "[COMPLETAR: rangos de precios reales por especialidad, si la "
            "clínica desea publicarlos, por ejemplo consulta general, "
            "limpieza dental, ortodoncia, etc.]."
        ),
    },
    {
        "id": "politica_cancelacion",
        "categoria": "Política de cancelación y reprogramación",
        "contenido": (
            "El paciente puede cancelar una cita o reprogramar/modificar una "
            "cita sin costo si avisa con al menos 24 horas de anticipación, "
            "ya sea por este chat o llamando a recepción. "
            "[COMPLETAR: política real de penalidad por inasistencia "
            "(No-Show) si la clínica aplica alguna]."
        ),
    },
    {
        "id": "primera_visita",
        "categoria": "Primera visita / qué llevar",
        "contenido": (
            "Para la primera cita, el paciente debe llegar 10 minutos antes "
            "con su DNI. Si cuenta con radiografías o historial dental "
            "previo, puede traerlo para agilizar el diagnóstico del "
            "especialista."
        ),
    },
    {
        "id": "especialidades_disponibles",
        "categoria": "Especialidades disponibles",
        "contenido": (
            "Clínica Cobba atiende las especialidades de Odontología "
            "General, Ortodoncia, Endodoncia, Periodoncia, Implantología y "
            "Odontopediatría. Para agendar, el paciente debe escribir que "
            "quiere agendar una cita y el bot le mostrará los horarios "
            "disponibles de cada especialidad."
        ),
    },
    {
        "id": "emergencias_dentales",
        "categoria": "Emergencias dentales",
        "contenido": (
            "Ante una emergencia dental (dolor intenso, golpe, sangrado "
            "abundante), el paciente debe escribir 'hablar con recepción' "
            "para que un humano lo atienda de inmediato. El bot no debe "
            "intentar resolver emergencias ni dar indicaciones médicas por "
            "su cuenta."
        ),
    },
]


def _tokenize(text: str) -> list[str]:
    """Tokenización simple (minúsculas, solo letras/números) para el fallback sin sklearn."""
    return re.findall(r"[a-záéíóúñ0-9]+", (text or "").lower())


def _keyword_score(query_tokens: set[str], doc_tokens: list[str]) -> float:
    """Puntaje de respaldo: proporción de palabras del documento que aparecen en la consulta."""
    if not doc_tokens:
        return 0.0
    overlap = sum(1 for t in doc_tokens if t in query_tokens)
    return overlap / len(set(doc_tokens))


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except Exception:
    # Si scikit-learn no está instalado, el sistema sigue funcionando con un
    # retriever más simple (conteo de palabras en común) en vez de romperse.
    _SKLEARN_OK = False


def retrieve_context(query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
    """
    Devuelve los k documentos de KB_DOCUMENTS más relevantes para `query`.

    - Usa similitud TF-IDF (scikit-learn) si está disponible.
    - Si no, usa un conteo simple de palabras en común como respaldo, para
      que el agente nunca se caiga por falta de esta dependencia opcional.
    - Filtra resultados con score menor a `min_score` para no inyectar
      contexto irrelevante en el prompt (evita "ruido" que confunda al LLM).
    """
    if not query or not query.strip():
        return []

    contents = [d["contenido"] for d in KB_DOCUMENTS]

    try:
        if _SKLEARN_OK:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(contents + [query])
            sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        else:
            query_tokens = set(_tokenize(query))
            sims = [_keyword_score(query_tokens, _tokenize(c)) for c in contents]
    except Exception:
        # Cualquier fallo en la recuperación no debe tumbar la conversación:
        # el agente simplemente responde sin contexto adicional de la KB.
        return []

    ranked = sorted(zip(KB_DOCUMENTS, sims), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, score in ranked[:k] if score >= min_score]


def format_context_for_prompt(docs: list[dict]) -> str:
    """Convierte los documentos recuperados en texto listo para insertar en el prompt del LLM."""
    if not docs:
        return (
            "(No se encontró información específica en la base de conocimiento "
            "para esta pregunta. No inventes la respuesta: indícalo con honestidad.)"
        )
    return "\n\n".join(f"[{d['categoria']}]\n{d['contenido']}" for d in docs)