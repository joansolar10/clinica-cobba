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
import unicodedata

# ── Normalización de texto en español ──────────────────────────────────────
# Sin esto, TfidfVectorizer compara palabras EXACTAS: "sábado" != "sábados",
# "seguro" != "seguros", y palabras funcionales (de, la, el...) generan
# similitud artificial con documentos totalmente irrelevantes.
_SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mi", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estas", "algunas", "algo",
    "nosotros", "tú", "te", "ti", "tu", "tus", "ellas", "es", "son",
    "soy", "eres", "somos", "tengo", "tiene", "tienes", "tenemos",
    "puede", "pueden", "debe", "deben", "está", "están", "estoy",
}


def _strip_accents(text: str) -> str:
    """Quita tildes: 'sábado' -> 'sabado'."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


# Comparamos tokens ya normalizados (sin tilde) contra esta versión también
# sin tilde de las stopwords, para que entradas como "sí"/"qué" (guardadas
# con tilde arriba) sigan filtrando correctamente sus variantes sin tilde
# ("si", "que") una vez que _analyze() normaliza antes de comparar.
_SPANISH_STOPWORDS_NORM = {_strip_accents(w) for w in _SPANISH_STOPWORDS}


def _normalize_word(word: str) -> str:
    """Normalización morfológica ligera para reducir singular/plural al
    mismo token: 'sábados' -> 'sabado', 'seguros' -> 'seguro'."""
    w = _strip_accents(word)
    if len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    return w


def _analyze(text: str) -> list[str]:
    """Tokeniza, normaliza (quita tildes y plurales simples) y luego filtra
    stopwords. El ORDEN importa: si filtráramos antes de normalizar,
    palabras con tilde como 'cuál' no calzarían con la entrada sin tilde
    'cual' de la lista de stopwords y se colarían como si fueran una
    palabra de contenido (causaba falsos positivos, ej. la pregunta
    "¿cuál es la capital de Francia?" matcheando documentos al azar)."""
    raw_tokens = re.findall(r"[a-záéíóúñ0-9]+", (text or "").lower())
    normalized = (_normalize_word(t) for t in raw_tokens)
    return [t for t in normalized if t not in _SPANISH_STOPWORDS_NORM and len(t) > 1]

# ── Base de conocimiento (documentos fuente para RAG) ──────────────────────
KB_DOCUMENTS: list[dict] = [
    {
        "id": "horarios_ubicacion",
        "categoria": "Horarios y ubicación",
        "contenido": (
            "Clínica Cobba atiende de lunes a sábado de 9:00 a.m. a 7:00 p.m. "
            "Nuestro horario de atención, es decir la hora en la que abrimos y "
            "atendemos, es de 9:00 a.m. a 7:00 p.m. de lunes a sábado. "
            "Domingos y feriados permanecemos cerrados, no abrimos. "
            "Nuestra dirección y ubicación es Av. España 662, Trujillo 13011. "
            "Si preguntas dónde quedamos, dónde estamos ubicados o cómo llegar, "
            "esa es la dirección de la clínica. "
            "Teléfono de recepción: 924 461 285."
        ),
    },
    {
        "id": "seguros_convenios",
        "categoria": "Seguros y convenios",
        "contenido": (
            "Trabajamos bajo la modalidad de reembolso con las principales "
            "aseguradoras del país: Rímac, Pacífico Seguros, La Positiva y "
            "Mapfre. El paciente paga la consulta directamente en la clínica "
            "y nosotros le entregamos la factura y el informe médico necesarios "
            "para que gestione el reembolso con su aseguradora. No trabajamos "
            "con descuento directo en caja (copago) por el momento."
        ),
    },
    {
        "id": "precios_referenciales",
        "categoria": "Precios referenciales",
        "contenido": (
            "El costo final de cada tratamiento se confirma tras la evaluación "
            "del especialista, ya que depende del diagnóstico. Si preguntas "
            "cuánto cuesta, cuál es el precio o el costo de una consulta, "
            "estos son los precios referenciales: Consulta/evaluación general "
            "S/ 80. Limpieza dental "
            "(profilaxis) S/ 120 a S/ 150. Resina simple S/ 150 a S/ 250. "
            "Extracción simple S/ 100 a S/ 180. Ortodoncia con brackets "
            "metálicos desde S/ 2,500, con evaluación previa sin costo."
        ),
    },
    {
        "id": "politica_cancelacion",
        "categoria": "Política de cancelación y reprogramación",
        "contenido": (
            "El paciente puede cancelar una cita o reprogramar/modificar una "
            "cita sin costo si avisa con al menos 24h (un día) de "
            "anticipación, ya sea por este chat o llamando a recepción. Si "
            "el paciente falta sin avisar (No-Show) en más de 2 ocasiones "
            "consecutivas, se le pedirá confirmar su siguiente cita con 24h "
            "de anticipación antes de que quede reservada en la agenda del "
            "doctor."
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
            "Odontopediatría. Contamos con especialistas en cada área: "
            "odontólogos generales, ortodoncistas, endodoncistas, "
            "periodoncistas, implantólogos y odontopediatras. "
            "Para agendar, el paciente debe escribir que quiere agendar una "
            "cita y el bot le mostrará los horarios disponibles de cada "
            "especialidad."
        ),
    },
    {
        "id": "emergencias_dentales",
        "categoria": "Emergencias dentales",
        "contenido": (
            "Ante una emergencia, urgencia o dolor dental (dolor intenso, "
            "me duele mucho, golpe, sangrado abundante), el paciente debe "
            "escribir 'hablar con recepción' para que un humano lo atienda "
            "de inmediato. El bot no debe intentar resolver emergencias ni "
            "dar indicaciones médicas por su cuenta."
        ),
    },
]


def _tokenize(text: str) -> list[str]:
    """Tokenización + normalización para el fallback sin sklearn."""
    return _analyze(text)


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
            vectorizer = TfidfVectorizer(tokenizer=_analyze, token_pattern=None)
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
    if not ranked or ranked[0][1] < min_score:
        return []

    # Umbral relativo: un score "bueno" varía mucho según la pregunta
    # (ej. 0.22 vs 0.08), así que en vez de un corte absoluto, nos quedamos
    # solo con los documentos que estén razonablemente cerca del mejor
    # resultado. Esto evita arrastrar documentos débiles/irrelevantes solo
    # porque el mejor resultado también tuvo un score bajo.
    top_score = ranked[0][1]
    return [
        doc for doc, score in ranked[:k]
        if score >= min_score and score >= 0.4 * top_score
    ]


def format_context_for_prompt(docs: list[dict]) -> str:
    """Convierte los documentos recuperados en texto listo para insertar en el prompt del LLM."""
    if not docs:
        return (
            "(No se encontró información específica en la base de conocimiento "
            "para esta pregunta. No inventes la respuesta: indícalo con honestidad.)"
        )

    def _strip_internal_notes(contenido: str) -> str:
        # Los marcadores [COMPLETAR: ...] son notas internas para el equipo
        # (datos aún no verificados), NUNCA deben llegar al paciente. Se
        # quitan aquí mismo, en el código, en vez de confiar en que el LLM
        # obedezca la instrucción de ignorarlos (un modelo chico como el
        # 8B a veces los copia igual si están literalmente en el contexto).
        text = re.sub(r"\[COMPLETAR:[^\]]*\]", "", contenido)
        text = re.sub(r"\s+\.", ".", text)   # limpia " ." residual
        text = re.sub(r"\.{2,}", ".", text)  # colapsa ".." residual
        text = re.sub(r"\s{2,}", " ", text)  # colapsa espacios dobles
        return text.strip()

    return "\n\n".join(
        f"[{d['categoria']}]\n{_strip_internal_notes(d['contenido'])}" for d in docs
    )