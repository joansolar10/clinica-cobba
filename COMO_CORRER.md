# 🏥 Clínica Cobba — Cómo correr el proyecto completo

## Requisitos previos
- Node.js 18+
- Python 3.10+
- Una API key gratis de Groq → https://console.groq.com

---

## 1️⃣ Configurar el backend (LangGraph + FastAPI)

```bash
# Entrar a la carpeta del backend
cd backend/

# Crear entorno virtual
python -m venv venv

# Activar el entorno
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Crear archivo de variables de entorno
cp .env.example .env
# Edita .env y pega tu GROQ_API_KEY real

# Arrancar el servidor
uvicorn main:app --reload --port 8000
```

✅ El backend estará en: http://localhost:8000  
✅ Documentación automática: http://localhost:8000/docs

---

## 2️⃣ Configurar el frontend (React + Vite)

Abre otra terminal:

```bash
# Volver a la raíz del proyecto
cd ..   # (si estabas en backend/)

# Crear el .env del frontend
cp .env.example .env
# Ya tiene VITE_BACKEND_URL=http://localhost:8000, no necesitas cambiarlo en local

# Instalar dependencias
npm install

# Arrancar el frontend
npm run dev
```

✅ El frontend estará en: http://localhost:5173

---

## 3️⃣ Probar que funciona

1. Abre http://localhost:5173
2. Escribe en el chat: **"Quiero agendar una cita"**
3. El mensaje va al backend → LangGraph lo procesa con Llama 3.1 (Groq) → responde
4. Sigue el flujo: especialidad → opción → nombre → apellido → DNI → confirmación

---

## 🚀 Despliegue en la nube (Railway)

### Backend
1. Crea proyecto en https://railway.app
2. Conecta el repo, selecciona carpeta `backend/`
3. Añade variable de entorno: `GROQ_API_KEY=tu_key`
4. Railway detecta automáticamente el `requirements.txt` y lo despliega

### Frontend
1. En Vercel (o Railway), conecta el repo
2. Añade variable: `VITE_BACKEND_URL=https://tu-backend.railway.app`
3. Build command: `npm run build` | Output: `dist/`

---

## 📁 Estructura del proyecto

```
clinica-cobba/
├── backend/
│   ├── agent.py          ← Grafo LangGraph (Router→Validator→Scheduler)
│   ├── main.py           ← Servidor FastAPI con endpoint /chat
│   ├── requirements.txt  ← Dependencias Python
│   └── .env.example      ← Plantilla de variables de entorno
├── src/
│   └── App.tsx           ← Frontend React (ya apunta al backend real)
├── .env.example          ← VITE_BACKEND_URL para el frontend
└── COMO_CORRER.md        ← Este archivo
```
