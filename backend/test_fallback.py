from dotenv import load_dotenv
load_dotenv()

from agent import run_agent

tests = [
    "¿A qué hora abren los sábados?",
    "¿Aceptan mi seguro Rimac?",
    "me duele mucho una muela, es urgente",
]

for q in tests:
    print(f"PACIENTE: {q}")
    result = run_agent(q, {"step": "idle"})
    print(f"INTENT DETECTADO: {result.get('intent')}")
    print(f"STEP: {result.get('step')}")
    print(f"BOT: {result['response']}")
    print("-" * 60)