from knowledge_base import retrieve_context, _SKLEARN_OK

print("sklearn disponible:", _SKLEARN_OK)
print("=" * 60)

tests = [
    "¿A qué hora abren los sábados?",
    "¿Aceptan mi seguro Rimac?",
    "me duele mucho una muela, es urgente",
    "¿cuál es la capital de Francia?",
]

for q in tests:
    docs = retrieve_context(q, k=3)
    print(f"PREGUNTA: {q}")
    if docs:
        for d in docs:
            print(f"   -> [{d['categoria']}]")
    else:
        print("   -> (sin resultados)")
    print()
    