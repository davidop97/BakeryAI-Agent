from pathlib import Path
import json

PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

def create_chunks():
    """
    Convierte las FAQs en chunks y las guarda en chunks.jsonl.
    Cada chunk es una pregunta+respuesta con metadatos.
    """
    faqs_path = PROCESSED_DIR / "faqs.json"
    chunks = []

    # Leer faqs.json
    with faqs_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        faqs = data.get("faqs", [])

    # Crear un chunk por FAQ
    for idx, faq in enumerate(faqs):
        pregunta = faq.get("pregunta", "").strip()
        respuesta = faq.get("respuesta", "").strip()
        if pregunta and respuesta:
            chunks.append({
                "id": f"faq-{idx}",
                "source": "faqs",
                "text": f"P: {pregunta}\nR: {respuesta}"
            })

    # Guardar chunks en chunks.jsonl
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Generados {len(chunks)} chunks en {CHUNKS_PATH}")

if __name__ == "__main__":
    create_chunks()
