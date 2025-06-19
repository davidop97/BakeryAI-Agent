from pathlib import Path
import json

PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

def create_chunks():
    """
    Convierte las FAQs y el catálogo en chunks y los guarda en chunks.jsonl.
    Cada chunk es una pregunta+respuesta (FAQs) o descripción de producto (catálogo) con metadatos.
    """
    faqs_path = PROCESSED_DIR / "faqs.json"
    catalog_path = PROCESSED_DIR / "catalog.json"
    chunks = []

    # Leer faqs.json
    if faqs_path.exists():
        with faqs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            faqs = data.get("faqs", [])
            for idx, faq in enumerate(faqs):
                pregunta = faq.get("pregunta", "").strip()
                respuesta = faq.get("respuesta", "").strip()
                if pregunta and respuesta:
                    chunks.append({
                        "id": f"faq-{idx}",
                        "source": "faqs",
                        "section": "faq",
                        "text": f"P: {pregunta}\nR: {respuesta}"
                    })

    # Leer catalog.json
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            productos = data.get("productos", [])
            for idx, producto in enumerate(productos):
                nombre = producto.get("nombre", "").strip()
                descripcion = producto.get("descripcion", "").strip()
                precio = producto.get("precio", 0)
                categoria = producto.get("categoria", "").strip()
                product_id = producto.get("id", "").strip()
                if nombre and descripcion:
                    chunks.append({
                        "id": f"producto-{idx}",
                        "source": "catalog",
                        "section": "producto",
                        "text": f"Producto: {nombre}\nDescripción: {descripcion}\nPrecio: ${precio}\nCategoría: {categoria}\nID: {product_id}"
                    })

    # Guardar chunks en chunks.jsonl
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Generados {len(chunks)} chunks en {CHUNKS_PATH}")

if __name__ == "__main__":
    create_chunks()