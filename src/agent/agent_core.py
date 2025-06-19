from pathlib import Path
import os
import json
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pydantic import SecretStr
from src.database.db_manager import DatabaseManager
from src.agent.llm_handler import LLMHandler

# Cargar variables de entorno desde .env
load_dotenv(override=True)

VECTOR_DIR = Path("data/processed/vectordb")
CATALOG_PATH = Path("data/processed/catalog.json")

def extract_order_info(consulta: str) -> tuple[str | None, int | None]:
    """
    Extrae el nombre del producto y la cantidad de una consulta de pedido.
    
    Args:
        consulta (str): Consulta del usuario.
    
    Returns:
        tuple: (nombre del producto, cantidad) o (None, None) si no es un pedido.
    """
    # Palabras clave para detectar pedidos
    order_keywords = r"(quiero|pedir|comprar|dame|necesito)\s+"
    if not re.search(order_keywords, consulta.lower()):
        return None, None

    # Extraer cantidad (e.g., "2 tortas", "tres croissants", "un croissant")
    quantity_match = re.search(r"(\d+|un|una|dos|tres|cuatro|cinco)\s+", consulta.lower())
    quantity = 1  # Default
    if quantity_match:
        num_str = quantity_match.group(1)
        num_map = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
        quantity = num_map.get(num_str, int(num_str)) if num_str.isdigit() else num_map.get(num_str, 1)

    # Extraer nombre del producto, ignorando artículos y plurales
    product_match = re.search(
        r"(?:quiero|pedir|comprar|dame|necesito)\s*(?:\d+|un|una|dos|tres|cuatro|cinco)?\s*(?:un|una|el|la|los|las)?\s*([\w\s]+?)(?:s)?(?:\s|$)",
        consulta.lower(),
        re.IGNORECASE
    )
    product_name = product_match.group(1).strip() if product_match else None

    return product_name, quantity

def validate_product(product_name: str) -> dict | None:
    """
    Valida si el producto existe en catalog.json, manejando variaciones.
    
    Args:
        product_name (str): Nombre del producto a buscar.
    
    Returns:
        dict: Detalles del producto si existe, None si no.
    """
    if not CATALOG_PATH.exists():
        return None
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
        productos = data.get("productos", [])
        for producto in productos:
            catalog_name = producto.get("nombre", "").lower().rstrip("s")
            if product_name.lower() in catalog_name or catalog_name in product_name.lower():
                return producto
    return None

def buscar_respuesta(consulta: str) -> str | None:
    """
    Procesa la consulta: primero intenta con pedidos/FAQs, luego con el LLM.
    
    Args:
        consulta (str): Texto de la consulta del usuario.
    
    Returns:
        str: Respuesta del LLM, FAQ, producto, o confirmación de pedido.
    """
    if not consulta or not isinstance(consulta, str):
        return None

    # Inicializar DatabaseManager y LLMHandler
    db_manager = DatabaseManager()
    try:
        llm_handler = LLMHandler()
    except ValueError as e:
        print(f"Error al inicializar LLMHandler: {e}")
        llm_handler = None

    # Obtener API key para OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY no encontrada en .env")
        return None

    # Instanciar embeddings
    embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key))

    # Cargar vector store
    try:
        vectordb = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error al cargar el vector store: {e}")
        return None

    # Verificar si es un pedido
    product_name, quantity = extract_order_info(consulta)
    if product_name and quantity:
        # Buscar producto en el vector store
        results = vectordb.similarity_search(product_name, k=1, score_threshold=0.9)
        if results and results[0].page_content and "Producto:" in results[0].page_content:
            # Validar contra catalog.json
            producto = validate_product(product_name)
            if producto:
                # Guardar pedido en orders.db
                order_id = db_manager.add_order(product_id=producto["id"], quantity=quantity)
                response = f"Pedido registrado: {quantity} x {producto['nombre']} por ${producto['precio'] * quantity}. ID del pedido: {order_id}."
                # Guardar interacción
                db_manager.add_interaction(consulta, response)
                return response
            else:
                response = f"No encontramos '{product_name}' en nuestro catálogo."
                db_manager.add_interaction(consulta, response)
                return response

    # Buscar en FAQs o productos (solo si parece una pregunta específica)
    faq_keywords = [
        "horario", "gluten", "domicilio", "pago", "personalizado", "café", "pan", "fresco",
        "vegano", "reserva", "torta", "wi-fi", "bebida", "diabético", "ubicación",
        "integral", "croissant", "galletas", "chocolate"
    ]
    if any(keyword in consulta.lower() for keyword in faq_keywords):
        results = vectordb.similarity_search(consulta, k=1, score_threshold=0.9)
        if results and results[0].page_content:
            text = results[0].page_content
            if "R:" in text:
                response = text.split("R:")[1].strip()
            elif "Producto:" in text:
                response = text.strip()
            else:
                response = text.strip()
            db_manager.add_interaction(consulta, response)
            return response

    # Consultar el LLM para todo lo demás
    if llm_handler:
        response = llm_handler.query_llm(consulta) or "No se encontró respuesta"
        return response
    else:
        response = "No se pudo procesar la consulta debido a un error con el LLM."
        db_manager.add_interaction(consulta, response)
        return response

if __name__ == "__main__":
    test_consultas = [
        "¿Cuáles son los horarios de atención?",
        "¿Tienen opciones sin gluten?",
        "¿Tienen torta de chocolate?",
        "Quiero 2 croissants",
        "Pedir una torta de chocolate",
        "¿Qué tal el clima?",
        "¿Puedes recomendar un postre?"
    ]
    for consulta in test_consultas:
        respuesta = buscar_respuesta(consulta)
        print(f"Consulta: {consulta}")
        print(f"Respuesta: {respuesta or 'No se encontró respuesta'}\n")