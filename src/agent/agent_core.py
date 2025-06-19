from pathlib import Path
import os
import json
import re
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pydantic import SecretStr
from src.database.db_manager import DatabaseManager
from src.agent.llm_handler import LLMHandler

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.debug(f"Procesando consulta para pedido: {consulta}")
    
    # Palabras clave para detectar pedidos
    order_keywords = r"\b(quiero|pedir|comprar|dame|necesito)\b"
    if not re.search(order_keywords, consulta.lower()):
        logging.debug("No se encontraron palabras clave de pedido")
        return None, None

    # Extraer cantidad (e.g., "2 tortas", "tres croissants", "un croissant")
    quantity_match = re.search(r"\b(\d+|un|una|dos|tres|cuatro|cinco)\b\s*", consulta.lower())
    quantity = 1  # Default
    if quantity_match:
        num_str = quantity_match.group(1)
        num_map = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
        quantity = num_map.get(num_str, int(num_str)) if num_str.isdigit() else num_map.get(num_str, 1)
        logging.debug(f"Cantidad detectada: {quantity}")

    # Extraer nombre del producto, ignorando artículos y plurales
    product_match = re.search(
        r"\b(?:quiero|pedir|comprar|dame|necesito)\b\s*(?:\d+|un|una|dos|tres|cuatro|cinco)?\s*(?:un|una|el|la|los|las)?\s*([\w\s]+?)(?:\s*de\s*[\w\s]+?)?(?:s)?(?:\s|$)",
        consulta.lower(),
        re.IGNORECASE
    )
    product_name = product_match.group(1).strip() if product_match else None
    if product_name and re.match(r"^(un|una|el|la|los|las|a)$", product_name.lower()):
        product_name = None  # Evitar capturar artículos como productos
    logging.debug(f"Nombre del producto detectado: {product_name}")

    return product_name, quantity

def validate_product(product_name: str) -> dict | None:
    """
    Valida si el producto existe en catalog.json, manejando variaciones.
    
    Args:
        product_name (str): Nombre del producto a buscar.
    
    Returns:
        dict: Detalles del producto si existe, None si no.
    """
    logging.debug(f"Validando producto: {product_name}")
    if not CATALOG_PATH.exists():
        logging.error("catalog.json no existe")
        return None
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
        productos = data.get("productos", [])
        for producto in productos:
            catalog_name = producto.get("nombre", "").lower().rstrip("s")
            if product_name.lower() in catalog_name or catalog_name in product_name.lower():
                logging.debug(f"Producto encontrado: {producto['nombre']}")
                return producto
    logging.debug("Producto no encontrado en catálogo")
    return None

def buscar_respuesta(consulta: str) -> str | None:
    """
    Procesa la consulta: primero intenta con pedidos/productos, luego FAQs, y finalmente el LLM.
    
    Args:
        consulta (str): Texto de la consulta del usuario.
    
    Returns:
        str: Respuesta del LLM, FAQ, producto, o confirmación de pedido.
    """
    if not consulta or not isinstance(consulta, str):
        logging.error("Consulta inválida")
        return None

    logging.info(f"Procesando consulta: {consulta}")

    # Inicializar DatabaseManager y LLMHandler
    db_manager = DatabaseManager()
    try:
        llm_handler = LLMHandler()
    except ValueError as e:
        logging.error(f"Error al inicializar LLMHandler: {e}")
        llm_handler = None

    # Obtener API key para OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY no encontrada en .env")
        return None

    # Instanciar embeddings
    embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key))

    # Cargar vector store
    try:
        vectordb = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Error al cargar el vector store: {e}")
        return None

    # Verificar si es un pedido
    product_name, quantity = extract_order_info(consulta)
    if product_name and quantity:
        logging.debug(f"Intento de pedido: {product_name}, cantidad: {quantity}")
        # Validar contra catalog.json
        producto = validate_product(product_name)
        if producto:
            # Buscar producto en el vector store para confirmar
            results = vectordb.similarity_search(product_name, k=1, score_threshold=0.8)
            if results and results[0].page_content and "Producto:" in results[0].page_content:
                # Guardar pedido en orders.db
                order_id = db_manager.add_order(product_id=producto["id"], quantity=quantity)
                response = f"Pedido registrado: {quantity} x {producto['nombre']} por ${producto['precio'] * quantity}. ID del pedido: {order_id}."
                logging.info(f"Pedido registrado: {response}")
                db_manager.add_interaction(consulta, response)
                return response
            else:
                logging.debug("Producto no encontrado en vector store")
        else:
            response = f"No encontramos '{product_name}' en nuestro catálogo."
            logging.warning(f"Producto no encontrado: {product_name}")
            db_manager.add_interaction(consulta, response)
            return response

    # Buscar productos (e.g., "¿Tienen torta de chocolate?")
    product_keywords = ["integral", "croissant", "galletas", "chocolate", "torta", "pan"]
    if any(keyword in consulta.lower() for keyword in product_keywords):
        logging.debug("Buscando producto en vector store")
        results = vectordb.similarity_search(consulta, k=1, score_threshold=0.8)
        if results and results[0].page_content and "Producto:" in results[0].page_content:
            response = results[0].page_content.strip()
            logging.info(f"Producto encontrado: {response}")
            db_manager.add_interaction(consulta, response)
            return response

    # Buscar en FAQs (solo si parece una pregunta específica)
    faq_keywords = [
        "horario", "gluten", "domicilio", "pago", "personalizado", "café", "fresco",
        "vegano", "reserva", "wi-fi", "bebida", "diabético", "ubicación"
    ]
    if any(keyword in consulta.lower() for keyword in faq_keywords):
        logging.debug("Buscando FAQ en vector store")
        results = vectordb.similarity_search(consulta, k=1, score_threshold=0.8)
        if results and results[0].page_content and "R:" in results[0].page_content:
            response = results[0].page_content.split("R:")[1].strip()
            logging.info(f"FAQ encontrada: {response}")
            db_manager.add_interaction(consulta, response)
            return response

    # Consultar el LLM para todo lo demás
    if llm_handler:
        logging.debug("Consultando LLM")
        response = llm_handler.query_llm(consulta)
        logging.info(f"Respuesta del LLM: {response}")
        return response
    else:
        response = "No se pudo procesar la consulta debido a un error con el LLM."
        logging.error("LLMHandler no disponible")
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