from pathlib import Path
from typing import Optional, Tuple
import os
import json
import re
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from src.database.db_manager import DatabaseManager
from src.agent.llm_handler import LLMHandler

# Configurar logging para depuración y seguimiento
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar variables de entorno desde .env
load_dotenv(override=True)

# Definir rutas de archivos y directorios
VECTOR_DIR = Path("data/processed/vectordb")
CATALOG_PATH = Path("data/processed/catalog.json")

# Almacenar historial de conversaciones por usuario
chat_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Obtiene o crea el historial de mensajes para un usuario específico.

    Args:
        session_id (str): Identificador único del usuario (e.g., número de teléfono).

    Returns:
        ChatMessageHistory: Objeto que contiene el historial de la conversación.
    """
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


def extract_order_info(consulta: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extrae el nombre del producto y la cantidad de una consulta de pedido.
    Soporta frases como "quiero 2 croissants", "dame una torta de chocolate", etc.
    """
    logging.debug(f"Procesando consulta para pedido: {consulta}")

    # Palabras clave para detectar pedidos
    order_keywords = r"\b(quiero|pedir|comprar|dame|necesito)\b"
    if not re.search(order_keywords, consulta.lower()):
        logging.debug("No se encontraron palabras clave de pedido")
        return None, None

    # Extraer cantidad
    quantity: int = 1
    quantity_match = re.search(r"\b(\d+|un|una|dos|tres|cuatro|cinco|media|mitad)\b", consulta.lower())
    if quantity_match:
        num_str = quantity_match.group(1)
        num_map = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
        if num_str in num_map:
            quantity = num_map[num_str]
        elif num_str.isdigit():
            quantity = int(num_str)
        else:
            quantity = 1
        logging.debug(f"Cantidad detectada: {quantity}")

    # Eliminar palabras clave y cantidad de la consulta
    texto = consulta.lower()
    texto = re.sub(order_keywords, "", texto)
    texto = re.sub(r"\b(\d+|un|una|dos|tres|cuatro|cinco|media|mitad)\b", "", texto)
    texto = re.sub(r"\b(de|el|la|los|las|un|una|y|por|para|a)\b", "", texto)
    product_name: Optional[str] = texto.strip(" .," )

    # Si el resultado es vacío, no se detectó producto
    if not product_name or len(product_name) < 3:
        product_name = None

    logging.debug(f"Nombre del producto detectado: {product_name}")
    return product_name, quantity

def validate_product(product_name: str, vectordb) -> dict | None:
    """
    Valida si el producto existe en catalog.json o vector store con búsqueda semántica.

    Args:
        product_name (str): Nombre del producto a buscar.
        vectordb: Vector store para búsqueda semántica.

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
                logging.debug(f"Producto encontrado en catálogo: {producto['nombre']}")
                return producto
    # Búsqueda semántica en vector store con umbral ajustado
    results = vectordb.similarity_search(product_name, k=1, score_threshold=0.7)
    if results and results[0].page_content and "Producto:" in results[0].page_content:
        product_info = results[0].page_content.split("Producto:")[1].split("\n")[0].strip()
        for producto in productos:
            if product_info.lower() in producto.get("nombre", "").lower():
                logging.debug(f"Producto encontrado en vector store: {producto['nombre']}")
                return producto
    logging.debug("Producto no encontrado en catálogo ni vector store")
    return None

def buscar_respuesta(consulta: str, session_id: Optional[str] = None) -> str | None:
    """
    Procesa la consulta: primero intenta con pedidos/productos, luego FAQs, y finalmente el LLM con historial si aplica.

    Args:
        consulta (str): Texto de la consulta del usuario.
        session_id (str, optional): Identificador único del usuario para historial. Defaults to None.

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
        # Validar contra catalog.json y vector store
        producto = validate_product(product_name, vectordb)
        if producto:
            # Guardar pedido en orders.db
            order_id = db_manager.add_order(product_id=producto["id"], quantity=quantity, status="procesado")
            response = f"Pedido registrado: {quantity} x {producto['nombre']} por ${producto['precio'] * quantity}. ID del pedido: {order_id}."
            logging.info(f"Pedido registrado: {response}")
            db_manager.add_interaction(consulta, response)
            if session_id:
                get_session_history(session_id).add_user_message(consulta)
                get_session_history(session_id).add_ai_message(response)
            return response
        else:
            response = f"No encontramos '{product_name}' en nuestro catálogo. ¿Te gustaría consultar algo más o intentarlo con otro producto?"
            logging.warning(f"Producto no encontrado: {product_name}")
            db_manager.add_interaction(consulta, response)
            if session_id:
                get_session_history(session_id).add_user_message(consulta)
                get_session_history(session_id).add_ai_message(response)
            return response

    # Buscar productos solo si no es un pedido
    product_keywords = ["integral", "croissant", "galletas", "chocolate", "torta", "pan"]
    if any(keyword in consulta.lower() for keyword in product_keywords) and not product_name:
        logging.debug("Buscando producto en vector store")
        results = vectordb.similarity_search(consulta, k=1, score_threshold=0.8)
        if results and results[0].page_content and "Producto:" in results[0].page_content:
            response = results[0].page_content.strip()
            logging.info(f"Producto encontrado: {response}")
            db_manager.add_interaction(consulta, response)
            if session_id:
                get_session_history(session_id).add_user_message(consulta)
                get_session_history(session_id).add_ai_message(response)
            return response

    # Buscar en FAQs
    faq_keywords = [
        "horario", "gluten", "domicilio", "pago", "personalizado", "café", "fresco",
        "vegano", "reserva", "wi-fi", "bebida", "diabético", "ubicación"
    ]
    if any(keyword in consulta.lower() for keyword in faq_keywords):
        logging.debug("Buscando FAQ en vector store")
        results = vectordb.similarity_search(consulta, k=1, score_threshold=0.85)
        if results and results[0].page_content and "R:" in results[0].page_content:
            response = results[0].page_content.split("R:")[1].strip()
            logging.info(f"FAQ encontrada: {response}")
            db_manager.add_interaction(consulta, response)
            if session_id:
                get_session_history(session_id).add_user_message(consulta)
                get_session_history(session_id).add_ai_message(response)
            return response

    # Consultar el LLM con historial si aplica
    if llm_handler and session_id:
        logging.debug("Consultando LLM con historial")
        history = get_session_history(session_id)
        prompt = PromptTemplate(input_variables=["context", "question"], 
                              template="Contexto: {context}\nPregunta: {question}\nRespuesta:")
        chain = (
            RunnablePassthrough.assign(context=lambda x: "\n".join([f"User: {m.content}" for m in history.messages] + 
                                                                  [f"AI: {m.content}" for m in history.messages if m.type == "ai"])[-1000:])
            | prompt
            | llm_handler.llm  # Asumiendo que LLMHandler tiene un atributo 'llm'
            | str
        )
        response = chain.invoke({"question": consulta})
        logging.info(f"Respuesta del LLM: {response}")
        history.add_user_message(consulta)
        history.add_ai_message(response)
        db_manager.add_interaction(consulta, response)
        return response
    elif llm_handler:
        logging.debug("Consultando LLM sin historial")
        response = llm_handler.query_llm(consulta) or ""
        logging.info(f"Respuesta del LLM: {response}")
        db_manager.add_interaction(consulta, response)
        return response
    else:
        response = "No se pudo procesar la consulta debido a un error con el LLM."
        logging.error("LLMHandler no disponible")
        db_manager.add_interaction(consulta, response)
        return response

if __name__ == "__main__":
    """
    Prueba básica del agente con una lista de consultas de ejemplo.
    """
    test_consultas = [
        "¿Cuáles son los horarios de atención?",
        "¿Tienen opciones sin gluten?",
        "¿Tienen torta de chocolate?",
        "Quiero 2 croissants",
        "Quiero pedir una torta de chocolate",
        "¿Qué tal el clima?",
        "¿Puedes recomendar un postre?"
    ]
    for consulta in test_consultas:
        respuesta = buscar_respuesta(consulta, session_id="test_user")
        print(f"Consulta: {consulta}")
        print(f"Respuesta: {respuesta or 'No se encontró respuesta'}\n")