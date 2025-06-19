from pathlib import Path
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pydantic import SecretStr

# Cargar variables de entorno desde .env
load_dotenv(override=True)

VECTOR_DIR = Path("data/processed/vectordb")

def buscar_respuesta(consulta: str) -> str | None:
    """
    Busca una respuesta en las FAQs usando un vector store con embeddings.
    
    Args:
        consulta (str): Texto de la consulta del usuario.
    
    Returns:
        str: Respuesta correspondiente si se encuentra una coincidencia.
        None: Si no se encuentra ninguna respuesta.
    """
    if not consulta or not isinstance(consulta, str):
        return None

    # Obtener API key
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

    # Buscar documentos similares
    results = vectordb.similarity_search(consulta, k=1)
    if results and results[0].page_content:
        text = results[0].page_content
        if "R:" in text:
            return text.split("R:")[1].strip()
        return text

    return None

if __name__ == "__main__":
    test_consultas = [
        "¿Cuáles son los horarios de atención?",
        "¿Qué horarios de atención tienen?",
        "¿En qué horario están abiertos?",
        "¿Tienen opciones sin gluten?",
        "¿Qué tal el clima?"
    ]
    for consulta in test_consultas:
        respuesta = buscar_respuesta(consulta)
        print(f"Consulta: {consulta}")
        print(f"Respuesta: {respuesta or 'No se encontró respuesta'}\n")