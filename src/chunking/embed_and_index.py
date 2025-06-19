from pathlib import Path
import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pydantic import SecretStr


load_dotenv(override=True)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
VECTOR_DIR = Path("data/processed/vectordb")

def load_chunks() -> list[Document]:
    """
    Carga los chunks desde chunks.jsonl como documentos de LangChain.
    """
    docs = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(
                page_content=rec["text"],
                metadata={
                    "id": rec["id"],
                    "source": rec["source"],
                }
            ))
    return docs

def run_embed_and_index():
    """
    Genera embeddings para los chunks y crea un vector store con FAISS.
    """
    # Obtener API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no encontrada en .env")

    # Cargar chunks
    docs = load_chunks()
    print(f"Cargando {len(docs)} documentos para embedding")

    # Instanciar embeddings
    embeddings = OpenAIEmbeddings(api_key=SecretStr(api_key))

    # Crear vector store
    vectordb = FAISS.from_documents(docs, embeddings)

    # Guardar en disco
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(VECTOR_DIR))
    print(f"Vector store guardado en: {VECTOR_DIR}")

if __name__ == "__main__":
    run_embed_and_index()