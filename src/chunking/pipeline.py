import subprocess
import sys
from .chunk import create_chunks
from .embed_and_index import run_embed_and_index

def run_pipeline(test_search: bool = False):
    """
    Ejecuta el pipeline de chunking, embeddings y búsqueda (opcional).
    
    Args:
        test_search (bool): Si True, ejecuta pruebas de búsqueda después de generar el vector store.
    """
    print("Iniciando pipeline de chunking y embeddings...")

    # Paso 1: Generar chunks
    print("Ejecutando chunking...")
    try:
        create_chunks()
    except Exception as e:
        print(f"Error en chunking: {e}")
        return

    # Paso 2: Generar embeddings y vector store
    print("Generando embeddings y vector store...")
    try:
        run_embed_and_index()
    except Exception as e:
        print(f"Error en generación de embeddings: {e}")
        return

    # Paso 3: Probar búsqueda (opcional)
    if test_search:
        print("Ejecutando pruebas de búsqueda...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.agent.agent_core"],
                capture_output=True,
                text=True
            )
            print("Resultados de búsqueda:")
            print(result.stdout)
            if result.stderr:
                print(f"Errores en búsqueda: {result.stderr}")
        except Exception as e:
            print(f"Error en pruebas de búsqueda: {e}")

    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    # Ejecutar el pipeline con pruebas de búsqueda
    run_pipeline(test_search=True)