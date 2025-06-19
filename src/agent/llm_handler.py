import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.database.db_manager import DatabaseManager

# Cargar variables de entorno
load_dotenv(override=True)

class LLMHandler:
    """
    Maneja consultas al LLM de OpenAI y registra interacciones.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en .env")
        self.model = "gpt-4o" 
        self.db_manager = DatabaseManager()
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=0.7
        )

    def query_llm(self, consulta: str) -> str | None:
        """
        Envía una consulta al LLM de OpenAI y registra la interacción.
        
        Args:
            consulta (str): Consulta del usuario.
        
        Returns:
            str: Respuesta del LLM, o None si hay un error.
        """
        if not consulta or not isinstance(consulta, str):
            return None

        system_prompt = (
            "Eres un asistente amigable para una panadería en Bogotá, Colombia. Responde en español, con un tono cálido y profesional. "
            "Si la consulta es sobre productos, recomienda opciones de nuestro catálogo: Pan Integral ($5000), Torta de Chocolate ($25000), "
            "Croissant ($3000), Galletas de Avena ($2000). Para preguntas generales, ofrece respuestas útiles y, si no sabes algo "
            "(e.g., clima), sugiere algo relacionado con la panadería."
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": consulta}
            ])
            answer = response.content.strip()
            # Guardar interacción
            self.db_manager.add_interaction(consulta, answer)
            return answer
        except Exception as e:
            error_msg = f"Error al consultar el LLM: {e}"
            self.db_manager.add_interaction(consulta, error_msg)
            return error_msg

if __name__ == "__main__":
    # Prueba básica
    try:
        llm = LLMHandler()
        test_consultas = [
            "¿Qué tal el clima?",
            "¿Puedes recomendar un postre?"
        ]
        for consulta in test_consultas:
            respuesta = llm.query_llm(consulta)
            print(f"Consulta: {consulta}")
            print(f"Respuesta: {respuesta or 'No se obtuvo respuesta'}\n")
    except ValueError as e:
        print(f"Error: {e}")