from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse # type: ignore
from twilio.rest import Client # type: ignore
from dotenv import load_dotenv
import os
from src.agent.agent_core import buscar_respuesta

# Cargar variables de entorno
load_dotenv(override=True)

app = Flask(__name__)

# Configurar cliente de Twilio
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
if not account_sid or not auth_token:
    raise ValueError("TWILIO_ACCOUNT_SID o TWILIO_AUTH_TOKEN no encontrados en .env")
client = Client(account_sid, auth_token)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """
    Maneja mensajes entrantes de WhatsApp y responde usando el agente.
    """
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")

    # Procesar consulta con el agente
    response_text = buscar_respuesta(incoming_msg, from_number) or "Lo siento, no entendí tu mensaje."

    # Crear respuesta TwiML
    resp = MessagingResponse()
    resp.message(response_text)

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)