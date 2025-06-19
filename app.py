import sqlite3
import streamlit as st
from pathlib import Path

# Definir rutas de las bases de datos
DATA_DIR = Path("data/processed")
INTERACTIONS_DB = DATA_DIR / "interactions.db"
ORDERS_DB = DATA_DIR / "orders.db"

# Título de la aplicación
st.title("Revisión de Conversaciones y Pedidos")

# Mostrar conversaciones
st.header("Conversaciones")
conn = sqlite3.connect(INTERACTIONS_DB)
cursor = conn.cursor()
cursor.execute("SELECT interaction_id, query, response, timestamp FROM interactions ORDER BY timestamp DESC")
conversaciones = cursor.fetchall()
if conversaciones:
    for conv in conversaciones:
        st.write(f"**ID:** {conv[0]}")
        st.write(f"**Consulta:** {conv[1]}")
        st.write(f"**Respuesta:** {conv[2]}")
        st.write(f"**Fecha:** {conv[3]}")
        st.write("---")
else:
    st.write("No hay conversaciones registradas.")
conn.close()

# Mostrar pedidos
st.header("Pedidos Registrados")
conn = sqlite3.connect(ORDERS_DB)
cursor = conn.cursor()
cursor.execute("SELECT order_id, product_id, quantity, timestamp, status FROM orders ORDER BY timestamp DESC")
pedidos = cursor.fetchall()
if pedidos:
    for ped in pedidos:
        st.write(f"**ID del Pedido:** {ped[0]}")
        st.write(f"**ID del Producto:** {ped[1]}")
        st.write(f"**Cantidad:** {ped[2]}")
        st.write(f"**Fecha:** {ped[3]}")
        st.write(f"**Estado:** {ped[4]}")
        st.write("---")
else:
    st.write("No hay pedidos registrados.")
conn.close()