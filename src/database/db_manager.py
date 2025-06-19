import sqlite3
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/processed")
ORDERS_DB = DATA_DIR / "orders.db"
INTERACTIONS_DB = DATA_DIR / "interactions.db"

class DatabaseManager:
    """
    Gestiona las bases de datos SQLite para pedidos e interacciones.
    """
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_orders_db()
        self._init_interactions_db()

    def _init_orders_db(self):
        """Inicializa la base de datos de pedidos."""
        with sqlite3.connect(ORDERS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def _init_interactions_db(self):
        """Inicializa la base de datos de interacciones."""
        with sqlite3.connect(INTERACTIONS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_order(self, product_id: str, quantity: int, status: str = "pendiente") -> int:
        """
        Añade un nuevo pedido a orders.db.
        
        Args:
            product_id (str): ID del producto desde catalog.json.
            quantity (int): Cantidad solicitada.
            status (str): Estado del pedido (default: 'pendiente').
        
        Returns:
            int: ID del pedido creado.
        """
        with sqlite3.connect(ORDERS_DB) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO orders (product_id, quantity, timestamp, status) VALUES (?, ?, ?, ?)",
                (product_id, quantity, timestamp, status)
            )
            conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to insert order and retrieve lastrowid.")
            return cursor.lastrowid

    def add_interaction(self, query: str, response: str) -> int:
        """
        Añade una nueva interacción a interactions.db.
        
        Args:
            query (str): Consulta del usuario.
            response (str): Respuesta del agente.
        
        Returns:
            int: ID de la interacción creada.
        """
        with sqlite3.connect(INTERACTIONS_DB) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO interactions (query, response, timestamp) VALUES (?, ?, ?)",
                (query, response, timestamp)
            )
            conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to insert order and retrieve lastrowid.")
            return cursor.lastrowid

    def get_order(self, order_id: int) -> dict | None:
        """
        Obtiene un pedido por su ID.
        
        Args:
            order_id (int): ID del pedido.
        
        Returns:
            dict: Detalles del pedido, o None si no existe.
        """
        with sqlite3.connect(ORDERS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "order_id": row[0],
                    "product_id": row[1],
                    "quantity": row[2],
                    "timestamp": row[3],
                    "status": row[4]
                }
        return None

    def get_interaction(self, interaction_id: int) -> dict | None:
        """
        Obtiene una interacción por su ID.
        
        Args:
            interaction_id (int): ID de la interacción.
        
        Returns:
            dict: Detalles de la interacción, o None si no existe.
        """
        with sqlite3.connect(INTERACTIONS_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions WHERE interaction_id = ?", (interaction_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "interaction_id": row[0],
                    "query": row[1],
                    "response": row[2],
                    "timestamp": row[3]
                }
        return None

if __name__ == "__main__":
    # Prueba básica
    db = DatabaseManager()
    order_id = db.add_order(product_id="1", quantity=2)
    print(f"Pedido creado: ID {order_id}")
    interaction_id = db.add_interaction(query="¿Tienen torta?", response="Sí, tenemos Torta de Chocolate por $25000.")
    print(f"Interacción creada: ID {interaction_id}")
    order = db.get_order(order_id)
    print(f"Pedido recuperado: {order}")
    interaction = db.get_interaction(interaction_id)
    print(f"Interacción recuperada: {interaction}")