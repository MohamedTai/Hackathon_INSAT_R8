import sqlite3
from pathlib import Path

DB_PATH = Path("warehouse.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return column in cols


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str, default_sql: str = "0"):
    if not _column_exists(conn, table, column):
        cur = conn.cursor()
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type} NOT NULL DEFAULT {default_sql}")
        conn.commit()


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # -------------------- EXISTANT --------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_ms INTEGER NOT NULL,
      direction TEXT NOT NULL,
      source TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS slots (
      slot_id TEXT PRIMARY KEY,
      L REAL NOT NULL,
      W REAL NOT NULL,
      H REAL NOT NULL,
      max_weight REAL NOT NULL,
      occupied INTEGER NOT NULL DEFAULT 0,
      current_item_id TEXT,
      current_weight REAL NOT NULL DEFAULT 0
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS movements (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_ms INTEGER NOT NULL,
      item_id TEXT,
      slot_id TEXT,
      direction TEXT NOT NULL,
      L REAL, W REAL, H REAL,
      est_qty REAL,
      est_weight REAL
    );
    """)

    conn.commit()

    # -------------------- NOUVEAU (slotting intelligent) --------------------
    # Ajout d'une distance à l'expédition (ou station d'input/output)
    _ensure_column(conn, "slots", "distance_to_shipping", "REAL", default_sql="0")

    # Catalogue produits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
      product_id TEXT PRIMARY KEY,
      name TEXT,
      L REAL NOT NULL,
      W REAL NOT NULL,
      H REAL NOT NULL,
      weight REAL NOT NULL DEFAULT 0,
      monthly_consumption REAL NOT NULL DEFAULT 0
    );
    """)

    # Inventaire (optionnel pour plus tard)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      product_id TEXT NOT NULL,
      qty REAL NOT NULL DEFAULT 0,
      FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
    """)

    # Commandes
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
      order_id TEXT PRIMARY KEY,
      status TEXT NOT NULL,         -- OPEN / CLOSED
      urgency TEXT NOT NULL,        -- LOW / MEDIUM / HIGH
      created_ts_ms INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS order_lines (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      order_id TEXT NOT NULL,
      product_id TEXT NOT NULL,
      quantity REAL NOT NULL,
      FOREIGN KEY(order_id) REFERENCES orders(order_id),
      FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
    """)

    # Résultats optimisation (historisé)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS assignments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      created_ts_ms INTEGER NOT NULL,
      product_id TEXT NOT NULL,
      slot_id TEXT NOT NULL,
      objective_value REAL NOT NULL,
      FOREIGN KEY(product_id) REFERENCES products(product_id),
      FOREIGN KEY(slot_id) REFERENCES slots(slot_id)
    );
    """)

    conn.commit()
    conn.close()
