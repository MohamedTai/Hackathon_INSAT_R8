# backend/schema.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "warehouse.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Slots
    cur.execute("""
    CREATE TABLE IF NOT EXISTS slots(
      slot_id TEXT PRIMARY KEY,
      L REAL NOT NULL,
      W REAL NOT NULL,
      H REAL NOT NULL,
      max_weight REAL NOT NULL,
      distance REAL NOT NULL DEFAULT 20,
      zone TEXT DEFAULT 'Z2',
      is_occupied INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Produits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products(
      product_id TEXT PRIMARY KEY,
      name TEXT,
      unit_weight REAL NOT NULL,
      unit_L REAL NOT NULL,
      unit_W REAL NOT NULL,
      unit_H REAL NOT NULL,
      monthly_demand REAL NOT NULL DEFAULT 10
    );
    """)

    # Stock
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory(
      product_id TEXT PRIMARY KEY,
      qty REAL NOT NULL,
      total_weight REAL NOT NULL,
      L REAL NOT NULL,
      W REAL NOT NULL,
      H REAL NOT NULL,
      slot_id TEXT,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(product_id) REFERENCES products(product_id),
      FOREIGN KEY(slot_id) REFERENCES slots(slot_id)
    );
    """)

    # Commandes (pour priorité)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders(
      order_id TEXT PRIMARY KEY,
      due_date TEXT NOT NULL,
      priority INTEGER NOT NULL DEFAULT 2
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS order_lines(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      order_id TEXT NOT NULL,
      product_id TEXT NOT NULL,
      qty REAL NOT NULL,
      FOREIGN KEY(order_id) REFERENCES orders(order_id),
      FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized:", DB_PATH)
