import sqlite3
from pathlib import Path
from datetime import date
import random

import pandas as pd
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "warehouse.db"


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

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

    # (Optionnel) commandes pour "order pressure" plus tard
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


def db_has_data():
    conn = get_conn()
    cur = conn.cursor()
    n_slots = cur.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    n_prod = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    conn.close()
    return (n_slots >= 30) and (n_prod >= 20)


def seed_db():
    """20 produits + 30 slots (DB réelle warehouse.db)"""
    conn = get_conn()
    cur = conn.cursor()

    # clear
    cur.execute("DELETE FROM order_lines;")
    cur.execute("DELETE FROM orders;")
    cur.execute("DELETE FROM inventory;")
    cur.execute("DELETE FROM products;")
    cur.execute("DELETE FROM slots;")

    # slots
    random.seed(11)
    zones = [("Z1", 5), ("Z2", 20), ("Z3", 50)]
    for i in range(1, 31):
        slot_id = f"S{i:02d}"
        zone, base = random.choice(zones)
        L = random.choice([1.2, 1.4, 1.6, 2.0])
        W = random.choice([0.8, 1.0, 1.2])
        H = random.choice([1.5, 1.8, 2.0, 2.4])
        maxw = random.choice([300, 500, 800, 1200])
        dist = base + random.choice([0, 2, 5, 8, 10])
        cur.execute("""
          INSERT INTO slots(slot_id,L,W,H,max_weight,distance,zone,is_occupied)
          VALUES (?,?,?,?,?,?,?,0)
        """, (slot_id, L, W, H, maxw, dist, zone))

    # products
    random.seed(7)
    for i in range(1, 21):
        pid = f"MAT-{i:05d}"
        name = f"Product {i:02d}"
        unit_L = random.choice([0.4, 0.6, 0.8, 1.0, 1.2])
        unit_W = random.choice([0.3, 0.4, 0.6, 0.8])
        unit_H = random.choice([0.3, 0.5, 0.7, 1.0, 1.4, 1.6])
        unit_weight = random.choice([5, 10, 15, 20, 25, 30, 40])
        monthly = random.choice([5, 10, 20, 30, 50, 80, 120])
        cur.execute("""
          INSERT INTO products(product_id,name,unit_weight,unit_L,unit_W,unit_H,monthly_demand)
          VALUES (?,?,?,?,?,?,?)
        """, (pid, name, unit_weight, unit_L, unit_W, unit_H, monthly))

    conn.commit()
    conn.close()


# -----------------------------
# Optimisation (ILP si dispo, sinon heuristique)
# -----------------------------
def compute_priority(conn):
    """Priorité P(product) = 0.6 * monthly_demand_norm + 0.4 * urgency_norm"""
    cur = conn.cursor()
    md = {r["product_id"]: float(r["monthly_demand"])
          for r in cur.execute("SELECT product_id, monthly_demand FROM products").fetchall()}

    # urgence depuis commandes (optionnel)
    today = date.today().isoformat()
    rows = cur.execute("""
      SELECT ol.product_id, o.due_date, o.priority, ol.qty
      FROM order_lines ol JOIN orders o ON o.order_id=ol.order_id
      WHERE o.due_date >= ?
    """, (today,)).fetchall()

    urg = {pid: 0.0 for pid in md}
    for r in rows:
        pid = r["product_id"]
        # simple score urgence
        urg[pid] += float(r["qty"]) * (4 - int(r["priority"]))

    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    md_n, urg_n = norm(md), norm(urg)
    return {pid: 0.6 * md_n[pid] + 0.4 * urg_n[pid] for pid in md}


def choose_slot_heuristic(conn, L, W, H, weight, product_id):
    cur = conn.cursor()
    slots = cur.execute("""
      SELECT slot_id,L,W,H,max_weight,distance
      FROM slots WHERE is_occupied=0
    """).fetchall()

    if not slots:
        return None, "No vacant slots."

    Pmap = compute_priority(conn)
    P = Pmap.get(product_id, 0.5)
    alpha = 0.2 + 0.6 * P  # 0.2..0.8

    best_sid, best_score = None, 1e18
    v_item = L * W * H

    for s in slots:
        if s["L"] >= L and s["W"] >= W and s["H"] >= H and s["max_weight"] >= weight:
            v_slot = s["L"] * s["W"] * s["H"]
            waste = max(v_slot - v_item, 0.0)
            score = alpha * float(s["distance"]) + (1 - alpha) * float(waste)
            if score < best_score:
                best_score = score
                best_sid = s["slot_id"]

    if best_sid is None:
        return None, "No eligible slot (size/weight constraints)."

    return best_sid, None


def choose_best_slot(conn, L, W, H, weight, product_id):
    # ILP (si tu veux l’activer plus tard) -> pour MVP on garde heuristique stable
    return choose_slot_heuristic(conn, L, W, H, weight, product_id)


# -----------------------------
# IN / OUT services
# -----------------------------
def inbound(product_id: str, qty: float, L: float, W: float, H: float, total_weight: float):
    conn = get_conn()
    cur = conn.cursor()

    sid, err = choose_best_slot(conn, L, W, H, total_weight, product_id)
    if sid is None:
        conn.close()
        return False, err

    # occuper slot
    cur.execute("UPDATE slots SET is_occupied=1 WHERE slot_id=?", (sid,))

    # upsert inventory
    today = date.today().isoformat()
    existing = cur.execute("SELECT * FROM inventory WHERE product_id=?", (product_id,)).fetchone()

    if existing:
        new_qty = float(existing["qty"]) + qty
        new_w = float(existing["total_weight"]) + total_weight
        cur.execute("""
          UPDATE inventory
          SET qty=?, total_weight=?, L=?, W=?, H=?, slot_id=?, updated_at=?
          WHERE product_id=?
        """, (new_qty, new_w, L, W, H, sid, today, product_id))
    else:
        cur.execute("""
          INSERT INTO inventory(product_id,qty,total_weight,L,W,H,slot_id,updated_at)
          VALUES (?,?,?,?,?,?,?,?)
        """, (product_id, qty, total_weight, L, W, H, sid, today))

    conn.commit()
    conn.close()
    return True, sid


def outbound(product_id: str, qty_out: float):
    conn = get_conn()
    cur = conn.cursor()

    row = cur.execute("SELECT * FROM inventory WHERE product_id=?", (product_id,)).fetchone()
    if not row:
        conn.close()
        return False, "Product not found."

    old_qty = float(row["qty"])
    if qty_out > old_qty:
        conn.close()
        return False, "Not enough stock."

    new_qty = old_qty - qty_out
    sid = row["slot_id"]
    today = date.today().isoformat()

    if new_qty == 0:
        cur.execute("DELETE FROM inventory WHERE product_id=?", (product_id,))
        if sid:
            cur.execute("UPDATE slots SET is_occupied=0 WHERE slot_id=?", (sid,))
    else:
        # règle de trois sur le poids total
        old_w = float(row["total_weight"])
        new_w = old_w * (new_qty / old_qty) if old_qty > 0 else 0.0
        cur.execute("UPDATE inventory SET qty=?, total_weight=?, updated_at=? WHERE product_id=?",
                    (new_qty, new_w, today, product_id))

    conn.commit()
    conn.close()
    return True, {"slot_id": sid, "remaining_qty": new_qty}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Warehouse MVP (Streamlit)", layout="wide")
st.title("📦 Warehouse MVP — Gestion Stock + Slotting (sans caméras)")

init_db()

with st.sidebar:
    st.header("⚙️ Setup")
    st.write(f"DB: `{DB_PATH.name}`")
    if st.button("🌱 Seed DB (20 produits, 30 slots)"):
        seed_db()
        st.success("Seed OK. Recharge la page si besoin.")

if not db_has_data():
    st.warning("DB vide. Clique sur **Seed DB** dans la sidebar pour créer 20 produits et 30 slots.")

tab_in, tab_out, tab_view = st.tabs(["IN (Entrée)", "OUT (Sortie)", "Dashboard"])

with tab_in:
    st.subheader("Entrée Marchandise (simulation, sans caméras)")
    col1, col2, col3 = st.columns(3)

    with col1:
        product_id = st.text_input("Product ID", value="MAT-00001")
        qty = st.number_input("Quantité", min_value=1.0, value=10.0, step=1.0)
        total_weight = st.number_input("Poids total (kg)", min_value=1.0, value=200.0, step=1.0)

    with col2:
        L = st.number_input("Longueur L (m)", min_value=0.01, value=1.200, step=0.001, format="%.3f")
        W = st.number_input("Largeur W (m)", min_value=0.01, value=0.800, step=0.001, format="%.3f")
        H = st.number_input("Hauteur H (m)", min_value=0.01, value=1.500, step=0.001, format="%.3f")

    with col3:
        st.caption("Le système choisit un slot vacant compatible (dimensions & poids) et met à jour l’inventaire.")
        if st.button("✅ Valider Entrée", type="primary"):
            ok, res = inbound(product_id, float(qty), float(L), float(W), float(H), float(total_weight))
            if ok:
                st.success(f"Entrée OK — Slot attribué: **{res}**")
            else:
                st.error(res)

with tab_out:
    st.subheader("Sortie Marchandise (simulation, sans caméras)")
    product_id_out = st.text_input("Product ID", value="MAT-00001", key="pid_out")
    qty_out = st.number_input("Quantité sortie", min_value=1.0, value=5.0, step=1.0)

    if st.button("✅ Valider Sortie", type="primary"):
        ok, res = outbound(product_id_out, float(qty_out))
        if ok:
            st.success(f"Sortie OK — Slot: **{res['slot_id']}**, Reste: **{res['remaining_qty']}**")
        else:
            st.error(res)

with tab_view:
    st.subheader("📌 Slots & 📦 Inventory")

    conn = get_conn()
    df_slots = pd.read_sql_query("SELECT * FROM slots ORDER BY slot_id", conn)
    df_inv = pd.read_sql_query("SELECT * FROM inventory ORDER BY product_id", conn)
    df_prod = pd.read_sql_query("SELECT * FROM products ORDER BY product_id", conn)
    conn.close()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Slots")
        st.dataframe(df_slots, width="stretch", height=420)
    with c2:
        st.markdown("### Inventory")
        st.dataframe(df_inv, width="stretch", height=420)

    st.markdown("### Products (référence)")
    st.dataframe(df_prod, width="stretch", height=260)
