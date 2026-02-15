# backend/seed_real_db.py
import random
from datetime import date, timedelta
from backend.schema import get_conn, init_db

def seed():
    init_db()
    conn = get_conn()
    cur = conn.cursor()

    # Clear (safe)
    cur.execute("DELETE FROM order_lines;")
    cur.execute("DELETE FROM orders;")
    cur.execute("DELETE FROM inventory;")
    cur.execute("DELETE FROM products;")
    cur.execute("DELETE FROM slots;")

    # Slots (30)
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

    # Products (20)
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

    # Initial inventory (empty) -> optional
    # Orders (8)
    random.seed(33)
    today = date.today()
    for i in range(1, 9):
        oid = f"ORD-{i:04d}"
        due = today + timedelta(days=random.choice([3, 7, 10, 14, 21, 30]))
        pr = random.choice([1, 2, 3])
        cur.execute("INSERT INTO orders(order_id,due_date,priority) VALUES (?,?,?)", (oid, due.isoformat(), pr))

        # 2-4 lines
        pids = [f"MAT-{j:05d}" for j in random.sample(range(1, 21), random.choice([2, 3, 4]))]
        for pid in pids:
            qty = random.choice([5, 10, 15, 20, 30])
            cur.execute("INSERT INTO order_lines(order_id,product_id,qty) VALUES (?,?,?)", (oid, pid, qty))

    conn.commit()
    conn.close()
    print("✅ Seed done.")

if __name__ == "__main__":
    seed()
