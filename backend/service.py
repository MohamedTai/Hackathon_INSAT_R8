# backend/service.py
from datetime import date
from backend.schema import get_conn
from backend.optimizer_ilp import choose_best_slot

def inbound(product_id: str, qty: float, L: float, W: float, H: float, total_weight: float):
    conn = get_conn()
    cur = conn.cursor()

    # choose slot
    sid = choose_best_slot(conn, L, W, H, total_weight, product_id)
    if sid is None:
        conn.close()
        return {"ok": False, "reason": "No eligible vacant slot (size/weight constraints)."}

    # mark slot occupied
    cur.execute("UPDATE slots SET is_occupied=1 WHERE slot_id=?", (sid,))

    # upsert inventory
    today = date.today().isoformat()
    existing = cur.execute("SELECT * FROM inventory WHERE product_id=?", (product_id,)).fetchone()

    if existing:
        new_qty = float(existing["qty"]) + qty
        new_w = float(existing["total_weight"]) + total_weight
        cur.execute("""
          UPDATE inventory SET qty=?, total_weight=?, L=?, W=?, H=?, slot_id=?, updated_at=?
          WHERE product_id=?
        """, (new_qty, new_w, L, W, H, sid, today, product_id))
    else:
        cur.execute("""
          INSERT INTO inventory(product_id,qty,total_weight,L,W,H,slot_id,updated_at)
          VALUES (?,?,?,?,?,?,?,?)
        """, (product_id, qty, total_weight, L, W, H, sid, today))

    conn.commit()
    conn.close()
    return {"ok": True, "slot_id": sid}

def outbound(product_id: str, qty_out: float):
    conn = get_conn()
    cur = conn.cursor()

    row = cur.execute("SELECT * FROM inventory WHERE product_id=?", (product_id,)).fetchone()
    if not row:
        conn.close()
        return {"ok": False, "reason": "Product not found in inventory."}

    new_qty = float(row["qty"]) - qty_out
    if new_qty < 0:
        conn.close()
        return {"ok": False, "reason": "Not enough stock."}

    sid = row["slot_id"]
    today = date.today().isoformat()

    if new_qty == 0:
        # delete & free slot
        cur.execute("DELETE FROM inventory WHERE product_id=?", (product_id,))
        if sid:
            cur.execute("UPDATE slots SET is_occupied=0 WHERE slot_id=?", (sid,))
    else:
        # proportionnel (règle de trois) sur poids
        old_qty = float(row["qty"])
        old_w = float(row["total_weight"])
        new_w = old_w * (new_qty / old_qty) if old_qty > 0 else 0.0
        cur.execute("UPDATE inventory SET qty=?, total_weight=?, updated_at=? WHERE product_id=?",
                    (new_qty, new_w, today, product_id))

    conn.commit()
    conn.close()
    return {"ok": True, "slot_id": sid, "remaining_qty": new_qty}
