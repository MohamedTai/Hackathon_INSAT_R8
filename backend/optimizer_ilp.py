# backend/optimizer_ilp.py
import sqlite3
from datetime import date
import uuid

try:
    import pulp
except ImportError:
    pulp = None

def compute_priority(conn):
    cur = conn.cursor()
    md = {r["product_id"]: float(r["monthly_demand"])
          for r in cur.execute("SELECT product_id, monthly_demand FROM products").fetchall()}

    today = date.today().isoformat()
    rows = cur.execute("""
      SELECT ol.product_id, o.due_date, o.priority, ol.qty
      FROM order_lines ol JOIN orders o ON o.order_id=ol.order_id
      WHERE o.due_date >= ?
    """, (today,)).fetchall()

    urg = {pid: 0.0 for pid in md}
    for r in rows:
        pid = r["product_id"]
        due = date.fromisoformat(r["due_date"])
        days = max((due - date.today()).days, 1)
        pr = int(r["priority"])
        qty = float(r["qty"])
        urg[pid] += (qty / days) * (4 - pr)

    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    md_n, urg_n = norm(md), norm(urg)
    return {pid: 0.6 * md_n[pid] + 0.4 * urg_n[pid] for pid in md}

def choose_slot_heuristic(conn, L, W, H, weight, product_id=None):
    """Fallback si PuLP indisponible."""
    cur = conn.cursor()
    slots = cur.execute("""
      SELECT slot_id,L,W,H,max_weight,distance,is_occupied
      FROM slots WHERE is_occupied=0
    """).fetchall()

    best = None
    best_score = 1e18

    P = 0.5
    if product_id:
        Pmap = compute_priority(conn)
        P = Pmap.get(product_id, 0.5)
    alpha = 0.2 + 0.6 * P

    for s in slots:
        if s["L"] >= L and s["W"] >= W and s["H"] >= H and s["max_weight"] >= weight:
            waste = (s["L"]*s["W"]*s["H"]) - (L*W*H)
            score = alpha * float(s["distance"]) + (1-alpha) * waste
            if score < best_score:
                best_score = score
                best = s["slot_id"]
    return best

def choose_slot_ilp(conn, L, W, H, weight, product_id):
    if pulp is None:
        return None

    Pmap = compute_priority(conn)
    P = Pmap.get(product_id, 0.5)
    alpha = 0.2 + 0.6 * P

    cur = conn.cursor()
    slots = cur.execute("""
      SELECT slot_id,L,W,H,max_weight,distance
      FROM slots WHERE is_occupied=0
    """).fetchall()
    if not slots:
        return None

    feasible = []
    for s in slots:
        if s["L"] >= L and s["W"] >= W and s["H"] >= H and s["max_weight"] >= weight:
            feasible.append(s)
    if not feasible:
        return None

    # ILP trivial: choisir 1 slot parmi faisables, minimiser score
    prob = pulp.LpProblem("choose_slot", pulp.LpMinimize)
    x = {s["slot_id"]: pulp.LpVariable(f"x_{s['slot_id']}", 0, 1, cat="Binary") for s in feasible}

    prob += pulp.lpSum([x[s["slot_id"]] for s in feasible]) == 1

    obj = []
    for s in feasible:
        waste = (s["L"]*s["W"]*s["H"]) - (L*W*H)
        score = alpha * float(s["distance"]) + (1-alpha) * float(waste)
        obj.append(score * x[s["slot_id"]])
    prob += pulp.lpSum(obj)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    for sid, var in x.items():
        if var.value() > 0.5:
            return sid
    return None

def choose_best_slot(conn, L, W, H, weight, product_id):
    sid = choose_slot_ilp(conn, L, W, H, weight, product_id)
    if sid:
        return sid
    return choose_slot_heuristic(conn, L, W, H, weight, product_id)
