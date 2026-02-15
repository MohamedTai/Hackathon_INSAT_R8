from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time
import sqlite3
import pulp

from backend.db import get_conn


# ------------------------------
# Logique existante (heuristique)
# ------------------------------
def choose_best_slot(L: float, W: float, H: float, est_weight: float = 0.0) -> Optional[str]:
    """
    Choisit un slot vacant qui respecte contraintes et minimise le "waste" (volume slot - volume item).
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT slot_id, L, W, H, max_weight, distance_to_shipping
        FROM slots
        WHERE occupied = 0
          AND L >= ? AND W >= ? AND H >= ?
          AND max_weight >= ?
        ORDER BY (L*W*H) ASC
    """, (L, W, H, est_weight))

    best_slot = None
    best_score = None
    item_vol = L * W * H

    for r in cur.fetchall():
        slot_vol = float(r["L"]) * float(r["W"]) * float(r["H"])
        waste = max(0.0, slot_vol - item_vol)
        dist = float(r["distance_to_shipping"] or 0.0)

        # score simple: waste + petit poids distance
        score = waste + 0.05 * dist

        if best_score is None or score < best_score:
            best_score = score
            best_slot = r["slot_id"]

    conn.close()
    return best_slot


def set_slot(slot_id: str, occupied: bool, item_id: Optional[str], weight: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE slots
        SET occupied = ?, current_item_id = ?, current_weight = ?
        WHERE slot_id = ?
    """, (1 if occupied else 0, item_id, float(weight), slot_id))
    conn.commit()
    conn.close()


# ------------------------------
# Optimisation ILP (slotting)
# ------------------------------
@dataclass(frozen=True)
class Product:
    product_id: str
    L: float
    W: float
    H: float
    weight: float
    monthly_consumption: float

    @property
    def volume(self) -> float:
        return self.L * self.W * self.H


@dataclass(frozen=True)
class Slot:
    slot_id: str
    L: float
    W: float
    H: float
    max_weight: float
    distance_to_shipping: float

    @property
    def volume(self) -> float:
        return self.L * self.W * self.H


def _normalize(values: Dict[str, float], eps: float = 1e-9) -> Dict[str, float]:
    if not values:
        return {}
    mx = max(values.values())
    return {k: v / (mx + eps) for k, v in values.items()}


def _compatible(p: Product, s: Slot) -> bool:
    return (p.L <= s.L and p.W <= s.W and p.H <= s.H and p.weight <= s.max_weight)


def _fetch_products(conn: sqlite3.Connection) -> List[Product]:
    cur = conn.cursor()
    cur.execute("""SELECT product_id, L, W, H, weight, monthly_consumption FROM products""")
    return [
        Product(
            product_id=r["product_id"],
            L=float(r["L"]), W=float(r["W"]), H=float(r["H"]),
            weight=float(r["weight"] or 0.0),
            monthly_consumption=float(r["monthly_consumption"] or 0.0),
        )
        for r in cur.fetchall()
    ]


def _fetch_slots(conn: sqlite3.Connection, only_vacant: bool = True) -> List[Slot]:
    cur = conn.cursor()
    if only_vacant:
        cur.execute("""SELECT slot_id, L, W, H, max_weight, distance_to_shipping FROM slots WHERE occupied = 0""")
    else:
        cur.execute("""SELECT slot_id, L, W, H, max_weight, distance_to_shipping FROM slots""")

    return [
        Slot(
            slot_id=r["slot_id"],
            L=float(r["L"]), W=float(r["W"]), H=float(r["H"]),
            max_weight=float(r["max_weight"] or 0.0),
            distance_to_shipping=float(r["distance_to_shipping"] or 0.0),
        )
        for r in cur.fetchall()
    ]


def _fetch_backlog(conn: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    """
    backlog[product_id] = {open_qty, urgent_qty}
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
          ol.product_id AS product_id,
          SUM(CASE WHEN o.status = 'OPEN' THEN ol.quantity ELSE 0 END) AS open_qty,
          SUM(CASE WHEN o.status = 'OPEN' AND o.urgency = 'HIGH' THEN ol.quantity ELSE 0 END) AS urgent_qty
        FROM order_lines ol
        JOIN orders o ON o.order_id = ol.order_id
        GROUP BY ol.product_id
    """)
    out: Dict[str, Dict[str, float]] = {}
    for r in cur.fetchall():
        pid = r["product_id"]
        out[pid] = {
            "open_qty": float(r["open_qty"] or 0.0),
            "urgent_qty": float(r["urgent_qty"] or 0.0),
        }
    return out


def optimize_slotting_ilp(
    alpha: float = 1.0,   # distance * demand
    beta: float = 2.0,    # distance * urgent
    gamma: float = 0.3,   # waste
    time_limit_s: int = 20,
    only_vacant_slots: bool = True,
) -> Dict:
    """
    Affecte chaque produit à un slot compatible en minimisant:
      alpha*dist*demand + beta*dist*urgent + gamma*waste
    Écrit les résultats dans assignments (historique) et renvoie la solution.
    """
    conn = get_conn()
    try:
        products = _fetch_products(conn)
        slots = _fetch_slots(conn, only_vacant=only_vacant_slots)
        backlog = _fetch_backlog(conn)

        if not products:
            return {"status": "NO_PRODUCTS", "message": "products table is empty", "assignments": []}
        if not slots:
            return {"status": "NO_SLOTS", "message": "no eligible slots (vacant?)", "assignments": []}

        demand_raw = {p.product_id: float(p.monthly_consumption or 0.0) for p in products}
        urgent_raw = {p.product_id: float(backlog.get(p.product_id, {}).get("urgent_qty", 0.0) or 0.0) for p in products}

        demand = _normalize(demand_raw)
        urgent = _normalize(urgent_raw)

        # cost only for compatible pairs
        cost: Dict[Tuple[str, str], float] = {}
        for p in products:
            for s in slots:
                if not _compatible(p, s):
                    continue

                dist = float(s.distance_to_shipping or 0.0)
                waste = 0.0
                if s.volume > 0:
                    waste = max(0.0, 1.0 - (p.volume / s.volume))

                cost[(p.product_id, s.slot_id)] = (
                    alpha * dist * float(demand.get(p.product_id, 0.0))
                    + beta * dist * float(urgent.get(p.product_id, 0.0))
                    + gamma * waste
                )

        # ILP
        keys = list(cost.keys())
        x = pulp.LpVariable.dicts("x", keys, lowBound=0, upBound=1, cat="Binary")
        prob = pulp.LpProblem("SmartWarehouseSlotting", pulp.LpMinimize)
        prob += pulp.lpSum(cost[k] * x[k] for k in keys)

        product_ids = [p.product_id for p in products]
        slot_ids = [s.slot_id for s in slots]

        # each product exactly 1 slot
        for pid in product_ids:
            possible = [sid for sid in slot_ids if (pid, sid) in x]
            if not possible:
                raise RuntimeError(f"Product {pid} has no compatible slot (infeasible).")
            prob += pulp.lpSum(x[(pid, sid)] for sid in possible) == 1, f"prod_{pid}"

        # each slot at most 1 product
        for sid in slot_ids:
            possible = [pid for pid in product_ids if (pid, sid) in x]
            if possible:
                prob += pulp.lpSum(x[(pid, sid)] for pid in possible) <= 1, f"slot_{sid}"

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
        status_code = prob.solve(solver)
        status = pulp.LpStatus[status_code]
        obj_val = float(pulp.value(prob.objective) or 0.0)

        assignment: Dict[str, str] = {}
        for (pid, sid) in keys:
            if pulp.value(x[(pid, sid)]) and pulp.value(x[(pid, sid)]) > 0.5:
                assignment[pid] = sid

        # persist
        run_id = f"run_{int(time.time()*1000)}"
        created_ts_ms = int(time.time() * 1000)

        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO assignments(run_id, created_ts_ms, product_id, slot_id, objective_value)
            VALUES (?,?,?,?,?)
        """, [(run_id, created_ts_ms, pid, sid, obj_val) for pid, sid in assignment.items()])
        conn.commit()

        return {
            "run_id": run_id,
            "status": status,
            "objective_value": obj_val,
            "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "time_limit_s": time_limit_s},
            "assignments": [{"product_id": pid, "slot_id": sid} for pid, sid in assignment.items()],
        }

    finally:
        conn.close()
