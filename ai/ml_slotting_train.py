import sqlite3
from dataclasses import dataclass
from pathlib import Path
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

DB_PATH = Path("slotting_synth.db")
MODEL_PATH = Path("slotting_model.joblib")

# ----------------------------
# Config
# ----------------------------
@dataclass
class SynthConfig:
    n_slots: int = 30
    n_items: int = 2000   # dataset items
    seed: int = 42

cfg = SynthConfig()

# ----------------------------
# Helpers
# ----------------------------
def m3(Lm, Wm, Hm):
    return float(Lm) * float(Wm) * float(Hm)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def round_m(x):
    # arrondi à 3 décimales en mètres
    return round(float(x), 3)

# ----------------------------
# DB: create + seed
# ----------------------------
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(conn):
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS slots;
    DROP TABLE IF EXISTS items;

    CREATE TABLE slots (
      slot_id TEXT PRIMARY KEY,
      Lm REAL NOT NULL,
      Wm REAL NOT NULL,
      Hm REAL NOT NULL,
      max_weight REAL NOT NULL,
      distance REAL NOT NULL
    );

    CREATE TABLE items (
      item_id TEXT PRIMARY KEY,
      Lm REAL NOT NULL,
      Wm REAL NOT NULL,
      Hm REAL NOT NULL,
      weight REAL NOT NULL,
      monthly_demand REAL NOT NULL,
      order_pressure REAL NOT NULL
    );
    """)
    conn.commit()

def seed_slots(conn, n_slots, seed):
    random.seed(seed)
    cur = conn.cursor()

    # Zones implicites via distance (petit=proche)
    # Dimensions typiques palette/étagère en mètres (ex: 1.2m x 0.8m etc.)
    L_choices = [1.2, 1.4, 1.6, 2.0]
    W_choices = [0.8, 1.0, 1.2]
    H_choices = [1.5, 1.8, 2.0, 2.4]
    maxw_choices = [300, 500, 800, 1200]
    dist_bases = [5, 20, 50]  # proche / moyen / loin

    for i in range(1, n_slots + 1):
        slot_id = f"S{i:02d}"
        Lm = random.choice(L_choices)
        Wm = random.choice(W_choices)
        Hm = random.choice(H_choices)
        maxw = random.choice(maxw_choices)
        base = random.choice(dist_bases)
        distance = base + random.choice([0, 2, 5, 8, 10])

        cur.execute(
            "INSERT INTO slots(slot_id, Lm, Wm, Hm, max_weight, distance) VALUES (?,?,?,?,?,?)",
            (slot_id, round_m(Lm), round_m(Wm), round_m(Hm), float(maxw), float(distance))
        )

    conn.commit()

def seed_items(conn, n_items, seed):
    random.seed(seed + 1)
    cur = conn.cursor()

    # Items: dimensions plus petites ou égales à des slots la plupart du temps
    for i in range(1, n_items + 1):
        item_id = f"MAT-{i:05d}"

        # Génération dims en mètres
        # (plage réaliste pour colis/palette)
        Lm = random.uniform(0.4, 1.6)
        Wm = random.uniform(0.3, 1.2)
        Hm = random.uniform(0.3, 2.0)

        # poids total (kg)
        weight = random.uniform(20, 900)

        # fréquence consommation (unités/mois) + pression commandes (0..1)
        monthly_demand = random.choice([5, 10, 20, 30, 50, 80, 120])
        order_pressure = random.random()

        cur.execute(
            "INSERT INTO items(item_id, Lm, Wm, Hm, weight, monthly_demand, order_pressure) VALUES (?,?,?,?,?,?,?)",
            (item_id, round_m(Lm), round_m(Wm), round_m(Hm), float(weight), float(monthly_demand), float(order_pressure))
        )

    conn.commit()

def build_synth_db():
    conn = connect()
    create_tables(conn)
    seed_slots(conn, cfg.n_slots, cfg.seed)
    seed_items(conn, cfg.n_items, cfg.seed)
    conn.close()
    print(f"✅ Synth DB created: {DB_PATH.resolve()}")

# ----------------------------
# Oracle cost (ground truth)
# ----------------------------
def load_slots_df(conn):
    return pd.read_sql_query("SELECT * FROM slots ORDER BY slot_id", conn)

def load_items_df(conn):
    return pd.read_sql_query("SELECT * FROM items ORDER BY item_id", conn)

def compute_priority(item_row):
    """
    Priorité P dans [0,1] basée sur :
      - monthly_demand (rotation)
      - order_pressure (pression des commandes)
    """
    # normalisation simple (monthly_demand max=120 min=5)
    md = (item_row["monthly_demand"] - 5) / (120 - 5)
    md = clamp(md, 0.0, 1.0)
    op = clamp(item_row["order_pressure"], 0.0, 1.0)
    return 0.6 * md + 0.4 * op

def oracle_best_slot(item_row, slots_df):
    """
    Renvoie slot_id optimal (oracle) selon contraintes dures + coût.
    Coût = alpha(P)*distance + (1-alpha)*waste_volume
    """
    Lm, Wm, Hm = item_row["Lm"], item_row["Wm"], item_row["Hm"]
    weight = item_row["weight"]
    v_item = m3(Lm, Wm, Hm)

    P = compute_priority(item_row)
    alpha = 0.2 + 0.6 * P  # 0.2..0.8

    feasible = slots_df[
        (slots_df["Lm"] >= Lm) &
        (slots_df["Wm"] >= Wm) &
        (slots_df["Hm"] >= Hm) &
        (slots_df["max_weight"] >= weight)
    ].copy()

    if feasible.empty:
        return None, None  # pas de slot

    # normalisation distance et waste
    dmin, dmax = slots_df["distance"].min(), slots_df["distance"].max()
    def nd(d):
        return 0.0 if dmax == dmin else (d - dmin) / (dmax - dmin)

    v_slot = feasible["Lm"] * feasible["Wm"] * feasible["Hm"]
    waste = (v_slot - v_item).clip(lower=0)

    # normalisation waste vs tous les slots (stable)
    # (évite écrasement si waste proche)
    waste_all = (slots_df["Lm"] * slots_df["Wm"] * slots_df["Hm"]) - v_item
    waste_all = waste_all.clip(lower=0)
    wmin, wmax = waste_all.min(), waste_all.max()
    def nw(w):
        return 0.0 if wmax == wmin else (w - wmin) / (wmax - wmin)

    feasible["distance_cost"] = feasible["distance"].apply(nd)
    feasible["waste_cost"] = waste.apply(nw)

    feasible["true_cost"] = alpha * feasible["distance_cost"] + (1 - alpha) * feasible["waste_cost"]

    best = feasible.sort_values("true_cost").iloc[0]
    return best["slot_id"], float(best["true_cost"])

# ----------------------------
# Training data: (item, slot) -> cost
# ----------------------------
def make_training_pairs(items_df, slots_df, max_slots_per_item=30):
    """
    Crée un dataset de paires (item, slot) avec:
      - features(item + slot)
      - target = true_cost (oracle)
    On garde aussi un flag feasible.
    """
    rows = []
    for _, item in items_df.iterrows():
        P = compute_priority(item)
        alpha = 0.2 + 0.6 * P
        v_item = m3(item["Lm"], item["Wm"], item["Hm"])

        # Pour MVP: on évalue tous les slots
        sample_slots = slots_df if max_slots_per_item >= len(slots_df) else slots_df.sample(max_slots_per_item, random_state=1)

        for _, slot in sample_slots.iterrows():
            feasible = int(
                slot["Lm"] >= item["Lm"] and
                slot["Wm"] >= item["Wm"] and
                slot["Hm"] >= item["Hm"] and
                slot["max_weight"] >= item["weight"]
            )

            # coût oracle si faisable, sinon coût élevé
            if feasible:
                # normaliser par stats globales
                dmin, dmax = slots_df["distance"].min(), slots_df["distance"].max()
                distance_cost = 0.0 if dmax == dmin else (slot["distance"] - dmin) / (dmax - dmin)

                v_slot = m3(slot["Lm"], slot["Wm"], slot["Hm"])
                waste = max(v_slot - v_item, 0.0)
                waste_all = (slots_df["Lm"] * slots_df["Wm"] * slots_df["Hm"]) - v_item
                waste_all = waste_all.clip(lower=0)
                wmin, wmax = float(waste_all.min()), float(waste_all.max())
                waste_cost = 0.0 if wmax == wmin else (waste - wmin) / (wmax - wmin)

                true_cost = alpha * distance_cost + (1 - alpha) * waste_cost
            else:
                true_cost = 5.0  # pénalité (plus grand que tout coût normalisé)

            rows.append({
                # item features
                "item_Lm": item["Lm"],
                "item_Wm": item["Wm"],
                "item_Hm": item["Hm"],
                "item_weight": item["weight"],
                "monthly_demand": item["monthly_demand"],
                "order_pressure": item["order_pressure"],
                "priority_P": P,
                # slot features
                "slot_Lm": slot["Lm"],
                "slot_Wm": slot["Wm"],
                "slot_Hm": slot["Hm"],
                "slot_max_weight": slot["max_weight"],
                "slot_distance": slot["distance"],
                # meta
                "slot_id": slot["slot_id"],
                "item_id": item["item_id"],
                "feasible": feasible,
                # target
                "true_cost": float(true_cost),
            })

    return pd.DataFrame(rows)

def train_model(df_pairs):
    feature_cols = [
        "item_Lm","item_Wm","item_Hm","item_weight",
        "monthly_demand","order_pressure","priority_P",
        "slot_Lm","slot_Wm","slot_Hm","slot_max_weight","slot_distance",
        "feasible"
    ]
    X = df_pairs[feature_cols].values
    y = df_pairs["true_cost"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=123
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"✅ Model trained. MAE(cost) = {mae:.4f}")

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH.resolve()}")

def predict_best_slot_for_item(item_row, slots_df, bundle):
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    P = compute_priority(item_row)

    rows = []
    for _, slot in slots_df.iterrows():
        feasible = int(
            slot["Lm"] >= item_row["Lm"] and
            slot["Wm"] >= item_row["Wm"] and
            slot["Hm"] >= item_row["Hm"] and
            slot["max_weight"] >= item_row["weight"]
        )
        feat = {
            "item_Lm": item_row["Lm"],
            "item_Wm": item_row["Wm"],
            "item_Hm": item_row["Hm"],
            "item_weight": item_row["weight"],
            "monthly_demand": item_row["monthly_demand"],
            "order_pressure": item_row["order_pressure"],
            "priority_P": P,
            "slot_Lm": slot["Lm"],
            "slot_Wm": slot["Wm"],
            "slot_Hm": slot["Hm"],
            "slot_max_weight": slot["max_weight"],
            "slot_distance": slot["distance"],
            "feasible": feasible,
            "slot_id": slot["slot_id"],
        }
        rows.append(feat)

    df = pd.DataFrame(rows)
    X = df[feature_cols].values
    df["pred_cost"] = model.predict(X)

    # Choisir le slot faisable avec coût prédit minimal
    feasible_df = df[df["feasible"] == 1]
    if feasible_df.empty:
        return None, None
    best = feasible_df.sort_values("pred_cost").iloc[0]
    return best["slot_id"], float(best["pred_cost"])

def main():
    build_synth_db()

    conn = connect()
    slots_df = load_slots_df(conn)
    items_df = load_items_df(conn)

    # Générer dataset paires et labels
    pairs = make_training_pairs(items_df, slots_df, max_slots_per_item=cfg.n_slots)
    print("Pairs dataset:", pairs.shape)

    # Entraîner
    train_model(pairs)

    # Démo prédiction vs oracle sur 5 items
    bundle = joblib.load(MODEL_PATH)

    print("\n=== DEMO PREDICTION ===")
    for i in range(5):
        item = items_df.sample(1, random_state=100+i).iloc[0]
        oracle_slot, oracle_cost = oracle_best_slot(item, slots_df)
        pred_slot, pred_cost = predict_best_slot_for_item(item, slots_df, bundle)

        print(f"Item {item['item_id']}: dims={item['Lm']:.3f}x{item['Wm']:.3f}x{item['Hm']:.3f} m, "
              f"w={item['weight']:.1f}kg, md={item['monthly_demand']}, op={item['order_pressure']:.2f}")
        print(f"  Oracle -> {oracle_slot} (cost={oracle_cost})")
        print(f"  ML     -> {pred_slot} (pred_cost={pred_cost})\n")

    conn.close()

if __name__ == "__main__":
    main()
