"""
ML Training Pipeline for Warehouse Slotting Optimization
Trains a HistGradientBoosting model to predict optimal slot allocation for items
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ===========================
# Config
# ===========================
@dataclass
class Config:
    n_slots: int = 30
    n_items: int = 2000
    seed: int = 42
    test_size: float = 0.2
    max_depth: int = 6
    learning_rate: float = 0.08
    max_iter: int = 250


# ===========================
# Paths
# ===========================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
MODELS_DIR = BASE_DIR / "src" / "models"

DB_PATH = DATA_DIR / "warehouse.db"
MODEL_PATH = MODELS_DIR / "slotting_model.joblib"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# Helper Functions
# ===========================
def m3(L, W, H):
    """Calculate volume in cubic meters"""
    return float(L) * float(W) * float(H)


def clamp(x, lo, hi):
    """Clamp value between bounds"""
    return max(lo, min(hi, x))


def round_m(x):
    """Round to 3 decimal places (meters)"""
    return round(float(x), 3)


# ===========================
# Database Functions
# ===========================
def connect_db():
    """Connect to SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    """Create database tables for slots and items"""
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
    """Generate synthetic slot data"""
    random.seed(seed)
    cur = conn.cursor()

    L_choices = [1.2, 1.4, 1.6, 2.0]
    W_choices = [0.8, 1.0, 1.2]
    H_choices = [1.5, 1.8, 2.0, 2.4]
    maxw_choices = [300, 500, 800, 1200]
    dist_bases = [5, 20, 50]

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
    """Generate synthetic item data"""
    random.seed(seed + 1)
    cur = conn.cursor()

    for i in range(1, n_items + 1):
        item_id = f"MAT-{i:05d}"

        Lm = random.uniform(0.4, 1.6)
        Wm = random.uniform(0.3, 1.2)
        Hm = random.uniform(0.3, 2.0)

        weight = random.uniform(20, 900)

        monthly_demand = random.choice([5, 10, 20, 30, 50, 80, 120])
        order_pressure = random.random()

        cur.execute(
            "INSERT INTO items(item_id, Lm, Wm, Hm, weight, monthly_demand, order_pressure) VALUES (?,?,?,?,?,?,?)",
            (item_id, round_m(Lm), round_m(Wm), round_m(Hm), float(weight), float(monthly_demand), float(order_pressure))
        )
    conn.commit()


def build_database():
    """Create and seed synthetic database"""
    cfg = Config()
    conn = connect_db()
    create_tables(conn)
    seed_slots(conn, cfg.n_slots, cfg.seed)
    seed_items(conn, cfg.n_items, cfg.seed)
    conn.close()
    print(f"✅ Database created: {DB_PATH.resolve()}")


def load_slots_df(conn):
    """Load slots from database"""
    return pd.read_sql_query("SELECT * FROM slots ORDER BY slot_id", conn)


def load_items_df(conn):
    """Load items from database"""
    return pd.read_sql_query("SELECT * FROM items ORDER BY item_id", conn)


# ===========================
# Oracle & Priority
# ===========================
def compute_priority(item_row):
    """Compute priority P in [0,1] based on demand and order pressure"""
    md = (item_row["monthly_demand"] - 5) / (120 - 5)
    md = clamp(md, 0.0, 1.0)
    op = clamp(item_row["order_pressure"], 0.0, 1.0)
    return 0.6 * md + 0.4 * op


def oracle_best_slot(item_row, slots_df):
    """Find optimal slot using ground truth oracle function"""
    Lm, Wm, Hm = item_row["Lm"], item_row["Wm"], item_row["Hm"]
    weight = item_row["weight"]
    v_item = m3(Lm, Wm, Hm)

    P = compute_priority(item_row)
    alpha = 0.2 + 0.6 * P

    feasible = slots_df[
        (slots_df["Lm"] >= Lm) &
        (slots_df["Wm"] >= Wm) &
        (slots_df["Hm"] >= Hm) &
        (slots_df["max_weight"] >= weight)
    ].copy()

    if feasible.empty:
        return None, None

    dmin, dmax = slots_df["distance"].min(), slots_df["distance"].max()

    def nd(d):
        return 0.0 if dmax == dmin else (d - dmin) / (dmax - dmin)

    v_slot = feasible["Lm"] * feasible["Wm"] * feasible["Hm"]
    waste = (v_slot - v_item).clip(lower=0)

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


# ===========================
# Training Data
# ===========================
def make_training_pairs(items_df, slots_df, max_slots_per_item=30):
    """Create training dataset of (item, slot) pairs with oracle costs"""
    rows = []
    for _, item in items_df.iterrows():
        P = compute_priority(item)
        alpha = 0.2 + 0.6 * P
        v_item = m3(item["Lm"], item["Wm"], item["Hm"])

        sample_slots = slots_df if max_slots_per_item >= len(slots_df) else slots_df.sample(max_slots_per_item, random_state=1)

        for _, slot in sample_slots.iterrows():
            feasible = int(
                slot["Lm"] >= item["Lm"] and
                slot["Wm"] >= item["Wm"] and
                slot["Hm"] >= item["Hm"] and
                slot["max_weight"] >= item["weight"]
            )

            if feasible:
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
                true_cost = 5.0

            rows.append({
                "item_Lm": item["Lm"],
                "item_Wm": item["Wm"],
                "item_Hm": item["Hm"],
                "item_weight": item["weight"],
                "monthly_demand": item["monthly_demand"],
                "order_pressure": item["order_pressure"],
                "priority_P": P,
                "slot_Lm": slot["Lm"],
                "slot_Wm": slot["Wm"],
                "slot_Hm": slot["Hm"],
                "slot_max_weight": slot["max_weight"],
                "slot_distance": slot["distance"],
                "slot_id": slot["slot_id"],
                "item_id": item["item_id"],
                "feasible": feasible,
                "true_cost": float(true_cost),
            })

    return pd.DataFrame(rows)


# ===========================
# Model Training
# ===========================
def train_model(df_pairs):
    """Train HistGradientBoosting model"""
    cfg = Config()
    
    feature_cols = [
        "item_Lm", "item_Wm", "item_Hm", "item_weight",
        "monthly_demand", "order_pressure", "priority_P",
        "slot_Lm", "slot_Wm", "slot_Hm", "slot_max_weight", "slot_distance",
        "feasible"
    ]
    
    X = df_pairs[feature_cols].values
    y = df_pairs["true_cost"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=123
    )

    model = HistGradientBoostingRegressor(
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        random_state=123
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"✅ Model trained. MAE = {mae:.4f}")

    # Save model
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH.resolve()}")
    
    return mae


def predict_best_slot(item_row, slots_df, model_bundle):
    """Predict best slot for item using trained model"""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

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

    feasible_df = df[df["feasible"] == 1]
    if feasible_df.empty:
        return None, None
    
    best = feasible_df.sort_values("pred_cost").iloc[0]
    return best["slot_id"], float(best["pred_cost"])


# ===========================
# Main
# ===========================
def main():
    print("🚀 Starting ML Training Pipeline...\n")
    
    cfg = Config()
    
    # Build database
    build_database()

    # Load data
    conn = connect_db()
    slots_df = load_slots_df(conn)
    items_df = load_items_df(conn)
    print(f"Loaded {len(slots_df)} slots and {len(items_df)} items\n")

    # Generate training pairs
    print("Generating training pairs...")
    pairs = make_training_pairs(items_df, slots_df, max_slots_per_item=cfg.n_slots)
    print(f"Dataset shape: {pairs.shape}\n")

    # Train model
    print("Training model...")
    mae = train_model(pairs)

    # Demo predictions
    print("\n=== DEMO: Predictions vs Oracle ===\n")
    model_bundle = joblib.load(MODEL_PATH)
    
    for i in range(5):
        item = items_df.sample(1, random_state=100 + i).iloc[0]
        oracle_slot, oracle_cost = oracle_best_slot(item, slots_df)
        pred_slot, pred_cost = predict_best_slot(item, slots_df, model_bundle)

        print(f"Item {item['item_id']}: {item['Lm']:.3f}x{item['Wm']:.3f}x{item['Hm']:.3f}m, {item['weight']:.1f}kg")
        print(f"  Oracle: {oracle_slot} (cost={oracle_cost:.4f})")
        print(f"  ML:     {pred_slot} (cost={pred_cost:.4f})\n")

    conn.close()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
