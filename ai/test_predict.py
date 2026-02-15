import sqlite3
from pathlib import Path
import pandas as pd
import joblib

DB_PATH = Path("slotting_synth.db")          # DB synthétique créée par ton script
MODEL_PATH = Path("slotting_model.joblib")   # Modèle entraîné

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def compute_priority(monthly_demand: float, order_pressure: float) -> float:
    # Même logique que l'entraînement
    md = (monthly_demand - 5) / (120 - 5)
    md = clamp(md, 0.0, 1.0)
    op = clamp(order_pressure, 0.0, 1.0)
    return 0.6 * md + 0.4 * op

def load_slots():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM slots ORDER BY slot_id", conn)
    conn.close()
    return df

def predict_best_slot(Lm, Wm, Hm, weight, monthly_demand, order_pressure):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    slots_df = load_slots()
    P = compute_priority(monthly_demand, order_pressure)

    rows = []
    for _, s in slots_df.iterrows():
        feasible = int(
            s["Lm"] >= Lm and
            s["Wm"] >= Wm and
            s["Hm"] >= Hm and
            s["max_weight"] >= weight
        )
        rows.append({
            "item_Lm": Lm,
            "item_Wm": Wm,
            "item_Hm": Hm,
            "item_weight": weight,
            "monthly_demand": monthly_demand,
            "order_pressure": order_pressure,
            "priority_P": P,
            "slot_Lm": float(s["Lm"]),
            "slot_Wm": float(s["Wm"]),
            "slot_Hm": float(s["Hm"]),
            "slot_max_weight": float(s["max_weight"]),
            "slot_distance": float(s["distance"]),
            "feasible": feasible,
            "slot_id": s["slot_id"]
        })

    df = pd.DataFrame(rows)
    X = df[feature_cols].values
    df["pred_cost"] = model.predict(X)

    feasible_df = df[df["feasible"] == 1].sort_values("pred_cost")
    if feasible_df.empty:
        return None, None, None

    best = feasible_df.iloc[0]
    top5 = feasible_df.head(5)[["slot_id", "pred_cost", "slot_distance", "slot_Lm", "slot_Wm", "slot_Hm", "slot_max_weight"]]
    return best["slot_id"], float(best["pred_cost"]), top5

def main():
    print("=== TEST PREDICTION SLOT (mètres, 3 décimales) ===")
    print("Exemple: L=1.200 W=0.800 H=1.500 weight=200 monthly_demand=50 order_pressure=0.6\n")

    while True:
        s = input("Entrez: L W H weight monthly_demand order_pressure  (ou Q): ").strip()
        if s.upper() == "Q":
            break

        try:
            Lm, Wm, Hm, weight, md, op = map(float, s.split())
        except Exception:
            print("Format incorrect. Exemple: 1.200 0.800 1.500 200 50 0.6")
            continue

        # arrondi affichage 3 décimales
        Lm = round(Lm, 3); Wm = round(Wm, 3); Hm = round(Hm, 3)

        slot_id, pred_cost, top5 = predict_best_slot(Lm, Wm, Hm, weight, md, op)
        if slot_id is None:
            print("❌ Aucun slot compatible (dimensions/poids).\n")
            continue

        print(f"✅ Slot recommandé: {slot_id} | pred_cost={pred_cost:.4f}")
        print("Top 5 slots (meilleurs scores):")
        print(top5.to_string(index=False))
        print()

if __name__ == "__main__":
    main()
