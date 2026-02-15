# streamlit_app.py
# Streamlit demo: ML slot recommendation using slotting_synth.db + slotting_model.joblib
# Run:
#   pip install streamlit pandas numpy scikit-learn joblib
#   streamlit run streamlit_app.py

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Paths (expected in same folder as this app) ----------
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "slotting_synth.db"
MODEL_PATH = APP_DIR / "slotting_model.joblib"


# ---------- Helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def compute_priority(monthly_demand: float, order_pressure: float) -> float:
    """
    Must match training logic:
      md normalized from [5..120], mixed with order_pressure.
    """
    md = (monthly_demand - 5.0) / (120.0 - 5.0)
    md = clamp(md, 0.0, 1.0)
    op = clamp(order_pressure, 0.0, 1.0)
    return 0.6 * md + 0.4 * op

@st.cache_data(show_spinner=False)
def load_slots(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM slots ORDER BY slot_id", conn)
    conn.close()
    return df

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Returns dict with:
      - model
      - feature_cols
    """
    return joblib.load(model_path)

def predict_best_slot(
    Lm: float, Wm: float, Hm: float,
    weight: float,
    monthly_demand: float,
    order_pressure: float,
    slots_df: pd.DataFrame,
    bundle: dict
):
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    P = compute_priority(monthly_demand, order_pressure)

    rows = []
    for _, s in slots_df.iterrows():
        feasible = int(
            float(s["Lm"]) >= Lm and
            float(s["Wm"]) >= Wm and
            float(s["Hm"]) >= Hm and
            float(s["max_weight"]) >= weight
        )

        rows.append({
            # item features
            "item_Lm": Lm,
            "item_Wm": Wm,
            "item_Hm": Hm,
            "item_weight": weight,
            "monthly_demand": monthly_demand,
            "order_pressure": order_pressure,
            "priority_P": P,
            # slot features
            "slot_Lm": float(s["Lm"]),
            "slot_Wm": float(s["Wm"]),
            "slot_Hm": float(s["Hm"]),
            "slot_max_weight": float(s["max_weight"]),
            "slot_distance": float(s["distance"]),
            # meta
            "feasible": feasible,
            "slot_id": s["slot_id"],
        })

    df = pd.DataFrame(rows)
    X = df[feature_cols].values
    df["pred_cost"] = model.predict(X)

    feasible_df = df[df["feasible"] == 1].sort_values("pred_cost")
    if feasible_df.empty:
        return None, None, df.sort_values("pred_cost")

    best = feasible_df.iloc[0]
    return str(best["slot_id"]), float(best["pred_cost"]), df.sort_values("pred_cost")


# ---------- UI ----------
st.set_page_config(page_title="Slotting ML Demo", layout="wide")
st.title("📦 Slotting ML Demo (Streamlit)")
st.write(
    "Cette app démontre comment le modèle ML choisit un **slot idéal** pour une marchandise "
    "en fonction des **dimensions (m)**, **poids (kg)**, **consommation mensuelle**, et **pression commandes**."
)

# Status checks
colA, colB = st.columns(2)
with colA:
    st.subheader("Fichiers requis")
    st.write(f"DB: `{DB_PATH.name}` -> {'✅ trouvée' if DB_PATH.exists() else '❌ introuvable'}")
    st.write(f"Model: `{MODEL_PATH.name}` -> {'✅ trouvé' if MODEL_PATH.exists() else '❌ introuvable'}")
with colB:
    st.subheader("Rappel")
    st.write("- Les dimensions sont en **mètres** et affichées avec **3 décimales**.")
    st.write("- Le modèle renvoie un **coût prédit** (plus petit = meilleur).")

if not DB_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Il manque `slotting_synth.db` ou `slotting_model.joblib` dans le même dossier que `streamlit_app.py`.\n\n"
        "➡️ Mets ces fichiers dans `ai/` puis relance l’app."
    )
    st.stop()

slots_df = load_slots(str(DB_PATH))
bundle = load_model(str(MODEL_PATH))

st.divider()
st.subheader("1) Entrer les caractéristiques de la marchandise")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    Lm = st.number_input("Longueur L (m)", min_value=0.01, value=1.200, step=0.001, format="%.3f")
with c2:
    Wm = st.number_input("Largeur W (m)", min_value=0.01, value=0.800, step=0.001, format="%.3f")
with c3:
    Hm = st.number_input("Hauteur H (m)", min_value=0.01, value=1.500, step=0.001, format="%.3f")
with c4:
    weight = st.number_input("Poids total (kg)", min_value=1.0, value=200.0, step=1.0)
with c5:
    monthly_demand = st.number_input("Consommation mensuelle", min_value=0.0, value=50.0, step=1.0)
with c6:
    order_pressure = st.slider("Pression commandes (0..1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

P = compute_priority(monthly_demand, order_pressure)
st.caption(f"Priorité calculée P = **{P:.3f}** (plus grand = plus prioritaire/proche expédition).")

st.divider()
st.subheader("2) Recommander un slot")

run = st.button("🔎 Recommander", type="primary")

if run:
    # Ensure 3-decimal display; internal values can remain float
    Lm_r, Wm_r, Hm_r = round(float(Lm), 3), round(float(Wm), 3), round(float(Hm), 3)

    best_slot, best_cost, ranked = predict_best_slot(
        Lm=Lm_r, Wm=Wm_r, Hm=Hm_r,
        weight=float(weight),
        monthly_demand=float(monthly_demand),
        order_pressure=float(order_pressure),
        slots_df=slots_df,
        bundle=bundle,
    )

    if best_slot is None:
        st.error("❌ Aucun slot compatible avec ces dimensions/poids.")
    else:
        st.success(f"✅ Slot recommandé: **{best_slot}**  | coût prédit = **{best_cost:.4f}**")

    st.subheader("Top 10 des slots (triés par coût prédit)")
    show = ranked.copy()

    # Add readable feasibility and show helpful columns first
    show["feasible"] = show["feasible"].map({0: "no", 1: "yes"})
    cols = [
        "slot_id", "pred_cost", "feasible",
        "slot_distance", "slot_Lm", "slot_Wm", "slot_Hm", "slot_max_weight",
        "priority_P", "item_Lm", "item_Wm", "item_Hm", "item_weight",
    ]
    st.dataframe(show[cols].head(10), use_container_width=True)

    st.subheader("Pourquoi ce slot ? (explication simple)")
    if best_slot is not None:
        best_row = ranked[(ranked["slot_id"] == best_slot)].iloc[0]
        st.write(
            f"- **Compatibilité**: {('OK' if best_row['feasible'] == 1 else 'NON')} (dimensions/poids)\n"
            f"- **Distance** (plus petit = mieux): {best_row['slot_distance']}\n"
            f"- **Capacité slot**: {best_row['slot_Lm']:.3f}×{best_row['slot_Wm']:.3f}×{best_row['slot_Hm']:.3f} m\n"
            f"- **Poids max slot**: {best_row['slot_max_weight']:.0f} kg\n"
            f"- **Priorité produit P**: {best_row['priority_P']:.3f} (impacte la préférence pour les slots proches)"
        )

st.divider()
with st.expander("Voir les slots disponibles (30 slots)"):
    st.dataframe(slots_df, use_container_width=True)
