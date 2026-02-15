"""
Model Evaluation Module
Evaluates ML model performance against oracle
"""

import joblib
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "training"))

from train import (
    connect_db, load_slots_df, load_items_df,
    oracle_best_slot, predict_best_slot,
    MODELS_DIR
)

MODEL_PATH = MODELS_DIR / "slotting_model.joblib"


def evaluate_model():
    """Evaluate model performance"""
    print("📊 Model Evaluation\n")
    
    conn = connect_db()
    slots_df = load_slots_df(conn)
    items_df = load_items_df(conn)
    
    model_bundle = joblib.load(MODEL_PATH)
    
    # Evaluate on sample of items
    sample_items = items_df.sample(n=min(100, len(items_df)), random_state=42)
    
    results = []
    for _, item in sample_items.iterrows():
        oracle_slot, oracle_cost = oracle_best_slot(item, slots_df)
        pred_slot, pred_cost = predict_best_slot(item, slots_df, model_bundle)
        
        if pred_slot and oracle_slot:
            match = 1 if pred_slot == oracle_slot else 0
            cost_diff = abs(pred_cost - oracle_cost)
            results.append({
                "item_id": item["item_id"],
                "oracle_slot": oracle_slot,
                "pred_slot": pred_slot,
                "match": match,
                "oracle_cost": oracle_cost,
                "pred_cost": pred_cost,
                "cost_diff": cost_diff,
            })
    
    df = pd.DataFrame(results)
    
    accuracy = df["match"].mean()
    avg_cost_diff = df["cost_diff"].mean()
    
    print(f"Evaluated on {len(df)} items")
    print(f"Slot Match Accuracy: {accuracy:.2%}")
    print(f"Avg Cost Difference: {avg_cost_diff:.4f}\n")
    
    print("Sample results:")
    print(df[["item_id", "oracle_slot", "pred_slot", "match", "cost_diff"]].head(10).to_string())
    
    conn.close()
    return accuracy, avg_cost_diff


if __name__ == "__main__":
    evaluate_model()
