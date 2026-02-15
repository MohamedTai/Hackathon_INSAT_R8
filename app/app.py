"""
Warehouse Management Streamlit Dashboard
Main application for warehouse optimization and monitoring
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "training"))

from train import (
    connect_db, load_slots_df, load_items_df,
    oracle_best_slot, predict_best_slot, compute_priority,
    build_database, make_training_pairs, train_model,
    MODELS_DIR, DB_PATH
)

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Warehouse MVP",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📦 Warehouse Management System")
st.markdown("ML-powered warehouse optimization")

# ===========================
# Sidebar Navigation
# ===========================
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Optimization", "Training", "Analytics"]
)

# ===========================
# Initialize Session State
# ===========================
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if MODELS_DIR.joinpath("slotting_model.joblib").exists():
    st.session_state.model_loaded = True


# ===========================
# Dashboard Page
# ===========================
def page_dashboard():
    st.header("📊 Dashboard")
    
    # Check if database exists
    if not DB_PATH.exists():
        st.warning("⚠️ Database not initialized. Please go to the **Training** page and click 'Start Training' first.")
        st.info("This will generate synthetic warehouse data and train the ML model.")
        return
    
    try:
        conn = connect_db()
        slots_df = load_slots_df(conn)
        items_df = load_items_df(conn)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Go to **Training** page to initialize the database.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Slots", len(slots_df))
    with col2:
        st.metric("Total Items", len(items_df))
    with col3:
        avg_slot_volume = (slots_df["Lm"] * slots_df["Wm"] * slots_df["Hm"]).mean()
        st.metric("Avg Slot Volume (m³)", f"{avg_slot_volume:.2f}")
    with col4:
        avg_item_weight = items_df["weight"].mean()
        st.metric("Avg Item Weight (kg)", f"{avg_item_weight:.1f}")
    
    st.divider()
    
    # Slots info
    st.subheader("📍 Slot Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Slot Dimensions (m)**")
        slot_stats = pd.DataFrame({
            "Length": [slots_df["Lm"].min(), slots_df["Lm"].max(), slots_df["Lm"].mean()],
            "Width": [slots_df["Wm"].min(), slots_df["Wm"].max(), slots_df["Wm"].mean()],
            "Height": [slots_df["Hm"].min(), slots_df["Hm"].max(), slots_df["Hm"].mean()],
        }, index=["Min", "Max", "Avg"])
        st.dataframe(slot_stats, use_container_width=True)
    
    with col2:
        st.write("**Distance Distribution**")
        fig = st.bar_chart(slots_df["distance"].value_counts().sort_index())
    
    # Items info
    st.subheader("📦 Item Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Item Demand Distribution**")
        fig = st.bar_chart(items_df["monthly_demand"].value_counts().sort_index())
    
    with col2:
        st.write("**Item Weight Distribution**")
        st.histogram(items_df["weight"], bins=20, use_container_width=True)
    
    conn.close()


# ===========================
# Optimization Page
# ===========================
def page_optimization():
    st.header("🎯 Slot Optimization")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model not trained yet. Go to Training page.")
        return
    
    conn = connect_db()
    slots_df = load_slots_df(conn)
    items_df = load_items_df(conn)
    
    st.subheader("Select Item to Optimize")
    
    item_id = st.selectbox(
        "Choose item",
        items_df["item_id"].values
    )
    
    if item_id:
        item = items_df[items_df["item_id"] == item_id].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Item Details**")
            st.json({
                "ID": item["item_id"],
                "Dimensions (m)": f"{item['Lm']:.3f} × {item['Wm']:.3f} × {item['Hm']:.3f}",
                "Weight (kg)": f"{item['weight']:.1f}",
                "Monthly Demand": int(item["monthly_demand"]),
                "Order Pressure": f"{item['order_pressure']:.2f}",
            })
        
        with col2:
            st.write("**Computed Priority**")
            priority = compute_priority(item)
            st.metric("Priority Score", f"{priority:.3f}")
        
        st.divider()
        
        # Load model
        model_bundle = joblib.load(MODELS_DIR / "slotting_model.joblib")
        
        # Get recommendations
        oracle_slot, oracle_cost = oracle_best_slot(item, slots_df)
        pred_slot, pred_cost = predict_best_slot(item, slots_df, model_bundle)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Oracle (Ground Truth)**")
            if oracle_slot:
                oracle_slot_info = slots_df[slots_df["slot_id"] == oracle_slot].iloc[0]
                st.success(f"✓ Slot: {oracle_slot}")
                st.json({
                    "Slot Dimensions (m)": f"{oracle_slot_info['Lm']:.3f} × {oracle_slot_info['Wm']:.3f} × {oracle_slot_info['Hm']:.3f}",
                    "Max Weight (kg)": f"{oracle_slot_info['max_weight']:.1f}",
                    "Distance (m)": f"{oracle_slot_info['distance']:.1f}",
                    "Cost": f"{oracle_cost:.4f}",
                })
            else:
                st.error("No feasible slot found")
        
        with col2:
            st.write("**ML Prediction**")
            if pred_slot:
                pred_slot_info = slots_df[slots_df["slot_id"] == pred_slot].iloc[0]
                st.info(f"📍 Slot: {pred_slot}")
                st.json({
                    "Slot Dimensions (m)": f"{pred_slot_info['Lm']:.3f} × {pred_slot_info['Wm']:.3f} × {pred_slot_info['Hm']:.3f}",
                    "Max Weight (kg)": f"{pred_slot_info['max_weight']:.1f}",
                    "Distance (m)": f"{pred_slot_info['distance']:.1f}",
                    "Cost": f"{pred_cost:.4f}",
                })
            else:
                st.error("No feasible slot found")
    
    conn.close()


# ===========================
# Training Page
# ===========================
def page_training():
    st.header("🚀 Model Training")
    
    st.write("Train the warehouse slotting optimization model.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        n_slots = st.slider("Number of Slots", 10, 100, 30)
        n_items = st.slider("Number of Items", 100, 5000, 2000)
        learning_rate = st.select_slider("Learning Rate", [0.01, 0.04, 0.08, 0.15], value=0.08)
    
    with col2:
        st.subheader("Status")
        if DB_PATH.exists():
            st.success("✓ Database exists")
        else:
            st.info("Database will be created during training")
        
        if st.session_state.model_loaded:
            st.success("✓ Model trained")
        else:
            st.warning("Model not trained")
    
    st.divider()
    
    if st.button("🔧 Start Training", key="train_btn", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.write("📦 Building database...")
            build_database()
            progress_bar.progress(25)
            
            status.write("📊 Loading data...")
            conn = connect_db()
            slots_df = load_slots_df(conn)
            items_df = load_items_df(conn)
            progress_bar.progress(50)
            
            status.write("🧮 Generating training pairs...")
            pairs = make_training_pairs(items_df, slots_df, max_slots_per_item=30)
            progress_bar.progress(75)
            
            status.write("🤖 Training model...")
            mae = train_model(pairs)
            progress_bar.progress(100)
            
            conn.close()
            st.session_state.model_loaded = True
            
            st.success("✅ Training complete!")
            st.metric("Mean Absolute Error", f"{mae:.4f}")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ===========================
# Analytics Page
# ===========================
def page_analytics():
    st.header("📈 Analytics & Metrics")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model not trained. Train model first.")
        return
    
    conn = connect_db()
    slots_df = load_slots_df(conn)
    items_df = load_items_df(conn)
    
    st.subheader("Model Performance")
    
    model_bundle = joblib.load(MODELS_DIR / "slotting_model.joblib")
    
    # Evaluate on sample
    sample_items = items_df.sample(n=min(50, len(items_df)), random_state=42)
    
    matches = 0
    cost_diffs = []
    
    for _, item in sample_items.iterrows():
        oracle_slot, oracle_cost = oracle_best_slot(item, slots_df)
        pred_slot, pred_cost = predict_best_slot(item, slots_df, model_bundle)
        
        if pred_slot and oracle_slot:
            if pred_slot == oracle_slot:
                matches += 1
            cost_diffs.append(abs(pred_cost - oracle_cost))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        accuracy = (matches / len(sample_items)) * 100
        st.metric("Slot Match Accuracy", f"{accuracy:.1f}%")
    
    with col2:
        avg_diff = sum(cost_diffs) / len(cost_diffs) if cost_diffs else 0
        st.metric("Avg Cost Difference", f"{avg_diff:.4f}")
    
    with col3:
        st.metric("Sample Size", len(sample_items))
    
    st.divider()
    
    st.subheader("Feature Importance")
    model = model_bundle["model"]
    
    feature_cols = model_bundle["feature_cols"]
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    st.bar_chart(importance_df.set_index("Feature"))
    
    conn.close()


# ===========================
# Router
# ===========================
if page == "Dashboard":
    page_dashboard()
elif page == "Optimization":
    page_optimization()
elif page == "Training":
    page_training()
elif page == "Analytics":
    page_analytics()
