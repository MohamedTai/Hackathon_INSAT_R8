# WarehouseMVP - Warehouse Management System

## Project Overview

A comprehensive warehouse management system combining:
- **Backend API** - FastAPI-based REST API for warehouse operations
- **AI/ML Services** - Machine learning models for slot optimization and warehouse analytics
- **Streamlit Interfaces** - Interactive dashboards for visualization and management

## Architecture

### Backend (`backend/`)
- **app.py** - Main FastAPI/Streamlit application
- **db.py** - Database layer and ORM
- **logic.py** - Core business logic
- **service.py** - Service layer
- **optimizer_ilp.py** - Integer Linear Programming optimization engine
- **schema.py** - Data models and schemas
- **warehouse.db** - SQLite database

### AI/ML Services (`ai/`)
- **main.py** - Keyboard simulator for IN/OUT event testing
- **streamlit_app.py** - ML dashboard and monitoring
- **ml_slotting_train.py** - Model training pipeline
- **backend_client.py** - Client for backend API
- **config.py** - Configuration management
- **handlers.py** - Event handlers
- **utils.py** - Utility functions
- **slotting_model.joblib** - Trained ML model (serialized)

## Setup

### Requirements
- Python 3.9+
- pandas, numpy, scikit-learn
- FastAPI, uvicorn
- Streamlit
- sqlalchemy

### Installation

```bash
# Using a single virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Running the Application

**Backend API:**
```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

**AI Streamlit Dashboard:**
```bash
cd ai
streamlit run streamlit_app.py
```

**Keyboard Simulator:**
```bash
cd ai
python main.py
```

## Features

- **Slot Management** - Manage warehouse slots with dimensions, weight limits, and zones
- **Product Management** - Track products with dimensions and demand forecasts
- **ML-based Slotting** - Optimize product placement using machine learning
- **Inventory Tracking** - Real-time inventory management
- **Performance Metrics** - Analytics and KPI dashboards

## File Structure

```
WarehouseMVP/
├── backend/
│   ├── app.py
│   ├── db.py
│   ├── logic.py
│   ├── service.py
│   ├── optimizer_ilp.py
│   ├── schema.py
│   ├── seed_real_db.py
│   └── warehouse.db
├── ai/
│   ├── main.py
│   ├── streamlit_app.py
│   ├── ml_slotting_train.py
│   ├── backend_client.py
│   ├── config.py
│   ├── handlers.py
│   ├── utils.py
│   ├── test_predict.py
│   └── slotting_model.joblib
├── .gitignore
└── README.md
```

## Development Notes

- Single virtual environment should be used for the entire project
- Database is SQLite-based for simplicity
- ML models are stored as joblib serialized objects
- Streamlit and FastAPI run on separate ports (8501 for Streamlit, 8000 for API)

