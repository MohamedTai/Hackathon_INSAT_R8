# WarehouseMVP - Warehouse Management System

## Project Overview

A lightweight **Streamlit-based warehouse management system** with integrated machine learning for optimal slot allocation.

## Architecture

### Application (`app/`)
- **app.py** - Main Streamlit application with 4 pages:
  - Dashboard: Overview of warehouse inventory
  - Optimization: ML-powered slot recommendations
  - Training: Model training interface
  - Analytics: Performance metrics and model evaluation

### ML Pipeline (`src/ml/`)
- **training/** - Model training with oracle ground truth
  - `train.py` - Core training logic and utilities
- **evaluation/** - Model performance metrics
  - `evaluate.py` - Accuracy and cost analysis
- **testing/** - Unit and integration tests
  - `test_ml.py` - Pytest test suite

### Data & Models (`src/`)
- **data/** - Synthetic warehouse data (SQLite database)
- **models/** - Trained model artifacts (joblib)

## Features

### Dashboard
- Real-time warehouse statistics
- Slot and item inventory overview
- Distribution analysis

### Optimization
- ML-powered slot recommendations
- Oracle (ground truth) comparison
- Item-slot feasibility validation

### Model Training
- Synthetic data generation
- HistGradientBoosting model training
- Configurable parameters

### Analytics
- Model accuracy metrics
- Cost prediction analysis
- Feature importance visualization

## Setup

### Requirements
- Python 3.9+
- Dependencies: streamlit, pandas, numpy, scikit-learn, joblib, pytest

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

### Training the Model

1. Go to **Training** page
2. Adjust parameters (optional)
3. Click "Start Training"
4. Model will be saved to `src/models/slotting_model.joblib`

### Running Tests

```bash
pytest src/ml/testing/test_ml.py -v
```

## File Structure

```
WarehouseMVP/
├── app/
│   └── app.py                           # Main Streamlit app
├── src/
│   ├── ml/
│   │   ├── training/
│   │   │   ├── train.py                 # Training pipeline
│   │   │   └── __init__.py
│   │   ├── evaluation/
│   │   │   ├── evaluate.py              # Performance evaluation
│   │   │   └── __init__.py
│   │   ├── testing/
│   │   │   ├── test_ml.py               # Test suite
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── data/                            # Data directory (auto-created)
│   ├── models/                          # Models directory (auto-created)
│   └── __init__.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ML Model Details

### Approach
- **Algorithm**: HistGradientBoosting Regressor
- **Task**: Predict cost of item-slot allocation
- **Training Data**: Oracle-based cost function
- **Features**: Item dimensions, weight, demand, priority + Slot dimensions, capacity, distance

### Oracle Function
Ground truth cost is computed as:
```
cost = α * distance_norm + (1 - α) * waste_norm
α = 0.2 + 0.6 * priority
```

Where:
- `priority` = 0.6 × demand_norm + 0.4 × order_pressure
- `distance_norm` = normalized slot distance (0 = close, 1 = far)
- `waste_norm` = normalized unused slot volume

### Performance
- Typical MAE: ~0.04 on test set
- Slot match accuracy: ~80-85%

## Development Notes

- All code uses Streamlit for UI (no backend API)
- Database is SQLite for simplicity
- Models stored as joblib binary files
- Synthetic data generation for demo (can be replaced with real data)
- Single virtual environment for entire project

