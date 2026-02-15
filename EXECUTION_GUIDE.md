# WarehouseMVP - Complete Step-by-Step Execution Guide

## 🚀 Project Setup & Execution

### Step 1: Navigate to Project Directory
```bash
cd C:\Users\21698\Desktop\WarehouseMVP
```

### Step 2: Create Virtual Environment (First Time Only)
```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

**Expected Output:**
```
(venv) PS C:\Users\21698\Desktop\WarehouseMVP>
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**This installs:**
- streamlit==1.28.0
- pandas==2.1.1
- numpy==1.24.3
- scikit-learn==1.3.1
- joblib==1.3.2
- pytest==7.4.3
- python-dotenv==1.0.0

---

## 📊 Run the Application

### Step 5: Launch Streamlit Application
```bash
streamlit run app/app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://YOUR_IP:8501
```

**The app opens automatically** in your default browser at `http://localhost:8501`

---

## 🎯 Using the Application

### PAGE 1: Dashboard
1. Open Streamlit at `http://localhost:8501`
2. Default page shows: **Dashboard**
3. View:
   - Total slots (30)
   - Total items (2000)
   - Average slot volume
   - Average item weight
   - Slot dimensions statistics
   - Distance distribution
   - Item demand distribution
   - Item weight distribution

### PAGE 2: Training (Train the Model)
1. Click **"Training"** in left sidebar
2. Configure parameters (optional):
   - Number of Slots: 10-100 (default: 30)
   - Number of Items: 100-5000 (default: 2000)
   - Learning Rate: 0.01, 0.04, 0.08, 0.15 (default: 0.08)
3. Click **"🔧 Start Training"** button
4. Wait for completion (approximately 76 seconds):
   - 📦 Building database (1s)
   - 📊 Loading data (1s)
   - 🧮 Generating training pairs (30s)
   - 🤖 Training model (45s)
5. View results:
   - ✅ Training complete!
   - Mean Absolute Error (MAE) displayed
   - Model saved to `src/models/slotting_model.joblib`

### PAGE 3: Optimization (Get Slot Recommendations)
1. Click **"Optimization"** in left sidebar
2. **Select Item to Optimize**:
   - Choose from dropdown (e.g., MAT-00001)
3. View **Item Details**:
   - ID, Dimensions, Weight, Monthly Demand, Order Pressure
4. View **Computed Priority**:
   - Priority Score (0.0-1.0)
5. View **Recommendations**:
   - **Oracle (Ground Truth)**: Optimal slot from ground truth
   - **ML Prediction**: ML model's recommended slot
   - Compare costs and see if they match

### PAGE 4: Analytics (Model Performance)
1. Click **"Analytics"** in left sidebar
2. View **Model Performance**:
   - Slot Match Accuracy (%)
   - Average Cost Difference
   - Sample Size
3. View **Feature Importance**:
   - Bar chart of which features matter most
   - Features ranked by importance

---

## 🧪 Run Tests

### Step 6: Run Test Suite
```bash
# Run all tests
pytest src/ml/testing/test_ml.py -v

# Run specific test class
pytest src/ml/testing/test_ml.py::TestHelpers -v

# Run with coverage
pytest src/ml/testing/test_ml.py --cov=src
```

**Expected Output:**
```
test_ml.py::TestHelpers::test_m3_volume PASSED
test_ml.py::TestHelpers::test_clamp PASSED
test_ml.py::TestDatabase::test_connect_db PASSED
...
====== X passed in Y.XXs ======
```

---

## 📈 Run Model Evaluation

### Step 7: Evaluate Model Performance
```bash
python src/ml/evaluation/evaluate.py
```

**Expected Output:**
```
📊 Model Evaluation

Evaluated on 100 items
Slot Match Accuracy: 82.00%
Avg Cost Difference: 0.0425

Sample results:
  item_id  oracle_slot  pred_slot  match  cost_diff
MAT-00001      S05         S05        1      0.0012
MAT-00002      S12         S12        1      0.0034
...
```

---

## 🤖 Train Model Directly (Command Line)

### Step 8: Alternative Training (No Streamlit UI)
```bash
python src/ml/training/train.py
```

**Expected Output:**
```
🚀 Starting ML Training Pipeline...

✅ Database created: C:\...\warehouse.db
Loaded 30 slots and 2000 items

Generating training pairs...
Dataset shape: (60000, 17)

Training model...
✅ Model trained. MAE = 0.0412
✅ Model saved: C:\...\slotting_model.joblib

=== DEMO: Predictions vs Oracle ===

Item MAT-00001: 0.523x0.812x1.456m, 234.5kg
  Oracle: S05 (cost=0.2345)
  ML:     S05 (cost=0.2389)

Item MAT-00002: 0.634x0.921x0.987m, 456.2kg
  Oracle: S12 (cost=0.1234)
  ML:     S12 (cost=0.1267)

...

✅ Training complete!
```

---

## 📁 Project Structure Created During Execution

After running the training, the following files are created:

```
WarehouseMVP/
├── src/
│   ├── data/
│   │   └── warehouse.db              # SQLite database (created)
│   │
│   └── models/
│       └── slotting_model.joblib     # Trained model (created)
│
├── app/
│   └── app.py                         # Streamlit app (exists)
│
└── src/ml/
    ├── training/
    │   └── train.py
    ├── evaluation/
    │   └── evaluate.py
    └── testing/
        └── test_ml.py
```

---

## 🔄 Complete Workflow Example

### Full End-to-End Execution (20 minutes)

```bash
# 1. Open PowerShell/Terminal
# 2. Navigate to project
cd C:\Users\21698\Desktop\WarehouseMVP

# 3. Activate virtual environment
venv\Scripts\Activate.ps1

# 4. Install dependencies (first time)
pip install -r requirements.txt

# 5. Launch Streamlit
streamlit run app/app.py
# Browser opens automatically at http://localhost:8501

# 6. In browser:
#    - View Dashboard page
#    - Go to Training page
#    - Click "Start Training" button
#    - Wait ~76 seconds

# 7. After training completes:
#    - Go to Optimization page
#    - Select an item from dropdown
#    - View Oracle vs ML predictions
#    - Go to Analytics page
#    - View model performance metrics

# 8. To run tests (in new terminal window with venv activated)
pytest src/ml/testing/test_ml.py -v

# 9. To evaluate model performance
python src/ml/evaluation/evaluate.py
```

---

## 🛑 Stop the Application

```bash
# Stop Streamlit (in terminal)
Ctrl + C

# Deactivate virtual environment
deactivate
```

---

## 🐛 Troubleshooting

### Issue: "Module not found: streamlit"
**Solution:**
```bash
pip install streamlit
```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
streamlit run app/app.py --server.port 8502
```

### Issue: "No model found" when going to Optimization page
**Solution:**
1. Go to Training page
2. Click "Start Training"
3. Wait for completion
4. Then go to Optimization page

### Issue: Database not found
**Solution:**
```bash
# Delete old database and retrain
rm src/data/warehouse.db
python src/ml/training/train.py
```

---

## 📊 Command Summary Table

| Task | Command | Time | Notes |
|------|---------|------|-------|
| Setup Environment | `python -m venv venv` | 30s | One-time |
| Activate venv | `venv\Scripts\Activate.ps1` | 1s | Each session |
| Install deps | `pip install -r requirements.txt` | 2-3m | First time only |
| Start Streamlit | `streamlit run app/app.py` | - | Keep running |
| Train model | Click in Training page | 76s | Via UI |
| Train via CLI | `python src/ml/training/train.py` | 76s | Direct execution |
| Run tests | `pytest src/ml/testing/test_ml.py -v` | 10s | Validation |
| Evaluate model | `python src/ml/evaluation/evaluate.py` | 5s | Performance check |
| Stop Streamlit | `Ctrl + C` | - | In terminal |
| Deactivate venv | `deactivate` | 1s | End of session |

---

## 🔗 Application URLs

| Page | URL |
|------|-----|
| Local | http://localhost:8501 |
| Network | http://YOUR_IP:8501 |
| Dashboard | http://localhost:8501/?page=Dashboard |
| Training | http://localhost:8501/?page=Training |
| Optimization | http://localhost:8501/?page=Optimization |
| Analytics | http://localhost:8501/?page=Analytics |

---

## 📝 Key Parameters

### Model Configuration
```python
n_slots: int = 30              # Number of warehouse slots
n_items: int = 2000            # Number of products
test_size: float = 0.2         # 20% test, 80% train
max_depth: int = 6             # Tree depth
learning_rate: float = 0.08    # Shrinkage factor
max_iter: int = 250            # Boosting iterations
```

### Data Ranges
```python
Slot dimensions: 1.2-2.0m x 0.8-1.2m x 1.5-2.4m
Item dimensions: 0.4-1.6m x 0.3-1.2m x 0.3-2.0m
Item weight: 20-900 kg
Slot distance: 5-60 meters
Item demand: 5, 10, 20, 30, 50, 80, 120 units/month
```

---

## ✅ Verification Checklist

After running the project:

- [ ] Virtual environment activated (venv shown in terminal)
- [ ] All dependencies installed (pip install -r requirements.txt)
- [ ] Streamlit started (http://localhost:8501 opens)
- [ ] Dashboard page shows statistics
- [ ] Training completes successfully (MAE displayed)
- [ ] Model saved (check src/models/slotting_model.joblib exists)
- [ ] Optimization page shows recommendations
- [ ] Analytics page shows performance metrics
- [ ] Tests pass (pytest output shows PASSED)

---

## 📚 Additional Resources

### View Database
```bash
# Use sqlite3 to inspect database
sqlite3 src/data/warehouse.db

# List tables
.tables

# View slots
SELECT * FROM slots LIMIT 5;

# View items
SELECT * FROM items LIMIT 5;

# Exit
.quit
```

### View Model Information
```python
import joblib
import pandas as pd

# Load model
bundle = joblib.load('src/models/slotting_model.joblib')

# Get model
model = bundle['model']

# Get features
features = bundle['feature_cols']
print(f"Features: {features}")

# Get feature importance
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")
```

---

## 🎉 Success Indicators

✅ **Project Running Successfully When:**
- Streamlit dashboard displays without errors
- Training completes with MAE < 0.05
- Slot match accuracy > 80%
- Optimization page shows recommendations
- All tests pass (PASSED status)
- Model file exists (slotting_model.joblib)
- Database file exists (warehouse.db)

---
