# 🌾 AgroSmart: AI Crop Recommendation & Precision Farming Agent

An end-to-end Machine Learning and AI Farming Advisory System built with **Python**, **Scikit-learn**, **XGBoost**, and **Streamlit**. Designed as a comprehensive, beginner-friendly student mini project that bridges advanced predictive modeling with practical agronomic decision-making.

---

## 📌 Project Overview

Agriculture faces major challenges from climate volatility, suboptimal soil nutrient management, and improper crop selection. **AgroSmart** tackles this by acting as an intelligent farming advisory agent:

1. **Phase 1 – Machine Learning Benchmark:** Trains and compares 5 leading classification algorithms (**Random Forest, XGBoost, Support Vector Machine, K-Nearest Neighbors, and Multi-Layer Perceptron**) on the benchmark Kaggle Crop Recommendation Dataset. Evaluates each model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices, and persists the best performing model.
2. **Phase 2 – AI Farming Agent & Interactive Dashboard:** Delivers a modern agricultural dashboard styled in organic greens and earthy tones. It takes real-time soil (N, P, K, pH) and micro-climatic (temperature, humidity, rainfall) inputs, recommends optimal crops with prediction confidence, highlights top alternative crops, provides soil health audits, flags agro-climatic risks, calculates fertilizer deficits (Urea, DAP, Potash), and logs cultivation history in an SQLite database.

---

## 📊 Dataset Details

- **Dataset Name:** Crop Recommendation Dataset (by Atharva Ingle)
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Observations:** 2,200 field records
- **Classes:** 22 distinct crops (100 samples per crop class):
  `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`
- **Features (7):**
  - **N:** Ratio of Nitrogen content in soil ($kg/ha$)
  - **P:** Ratio of Phosphorus content in soil ($kg/ha$)
  - **K:** Ratio of Potassium content in soil ($kg/ha$)
  - **temperature:** Ambient temperature in degrees Celsius (°C)
  - **humidity:** Relative humidity in %
  - **ph:** Soil pH value (0 to 14)
  - **rainfall:** Precipitation in millimeters ($mm$)

> 💡 **Dataset Auto-Download:** The project automatically verifies and fetches the genuine Kaggle dataset upon first run. If offline, place `Crop_recommendation.csv` in the `data/` folder.

---

## 🏆 Model Benchmark Results

All 5 models were trained and benchmarked on the identical holdout test split (**80% train, 20% test**, stratified, `random_state=42`):

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **99.55%** | **99.57%** | **99.55%** | **99.55%** | 🥇 **Best Model (Saved)** |
| **XGBoost Classifier** | **99.09%** | **99.13%** | **99.09%** | **99.08%** | 🥈 Runner-up |
| **Support Vector Machine (RBF + Scaler)** | **98.86%** | **98.96%** | **98.86%** | **98.87%** | 🥉 Top Tier |
| **Multi-Layer Perceptron (MLP)** | **98.64%** | **98.66%** | **98.64%** | **98.64%** | Highly Accurate |
| **K-Nearest Neighbors (KNN)** | **98.18%** | **98.29%** | **98.18%** | **98.16%** | Solid Baseline |

---

## 📁 Project Directory Structure

```text
mini project/
│
├── data/
│   ├── Crop_recommendation.csv      # Kaggle dataset (2,200 rows x 8 cols)
│   └── cultivation_history.db       # Persistent SQLite history database
│
├── models/
│   ├── best_crop_model.joblib       # Best performing model (Random Forest)
│   ├── label_encoder.joblib         # Categorical crop label encoder
│   ├── all_trained_models.joblib    # All 5 trained model instances
│   ├── model_comparison_results.csv # Summary benchmark leaderboard
│   └── model_metrics.json           # Per-class classification reports & matrices
│
├── charts/
│   ├── model_comparison.png         # Multi-metric grouped benchmark bar chart
│   ├── confusion_matrix.png         # 22x22 Seaborn heatmap for best model
│   ├── feature_importance.png       # Agricultural feature importance ranking
│   └── correlation_matrix.png       # Soil & weather correlation heatmap
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset ingestion, bounds & preprocessing
│   ├── train_models.py              # ML training, cross-validation & artifact saving
│   ├── crop_advisor.py              # Agronomic knowledge base, risks & fertilizer math
│   └── db_manager.py                # SQLite repository for cultivation logging
│
├── app.py                           # Full modern Streamlit web application
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation & setup guide
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure **Python 3.10+** (tested up to Python 3.14) is installed on your computer.

### 2. Clone / Open Directory
Open PowerShell or your terminal in the project directory:
```powershell
cd "c:\Users\Megana\Documents\mini project"
```

### 3. Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚙️ Running the Project

### Step 1: Run Model Training & Evaluation (Phase 1)
To retrain the 5 algorithms and regenerate the evaluation charts:
```powershell
python -m src.train_models
```
*Outputs will be saved in `models/` and `charts/`.*

### Step 2: Launch the Streamlit Web Dashboard (Phase 2)
Launch the interactive web application:
```powershell
streamlit run app.py
```
Your default web browser will automatically open to `http://localhost:8501`.

---

## 🖥️ Application Sections

1. **🌾 Dashboard:**
   - Summary metric cards displaying dataset records, crop coverage, and feature counts.
   - Interactive histogram and distribution visualizer for any of the 7 agricultural parameters.
   - Environmental correlation matrix heatmap and crop taxonomy directory.
2. **📊 Model Performance:**
   - Full Leaderboard comparing Random Forest, XGBoost, SVM, KNN, and MLP across Accuracy, Precision, Recall, and F1.
   - High-resolution evaluation charts (`model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`).
   - Interactive Model Inspector: Select any model to view its complete Scikit-learn classification report.
3. **🎯 Crop Recommendation:**
   - Interactive sliders and numeric inputs for N, P, K, Temperature, Humidity, pH, and Rainfall.
   - **Quick Presets:** One-click soil profiles (Alluvial Paddy, Loamy Black, Sandy Arid, Orchard, Plantation).
   - Real-time prediction with **confidence percentage** and **Top 3 Candidate crops**.
   - **4 In-Depth Advisory Tabs:**
     - *📖 Cultivation Guidance:* Duration, soil type, irrigation schedule, fertilizer protocol, and yield expectations.
     - *🧪 Soil Health Analysis:* Score (0-100), nutrient deficit/excess diagnosis, and lime/gypsum remediation.
     - *⚠️ Agro-Climatic Risks:* Early warnings for heat stress, chilling injury, drought, waterlogging, or fungal moisture.
     - *🌱 Fertilizer Calculator:* Translates exact NPK deficits into commercial 50-kg bags of **Urea**, **DAP**, and **MOP (Potash)**.
4. **🤖 AI Farming Assistant:**
   - Conversational agricultural advisor addressing questions on pH amendment, organic pest remedies, water conservation, and crop rotation.
   - Reference seasonal cropping calendar (Kharif, Rabi, Zaid) and soil pH scale.
5. **📋 Cultivation History:**
   - Persistent SQLite table recording all generated recommendations with timestamps and field notes.
   - Filter records by crop name or search history.
   - Single-click **CSV Export** for farmer record keeping.
   - Record deletion and history clearing tools.

---

## 🎓 Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Visualization:** Matplotlib, Seaborn
- **Web UI & Dashboard:** Streamlit
- **Model Serialization:** Joblib
- **Database:** SQLite

---

## 💡 Viva / Presentation Key Talking Points

- **Why did Random Forest perform best?**
  Random Forest creates an ensemble of decorrelated decision trees, making it exceptionally robust against non-linear thresholds in agricultural parameters (e.g., specific crops thrive in narrow temperature and rainfall bands).
- **Why did SVM, KNN, and MLP use Pipelines?**
  Features like Rainfall (20-300 mm) and Potassium (5-205 kg/ha) have significantly larger numerical scales than pH (3.5-9.9). Using `StandardScaler()` inside a Scikit-learn `Pipeline` prevents data leakage and ensures distance-based metrics are not dominated by large-scale features.
- **How is data leakage avoided?**
  A stratified 80/20 train-test split (`random_state=42`) is applied uniformly across all models, ensuring that each class is represented proportionally in both partitions.
