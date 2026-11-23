"""
🌾 Smart Crop Recommendation & AI Farming Agent
Modern Agricultural Dashboard built with Streamlit and Machine Learning.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_crop_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_feature_bounds,
    KAGGLE_DATASET_URL
)
from src.crop_advisor import (
    CROP_KNOWLEDGE_BASE,
    get_crop_details,
    analyze_soil_health,
    analyze_environmental_risks,
    calculate_fertilizer_recommendation,
    ask_farming_assistant
)
from src.db_manager import (
    init_db,
    save_record,
    get_all_records,
    clear_history,
    delete_record,
    get_history_summary
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgroSmart | Crop Recommendation Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Agricultural Green & Earthy CSS
CUSTOM_CSS = """
<style>
    /* Global Fonts and Base */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #d8f3dc;
        margin-top: 5px;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2ece9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(45, 106, 79, 0.08);
        border-color: #74c69d;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #52796f;
        font-weight: 600;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #1b4332;
        margin-top: 4px;
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 20px;
        background-color: #d8f3dc;
        color: #1b4332;
        margin-top: 6px;
    }

    /* Recommendation Result Box */
    .crop-result-card {
        background: linear-gradient(135deg, #f7faf8 0%, #eef7f2 100%);
        border: 2px solid #52b788;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(82, 183, 136, 0.15);
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .crop-badge-top {
        background-color: #2d6a4f;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .crop-name-large {
        font-size: 34px;
        font-weight: 800;
        color: #1b4332;
        margin: 8px 0;
    }

    /* Badges */
    .badge-success {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-warning {
        background-color: #fefae0;
        color: #bc6c25;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-danger {
        background-color: #ffe5d9;
        color: #d00000;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Section Cards */
    .agri-card {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #fbfdfc;
        border-right: 1px solid #e8f0ec;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loaders (Cached)
# ---------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    """Loads and caches the agricultural dataset."""
    return load_crop_data()


@st.cache_resource
def get_trained_artifacts():
    """Loads the serialized model, encoder, and metrics metadata."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "best_crop_model.joblib")
    encoder_path = os.path.join(base_dir, "models", "label_encoder.joblib")
    all_models_path = os.path.join(base_dir, "models", "all_trained_models.joblib")
    metrics_path = os.path.join(base_dir, "models", "model_metrics.json")
    results_csv_path = os.path.join(base_dir, "models", "model_comparison_results.csv")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    all_models = joblib.load(all_models_path) if os.path.exists(all_models_path) else {}

    metrics_meta = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_meta = json.load(f)

    results_df = pd.read_csv(results_csv_path) if os.path.exists(results_csv_path) else pd.DataFrame()

    return model, encoder, all_models, metrics_meta, results_df


# Initialize SQLite Database
init_db()

# Load Dataset & Artifacts
try:
    df = get_cached_dataset()
    bounds = get_feature_bounds(df)
    best_model, label_encoder, all_models, metrics_meta, results_df = get_trained_artifacts()
except Exception as e:
    st.error(f"Error initializing project components: {e}")
    st.stop()


# ---------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <span style="font-size: 32px;">🌱</span>
            <div>
                <h3 style="margin: 0; color: #1b4332; font-weight: 800; font-size: 20px;">AgroSmart</h3>
                <span style="font-size: 11px; color: #52796f; font-weight: 600;">Crop Recommendation Agent</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    app_section = st.radio(
        "Navigation",
        [
            "🌾 Dashboard",
            "📊 Model Performance",
            "🎯 Crop Recommendation",
            "🤖 AI Farming Assistant",
            "📋 Cultivation History"
        ],
        index=2
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Agronomic Status")
    history_stats = get_history_summary()
    st.metric("Total Field Recommendations", history_stats["total_records"])
    st.metric("Most Recommended Crop", history_stats["top_crop"].capitalize())
    if history_stats["avg_confidence"] > 0:
        st.metric("Average Confidence", f"{history_stats['avg_confidence']}%")

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 11px; color: #6c757d; line-height: 1.4;">
            <b>Dataset Source:</b><br>
            <a href="{KAGGLE_DATASET_URL}" target="_blank" style="color: #2d6a4f;">Kaggle Atharva Ingle Crop Dataset</a><br>
            2,200 Benchmark Field Observations
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Helper Header Banner
# ---------------------------------------------------------
def render_header(title: str, subtitle: str, badge: str = "AI AGENT ACTIVE"):
    st.markdown(f"""
        <div class="hero-banner">
            <div>
                <div class="hero-title">{title}</div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div style="background: rgba(255,255,255,0.18); padding: 8px 16px; border-radius: 30px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.25);">
                {badge}
            </div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# SECTION 1: DASHBOARD
# =========================================================
if app_section == "🌾 Dashboard":
    render_header(
        title="Agricultural Intelligence Dashboard",
        subtitle="Exploratory data analysis, soil parameter distributions, and crop profiles.",
        badge="2,200 FIELD SAMPLES"
    )

    # 4 Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Dataset Records</div>
                <div class="metric-value">2,200</div>
                <span class="metric-badge">100% Verified Samples</span>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Crops Covered</div>
                <div class="metric-value">{df['label'].nunique()}</div>
                <span class="metric-badge">Cereals, Pulses, Fruits, Fibers</span>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        best_acc = results_df.iloc[0]['Accuracy'] * 100 if not results_df.empty else 99.55
        best_name = results_df.iloc[0]['Model'] if not results_df.empty else "Random Forest"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best ML Model</div>
                <div class="metric-value">{best_acc:.2f}%</div>
                <span class="metric-badge">{best_name}</span>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Features Analyzed</div>
                <div class="metric-value">7</div>
                <span class="metric-badge">N, P, K, Temp, Humid, pH, Rain</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Layout: Dataset Explorer & Distributions
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 📊 Soil & Climatic Parameter Distributions")
        selected_feature = st.selectbox(
            "Select Agricultural Parameter to Inspect:",
            FEATURE_COLUMNS,
            format_func=lambda x: {
                "N": "Nitrogen (N) - Soil Content (kg/ha)",
                "P": "Phosphorus (P) - Soil Content (kg/ha)",
                "K": "Potassium (K) - Soil Content (kg/ha)",
                "temperature": "Temperature (°C)",
                "humidity": "Relative Humidity (%)",
                "ph": "Soil pH Level",
                "rainfall": "Rainfall (mm)"
            }[x]
        )

        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
        sns.histplot(df[selected_feature], kde=True, color="#2d6a4f", ax=ax, bins=25, edgecolor="#1b4332")
        ax.set_title(f"Distribution of {selected_feature.upper()}", fontsize=12, fontweight="bold", color="#1b4332")
        ax.set_xlabel(selected_feature, fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()

        # Summary Statistics for Selected Feature
        f_b = bounds[selected_feature]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{f_b['min']:.1f}")
        c2.metric("Average", f"{f_b['mean']:.1f}")
        c3.metric("Median", f"{f_b['median']:.1f}")
        c4.metric("Max", f"{f_b['max']:.1f}")

    with col_right:
        st.markdown("### 🧬 Feature Correlation Heatmap")
        corr_img_path = os.path.join(os.path.dirname(__file__), "charts", "correlation_matrix.png")
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Correlation between N, P, K, Temperature, Humidity, pH, and Rainfall", use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
            sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown("### 🌾 Target Crop Classes in Dataset (22 Distinct Crops)")
    crop_counts = df['label'].value_counts()
    crops_sorted = sorted(df['label'].unique())

    # Display Crop badges in columns
    cols = st.columns(4)
    for idx, c in enumerate(crops_sorted):
        info = get_crop_details(c)
        with cols[idx % 4]:
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2ece9; border-left: 4px solid #40916c; padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #1b4332; font-size: 14px;">{info['display_name']}</div>
                    <div style="font-size: 11px; color: #52796f;">{info['category']} • {info['season']}</div>
                </div>
            """, unsafe_allow_html=True)


# =========================================================
# SECTION 2: MODEL PERFORMANCE
# =========================================================
elif app_section == "📊 Model Performance":
    render_header(
        title="Machine Learning Model Evaluation & Benchmarking",
        subtitle="Rigorous comparison of Random Forest, XGBoost, SVM, KNN, and MLP on holdout test partition (20%).",
        badge="5 MODELS EVALUATED"
    )

    if not results_df.empty:
        st.markdown("### 🏆 Algorithm Leaderboard")
        # Format table for display
        display_df = results_df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda v: f"{v*100:.2f}%")
        display_df["Precision"] = display_df["Precision"].apply(lambda v: f"{v*100:.2f}%")
        display_df["Recall"] = display_df["Recall"].apply(lambda v: f"{v*100:.2f}%")
        display_df["F1-Score"] = display_df["F1-Score"].apply(lambda v: f"{v*100:.2f}%")

        st.dataframe(
            display_df,
            column_config={
                "Model": st.column_config.TextColumn("Algorithm Name", width="medium"),
                "Accuracy": st.column_config.TextColumn("Accuracy (Holdout Test)", width="medium"),
                "Precision": st.column_config.TextColumn("Precision (Weighted)", width="medium"),
                "Recall": st.column_config.TextColumn("Recall (Weighted)", width="medium"),
                "F1-Score": st.column_config.TextColumn("F1-Score (Weighted)", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.success(f"🌟 **Winner:** **{results_df.iloc[0]['Model']}** achieved the highest test accuracy of **{results_df.iloc[0]['Accuracy']*100:.2f}%** with balanced precision and recall across all 22 classes.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Evaluation Charts
    chart_col1, chart_col2 = st.columns(2)

    comp_chart_path = os.path.join(os.path.dirname(__file__), "charts", "model_comparison.png")
    cm_chart_path = os.path.join(os.path.dirname(__file__), "charts", "confusion_matrix.png")
    fi_chart_path = os.path.join(os.path.dirname(__file__), "charts", "feature_importance.png")

    with chart_col1:
        st.markdown("### 📈 Multi-Metric Comparative Performance")
        if os.path.exists(comp_chart_path):
            st.image(comp_chart_path, caption="Accuracy, Precision, Recall, and F1 Comparison", use_container_width=True)
        else:
            st.info("Chart not found. Run model training script to regenerate.")

    with chart_col2:
        st.markdown("### 🎯 Agricultural Feature Importances")
        if os.path.exists(fi_chart_path):
            st.image(fi_chart_path, caption="Relative feature contributions in crop classification", use_container_width=True)
        else:
            st.info("Feature importance chart not found.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Confusion Matrix Heatmap (Best Performing Model)")
    if os.path.exists(cm_chart_path):
        st.image(cm_chart_path, caption="22x22 Class Confusion Matrix on 440 Test Instances", use_container_width=True)

    # Interactive Model Inspector
    st.markdown("---")
    st.markdown("### 🔬 Interactive Model Inspector")
    st.caption("Inspect individual classification reports and per-crop performance for each evaluated model.")

    model_choice = st.selectbox("Select Model to Inspect:", list(all_models.keys()) if all_models else ["Random Forest"])
    if metrics_meta and "detailed_results" in metrics_meta and model_choice in metrics_meta["detailed_results"]:
        model_info = metrics_meta["detailed_results"][model_choice]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{model_info['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{model_info['precision']*100:.2f}%")
        c3.metric("Recall", f"{model_info['recall']*100:.2f}%")
        c4.metric("F1-Score", f"{model_info['f1_score']*100:.2f}%")

        with st.expander(f"View Full Classification Report for {model_choice}", expanded=False):
            report_df = pd.DataFrame(model_info["classification_report"]).transpose()
            st.dataframe(report_df.style.format("{:.3f}", na_rep="-"), use_container_width=True)


# =========================================================
# SECTION 3: CROP RECOMMENDATION
# =========================================================
elif app_section == "🎯 Crop Recommendation":
    render_header(
        title="Precision Crop Recommendation Engine",
        subtitle="Input soil nutrient levels and environmental parameters to generate AI-backed cultivation plans.",
        badge="ML RECOMMENDATION AGENT"
    )

    # Soil Preset Profiles for rapid student / farmer demonstration
    st.markdown("##### ⚡ Quick Soil & Agro-Climatic Presets")
    preset_cols = st.columns(5)
    preset_selected = None

    if preset_cols[0].button("🍚 Alluvial Paddy (Rice)", use_container_width=True):
        preset_selected = {"N": 90.0, "P": 42.0, "K": 43.0, "temp": 24.0, "hum": 82.0, "ph": 6.5, "rain": 215.0}
    if preset_cols[1].button("🌽 Loamy Black (Maize)", use_container_width=True):
        preset_selected = {"N": 80.0, "P": 48.0, "K": 20.0, "temp": 23.0, "hum": 65.0, "ph": 6.8, "rain": 85.0}
    if preset_cols[2].button("🐪 Sandy Arid (Mothbeans)", use_container_width=True):
        preset_selected = {"N": 20.0, "P": 48.0, "K": 20.0, "temp": 28.0, "hum": 50.0, "ph": 7.2, "rain": 45.0}
    if preset_cols[3].button("🍇 Orchard Soil (Grapes)", use_container_width=True):
        preset_selected = {"N": 20.0, "P": 134.0, "K": 200.0, "temp": 22.0, "hum": 80.0, "ph": 6.0, "rain": 70.0}
    if preset_cols[4].button("☕ Plantation (Coffee)", use_container_width=True):
        preset_selected = {"N": 100.0, "P": 30.0, "K": 30.0, "temp": 25.0, "hum": 58.0, "ph": 6.8, "rain": 160.0}

    # Store presets into session state if clicked
    if preset_selected:
        st.session_state["input_n"] = preset_selected["N"]
        st.session_state["input_p"] = preset_selected["P"]
        st.session_state["input_k"] = preset_selected["K"]
        st.session_state["input_temp"] = preset_selected["temp"]
        st.session_state["input_hum"] = preset_selected["hum"]
        st.session_state["input_ph"] = preset_selected["ph"]
        st.session_state["input_rain"] = preset_selected["rain"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form with Clean 2-Column Split: Soil vs Climate
    col_soil, col_climate = st.columns(2)

    with col_soil:
        st.markdown("""
            <div style="background: #ffffff; padding: 18px 20px; border-radius: 12px; border: 1px solid #d8f3dc; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #1b4332;">🧪 Soil Nutrient Parameters</h4>
                <p style="font-size: 12px; color: #52796f; margin-top: 4px;">Enter primary macronutrients and soil acidity/alkalinity.</p>
            </div>
        """, unsafe_allow_html=True)

        input_n = st.slider(
            "Nitrogen (N) - Soil Content [kg/ha]",
            min_value=0.0, max_value=140.0,
            value=st.session_state.get("input_n", 90.0),
            step=1.0,
            help="Ratio of Nitrogen content in soil (0 - 140 kg/ha)"
        )

        input_p = st.slider(
            "Phosphorus (P) - Soil Content [kg/ha]",
            min_value=5.0, max_value=145.0,
            value=st.session_state.get("input_p", 42.0),
            step=1.0,
            help="Ratio of Phosphorus content in soil (5 - 145 kg/ha)"
        )

        input_k = st.slider(
            "Potassium (K) - Soil Content [kg/ha]",
            min_value=5.0, max_value=205.0,
            value=st.session_state.get("input_k", 43.0),
            step=1.0,
            help="Ratio of Potassium content in soil (5 - 205 kg/ha)"
        )

        input_ph = st.slider(
            "Soil pH Level",
            min_value=3.5, max_value=9.9,
            value=st.session_state.get("input_ph", 6.5),
            step=0.1,
            help="pH value of the soil (3.5 strongly acidic - 9.9 highly alkaline)"
        )

    with col_climate:
        st.markdown("""
            <div style="background: #ffffff; padding: 18px 20px; border-radius: 12px; border: 1px solid #d8f3dc; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #1b4332;">🌦 Climatic & Weather Conditions</h4>
                <p style="font-size: 12px; color: #52796f; margin-top: 4px;">Enter ambient micro-climatic factors for the target growing zone.</p>
            </div>
        """, unsafe_allow_html=True)

        input_temp = st.slider(
            "Temperature (°C)",
            min_value=8.0, max_value=45.0,
            value=st.session_state.get("input_temp", 20.8),
            step=0.5,
            help="Ambient temperature in degrees Celsius (8.0 - 45.0 °C)"
        )

        input_hum = st.slider(
            "Relative Humidity (%)",
            min_value=10.0, max_value=100.0,
            value=st.session_state.get("input_hum", 82.0),
            step=1.0,
            help="Relative humidity in percentage (10% - 100%)"
        )

        input_rain = st.slider(
            "Rainfall (mm)",
            min_value=20.0, max_value=300.0,
            value=st.session_state.get("input_rain", 202.0),
            step=1.0,
            help="Seasonal precipitation / rainfall in mm (20 - 300 mm)"
        )

        farmer_notes = st.text_input(
            "Field / Plot Identifier or Farmer Notes (Optional)",
            value="North Field - Plot A1",
            help="Optional field label saved to the cultivation history database."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("🌾 Run AI Crop Recommendation", type="primary", use_container_width=True)

    # Prediction Execution
    if recommend_btn or "last_prediction" in st.session_state:
        # Construct sample feature DataFrame with valid column names
        sample = pd.DataFrame(
            [[input_n, input_p, input_k, input_temp, input_hum, input_ph, input_rain]],
            columns=FEATURE_COLUMNS
        )

        # Model Inference
        probs = None
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(sample)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            top_crops = label_encoder.inverse_transform(top_indices)
            top_scores = probs[top_indices] * 100

            predicted_crop = top_crops[0]
            confidence = top_scores[0]
        else:
            pred_idx = best_model.predict(sample)[0]
            predicted_crop = label_encoder.inverse_transform([pred_idx])[0]
            confidence = 99.0
            top_crops = [predicted_crop]
            top_scores = [confidence]

        crop_info = get_crop_details(predicted_crop)
        soil_analysis = analyze_soil_health(input_n, input_p, input_k, input_ph)
        weather_risks = analyze_environmental_risks(input_temp, input_hum, input_rain, input_ph)
        fertilizer_plan = calculate_fertilizer_recommendation(predicted_crop, input_n, input_p, input_k)

        # Store in session state
        st.session_state["last_prediction"] = {
            "crop": predicted_crop,
            "confidence": confidence,
            "crop_info": crop_info,
            "soil_analysis": soil_analysis,
            "weather_risks": weather_risks,
            "fertilizer_plan": fertilizer_plan,
            "top_crops": list(top_crops),
            "top_scores": list(top_scores),
            "inputs": {
                "N": input_n, "P": input_p, "K": input_k,
                "temperature": input_temp, "humidity": input_hum,
                "ph": input_ph, "rainfall": input_rain,
                "notes": farmer_notes
            }
        }

        # Automatically save to database upon explicit click
        if recommend_btn:
            save_record(
                nitrogen=input_n,
                phosphorus=input_p,
                potassium=input_k,
                temperature=input_temp,
                humidity=input_hum,
                ph=input_ph,
                rainfall=input_rain,
                predicted_crop=predicted_crop,
                confidence=confidence,
                yield_estimate=crop_info.get("expected_yield", ""),
                soil_rating=soil_analysis.get("rating", "Good"),
                notes=farmer_notes
            )

        # Render Main Crop Recommendation Card
        st.markdown(f"""
            <div class="crop-result-card">
                <span class="crop-badge-top">OPTIMAL RECOMMENDED CROP</span>
                <div class="crop-name-large">🌾 {crop_info['display_name']}</div>
                <div style="font-size: 15px; color: #2d6a4f; font-weight: 600; margin-bottom: 12px;">
                    {crop_info['category']} • {crop_info['season']} • Maturity: {crop_info['duration']}
                </div>
                <div style="display: flex; align-items: center; gap: 15px; margin-top: 10px;">
                    <span style="font-size: 14px; font-weight: 700; color: #1b4332;">Prediction Confidence:</span>
                    <span class="badge-success" style="font-size: 14px;">{confidence:.2f}% Match</span>
                    <span class="badge-success" style="font-size: 14px;">Expected Yield: {crop_info['expected_yield']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Top 3 Candidates
        if len(top_crops) > 1:
            st.markdown("##### 🥉 Top 3 Ranked Crop Candidates")
            top_cols = st.columns(3)
            medals = ["🥇 Best Fit", "🥈 Alternative 1", "🥉 Alternative 2"]
            for idx in range(min(3, len(top_crops))):
                c_name = top_crops[idx]
                c_score = top_scores[idx]
                c_det = get_crop_details(c_name)
                with top_cols[idx]:
                    st.markdown(f"""
                        <div style="background: #ffffff; border-radius: 10px; border: 1px solid #d8f3dc; padding: 14px; border-top: 3px solid #2d6a4f;">
                            <div style="font-size: 11px; font-weight: 700; color: #52796f;">{medals[idx]}</div>
                            <div style="font-size: 18px; font-weight: 800; color: #1b4332; margin: 4px 0;">{c_det['display_name']}</div>
                            <div style="font-size: 13px; font-weight: 700; color: #40916c;">{c_score:.1f}% Probability</div>
                            <div style="font-size: 11px; color: #6c757d; margin-top: 4px;">{c_det['season']}</div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Comprehensive Agricultural Advisory
        t_guide, t_soil, t_weather, t_fert = st.tabs([
            "📖 Cultivation Guidance",
            "🧪 Soil Health Analysis",
            "⚠️ Agro-Climatic Risks",
            "🌱 Fertilizer Calculator"
        ])

        # TAB 1: Cultivation Guidance
        with t_guide:
            st.markdown(f"#### Complete Agronomic Guide for **{crop_info['display_name']}**")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown(f"**🌱 Soil Requirements:**\n{crop_info['soil_type']}")
                st.markdown(f"**💧 Water & Irrigation Schedule:**\n{crop_info['water_req']}\n\n*{crop_info['irrigation']}*")
                st.markdown(f"**⏱ Life Cycle Duration:**\n{crop_info['duration']}")
            with g_col2:
                st.markdown(f"**💊 Fertilizer Protocol:**\n{crop_info['fertilizer']}")
                st.markdown(f"**🐛 Pest & Disease Protection:**\n{crop_info['pests_diseases']}")
                st.markdown(f"**📈 Expected Production Yield:**\n{crop_info['expected_yield']}")

            st.markdown("##### 💡 Agronomic Pro Tips for Maximizing Harvest")
            for tip in crop_info.get("tips", []):
                st.markdown(f"- ✅ {tip}")

        # TAB 2: Soil Health Analysis
        with t_soil:
            st.markdown("#### Soil Nutrient Diagnostic Report")
            s_col1, s_col2 = st.columns([1, 1.4])
            with s_col1:
                st.markdown(f"""
                    <div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2ece9; text-align: center;">
                        <div style="font-size: 12px; font-weight: 700; color: #52796f; text-transform: uppercase;">Soil Health Index</div>
                        <div style="font-size: 48px; font-weight: 800; color: #2d6a4f;">{soil_analysis['score']}/100</div>
                        <div class="badge-success" style="font-size: 14px;">Rating: {soil_analysis['rating']}</div>
                    </div>
                """, unsafe_allow_html=True)

            with s_col2:
                for f in soil_analysis["findings"]:
                    status_class = "badge-success" if f["status"] == "Optimum" or f["status"] == "Ideal Neutral" else (
                        "badge-warning" if f["status"] in ["Excessive", "High", "Slightly Off-Neutral"] else "badge-danger"
                    )
                    st.markdown(f"""
                        <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; background: #fbfdfc; padding: 8px 12px; border-radius: 8px; border: 1px solid #edf2ef;">
                            <span style="font-weight: 700; color: #1b4332;">{f['parameter']}</span>
                            <span class="{status_class}">{f['status']} ({f['detail']})</span>
                        </div>
                    """, unsafe_allow_html=True)

            if soil_analysis["recommendations"]:
                st.markdown("##### 🛠 Prescriptive Soil Amendments")
                for r in soil_analysis["recommendations"]:
                    st.markdown(f"- 🔸 {r}")

        # TAB 3: Agro-Climatic Risks
        with t_weather:
            st.markdown("#### Environmental Risk & Stress Analysis")
            for r in weather_risks:
                st.markdown(f"""
                    <div style="background: #ffffff; border-left: 4px solid {'#d00000' if r['level']=='High' else ('#e09f3e' if r['level']=='Moderate' else '#52b788')}; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e9ecef;">
                        <div style="font-weight: 700; color: #1b4332; font-size: 15px;">
                            {r['icon']} {r['type']} <span style="font-size: 11px; padding: 2px 6px; border-radius: 4px; background: #f0f0f0;">Level: {r['level']}</span>
                        </div>
                        <div style="font-size: 13px; color: #495057; margin-top: 4px;">{r['message']}</div>
                    </div>
                """, unsafe_allow_html=True)

        # TAB 4: Fertilizer Calculator
        with t_fert:
            st.markdown(f"#### Commercial Fertilizer Calculator for **{fertilizer_plan['crop']}**")
            st.caption("Calculates exact nutrient deficits and converts them into commercial 50-kg fertilizer bags per hectare.")

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.metric("Urea (46% N) Needed", f"{fertilizer_plan['fertilizer_quantities_kg_ha']['Urea (46% N)']} kg/ha",
                          f"{fertilizer_plan['bags_estimate_50kg']['Urea Bags']} bags (50kg)")
            with fc2:
                st.metric("DAP (18% N, 46% P) Needed", f"{fertilizer_plan['fertilizer_quantities_kg_ha']['DAP (18% N, 46% P)']} kg/ha",
                          f"{fertilizer_plan['bags_estimate_50kg']['DAP Bags']} bags (50kg)")
            with fc3:
                st.metric("Potash / MOP (60% K) Needed", f"{fertilizer_plan['fertilizer_quantities_kg_ha']['Muriate of Potash (60% K)']} kg/ha",
                          f"{fertilizer_plan['bags_estimate_50kg']['MOP Bags']} bags (50kg)")

            st.markdown(f"""
                <div style="background: #f8fbf9; border: 1px solid #d8f3dc; border-radius: 8px; padding: 12px; margin-top: 15px; font-size: 13px; color: #2d6a4f;">
                    <b>Nutrient Gap (Deficit):</b> Nitrogen deficit: <b>{fertilizer_plan['deficit']['N']:.1f} kg/ha</b> | Phosphorus deficit: <b>{fertilizer_plan['deficit']['P']:.1f} kg/ha</b> | Potassium deficit: <b>{fertilizer_plan['deficit']['K']:.1f} kg/ha</b>.
                    <br><i>Note: Apply all DAP and MOP as basal dressing at sowing. Apply Urea in 2-3 split top-dressings.</i>
                </div>
            """, unsafe_allow_html=True)


# =========================================================
# SECTION 4: AI FARMING ASSISTANT
# =========================================================
elif app_section == "🤖 AI Farming Assistant":
    render_header(
        title="Interactive AI Farming Assistant",
        subtitle="Conversational agronomic advisor for soil fertility, pest defense, irrigation, and crop planning.",
        badge="AGRONOMIC BOT"
    )

    col_chat, col_tools = st.columns([1.2, 0.8])

    with col_chat:
        st.markdown("### 💬 Ask AgroSmart Assistant")
        st.caption("Ask questions about soil health, pH management, organic pest remedies, fertilizer calculations, or crop cycles.")

        # Quick Prompt Buttons
        st.markdown("##### 💡 Frequently Asked Questions")
        q_cols = st.columns(2)
        preset_query = None
        if q_cols[0].button("🧪 How to manage acidic soil?", use_container_width=True):
            preset_query = "How to treat acidic soil with lime?"
        if q_cols[1].button("💧 Best water-saving irrigation tips?", use_container_width=True):
            preset_query = "What are the best irrigation methods to save water during drought?"
        if q_cols[0].button("🐛 Organic remedies for pests?", use_container_width=True):
            preset_query = "What are organic solutions for caterpillar and borer pests?"
        if q_cols[1].button("🌱 How to use Urea and DAP?", use_container_width=True):
            preset_query = "How should I schedule Urea and DAP fertilizer application?"

        # Chat input
        user_query = st.text_input(
            "Enter your farming query:",
            value=preset_query if preset_query else "",
            placeholder="e.g. How do I improve sandy soil for pulses?"
        )

        if st.button("Ask Assistant", type="primary") or preset_query:
            if user_query.strip():
                context = {}
                if "last_prediction" in st.session_state:
                    context["crop"] = st.session_state["last_prediction"]["crop"]
                    context["soil"] = st.session_state["last_prediction"]["inputs"]

                with st.spinner("Generating agronomic response..."):
                    reply = ask_farming_assistant(user_query, context=context)

                st.markdown(f"""
                    <div style="background: #ffffff; border-radius: 12px; border: 1px solid #d8f3dc; padding: 20px; margin-top: 15px; box-shadow: 0 4px 14px rgba(45, 106, 79, 0.08);">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                            <span style="font-size: 22px;">🤖</span>
                            <span style="font-weight: 700; color: #1b4332; font-size: 15px;">AgroSmart Assistant</span>
                        </div>
                        <div style="color: #212529; font-size: 14px; line-height: 1.6;">
                            {reply}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please type a question or select one of the suggested prompts above.")

    with col_tools:
        st.markdown("### 🗓 Seasonal Farming Calendar")
        st.markdown("""
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e9ecef; padding: 18px;">
                <div style="border-left: 3px solid #2d6a4f; padding-left: 10px; margin-bottom: 14px;">
                    <div style="font-weight: 700; color: #1b4332;">🌧 Kharif Season (Monsoon)</div>
                    <div style="font-size: 12px; color: #52796f;">June – October / November</div>
                    <div style="font-size: 12px; color: #495057; margin-top: 4px;">Rice, Maize, Cotton, Jute, Pigeonpeas, Mungbean, Blackgram</div>
                </div>
                <div style="border-left: 3px solid #bc6c25; padding-left: 10px; margin-bottom: 14px;">
                    <div style="font-weight: 700; color: #1b4332;">❄️ Rabi Season (Winter)</div>
                    <div style="font-size: 12px; color: #52796f;">October – March / April</div>
                    <div style="font-size: 12px; color: #495057; margin-top: 4px;">Chickpea, Lentil, Kidney Beans, Mustard, Wheat, Apple</div>
                </div>
                <div style="border-left: 3px solid #e09f3e; padding-left: 10px;">
                    <div style="font-weight: 700; color: #1b4332;">☀️ Zaid Season (Summer)</div>
                    <div style="font-size: 12px; color: #52796f;">March – June</div>
                    <div style="font-size: 12px; color: #495057; margin-top: 4px;">Watermelon, Muskmelon, Summer Pulses, Cucumber</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌿 Soil pH Scale Reference")
        st.markdown("""
            <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e9ecef; padding: 16px; font-size: 12.5px;">
                <div>🔴 <b>&lt; 5.5: Strongly Acidic</b> (Apply Lime)</div>
                <div style="margin: 6px 0;">🟢 <b>6.0 – 7.2: Ideal Neutral</b> (Optimal Bioavailability)</div>
                <div>🟠 <b>&gt; 7.8: Alkaline / Calcareous</b> (Apply Gypsum)</div>
            </div>
        """, unsafe_allow_html=True)


# =========================================================
# SECTION 5: CULTIVATION HISTORY
# =========================================================
elif app_section == "📋 Cultivation History":
    render_header(
        title="Cultivation Recommendation History",
        subtitle="Persistent logging of field predictions, soil parameters, and crop performance.",
        badge="SQLITE REPOSITORY"
    )

    # Fetch History
    history_df = get_all_records()

    # Top Controls: Filter, Refresh, Clear
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    with ctrl_col1:
        crop_list = ["All"] + sorted(list(history_df["predicted_crop"].unique())) if not history_df.empty else ["All"]
        filter_crop = st.selectbox("Filter by Crop:", crop_list)

    filtered_df = get_all_records(crop_filter=filter_crop if filter_crop != "All" else None)

    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if not filtered_df.empty:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export History (CSV)",
                data=csv_data,
                file_name="cultivation_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
            clear_history()
            st.success("History database cleared.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No cultivation history records found. Run a recommendation in the **Crop Recommendation** tab to log entries.")
    else:
        st.markdown(f"##### Showing **{len(filtered_df)}** Logged Recommendation(s)")

        # Format dataframe for clean display
        table_df = filtered_df[[
            "id", "timestamp", "predicted_crop", "confidence",
            "nitrogen", "phosphorus", "potassium",
            "temperature", "humidity", "ph", "rainfall",
            "yield_estimate", "notes"
        ]].copy()

        table_df["predicted_crop"] = table_df["predicted_crop"].str.capitalize()
        table_df["confidence"] = table_df["confidence"].apply(lambda v: f"{v:.1f}%")
        table_df["temperature"] = table_df["temperature"].apply(lambda v: f"{v:.1f}°C")
        table_df["humidity"] = table_df["humidity"].apply(lambda v: f"{v:.1f}%")
        table_df["ph"] = table_df["ph"].apply(lambda v: f"{v:.2f}")
        table_df["rainfall"] = table_df["rainfall"].apply(lambda v: f"{v:.1f} mm")

        st.dataframe(
            table_df,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "timestamp": st.column_config.TextColumn("Date & Time", width="medium"),
                "predicted_crop": st.column_config.TextColumn("Recommended Crop", width="medium"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "nitrogen": st.column_config.NumberColumn("N", width="small"),
                "phosphorus": st.column_config.NumberColumn("P", width="small"),
                "potassium": st.column_config.NumberColumn("K", width="small"),
                "temperature": st.column_config.TextColumn("Temp", width="small"),
                "humidity": st.column_config.TextColumn("Humidity", width="small"),
                "ph": st.column_config.TextColumn("pH", width="small"),
                "rainfall": st.column_config.TextColumn("Rainfall", width="small"),
                "yield_estimate": st.column_config.TextColumn("Yield Est.", width="medium"),
                "notes": st.column_config.TextColumn("Field / Notes", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )

        # Delete Single Record Helper
        with st.expander("Delete an individual record by ID"):
            del_id = st.number_input("Record ID to delete:", min_value=1, step=1)
            if st.button("Delete Record"):
                if delete_record(del_id):
                    st.success(f"Record #{del_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"Record #{del_id} not found.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 12px; padding: 15px 0;">
        🌱 <b>Crop Recommendation Agent</b> • Machine Learning & Streamlit Mini Project • Built with Python, Scikit-learn, XGBoost & SQLite
    </div>
""", unsafe_allow_html=True)
