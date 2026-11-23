"""
Comprehensive Automated Test Suite for Crop Recommendation Project
Tests data loading, model inference, crop advisor, and SQLite history database.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.data_loader import load_crop_data, preprocess_data, FEATURE_COLUMNS, TARGET_COLUMN
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
    get_history_summary
)


def test_data_loader():
    print("Testing data loader...")
    df = load_crop_data()
    assert df is not None, "DataFrame should not be None"
    assert df.shape == (2200, 8), f"Expected shape (2200, 8), got {df.shape}"
    assert df["label"].nunique() == 22, f"Expected 22 unique crops, got {df['label'].nunique()}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected nulls"
    print("  ✅ Data loader passed!")


def test_model_inference():
    print("Testing model inference...")
    base_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
    model_path = os.path.join(base_dir, "models", "best_crop_model.joblib")
    encoder_path = os.path.join(base_dir, "models", "label_encoder.joblib")

    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(encoder_path), f"Missing {encoder_path}"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Test sample for Rice: N=90, P=42, K=43, temp=20.8, hum=82.0, ph=6.5, rain=202.9
    sample_rice = pd.DataFrame([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]], columns=FEATURE_COLUMNS)
    pred_idx = model.predict(sample_rice)[0]
    pred_crop = encoder.inverse_transform([pred_idx])[0]
    assert pred_crop.lower() == "rice", f"Expected 'rice', got '{pred_crop}'"

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample_rice)[0]
        confidence = probs[pred_idx] * 100
        assert confidence > 70.0, f"Expected confidence > 70%, got {confidence:.2f}%"
        print(f"  Confidence for rice prediction: {confidence:.2f}%")

    # Test sample for Cotton: N=120, P=45, K=20, temp=25.0, hum=70.0, ph=7.0, rain=80.0
    sample_cotton = pd.DataFrame([[120, 45, 20, 25.0, 70.0, 7.0, 80.0]], columns=FEATURE_COLUMNS)
    pred_idx2 = model.predict(sample_cotton)[0]
    pred_crop2 = encoder.inverse_transform([pred_idx2])[0]
    print(f"  Prediction for N=120, P=45, K=20: {pred_crop2}")

    print("  ✅ Model inference passed!")


def test_crop_advisor():
    print("Testing crop advisor knowledge base...")
    assert len(CROP_KNOWLEDGE_BASE) == 22, f"Expected 22 crops, got {len(CROP_KNOWLEDGE_BASE)}"

    for crop in CROP_KNOWLEDGE_BASE:
        details = get_crop_details(crop)
        assert "display_name" in details, f"Missing display_name for {crop}"
        assert "fertilizer" in details, f"Missing fertilizer for {crop}"
        assert "irrigation" in details, f"Missing irrigation for {crop}"
        assert "duration" in details, f"Missing duration for {crop}"
        assert len(details.get("tips", [])) > 0, f"Missing tips for {crop}"

    # Test soil health
    soil_res = analyze_soil_health(N=30, P=20, K=20, ph=4.8)
    assert soil_res["score"] < 70, "Severely deficient soil should have low health score"

    # Test weather risks
    risks = analyze_environmental_risks(temperature=42.0, humidity=92.0, rainfall=310.0, ph=6.5)
    assert len(risks) >= 2, "Extreme weather should flag multiple risks"

    # Test fertilizer calculator
    fert_calc = calculate_fertilizer_recommendation("rice", N=50, P=20, K=20)
    assert fert_calc["fertilizer_quantities_kg_ha"]["Urea (46% N)"] > 0
    assert fert_calc["fertilizer_quantities_kg_ha"]["DAP (18% N, 46% P)"] > 0

    # Test assistant response
    resp = ask_farming_assistant("How to improve acidic soil?")
    assert "lime" in resp.lower() or "limestone" in resp.lower(), "Assistant response should discuss lime"

    print("  ✅ Crop advisor passed!")


def test_database():
    print("Testing SQLite history manager...")
    init_db()
    rec_id = save_record(
        nitrogen=85.0, phosphorus=40.0, potassium=42.0,
        temperature=22.5, humidity=80.0, ph=6.6, rainfall=210.0,
        predicted_crop="rice", confidence=98.5,
        yield_estimate="4.5 t/ha", soil_rating="Good", notes="Automated Test Record"
    )
    assert rec_id > 0, "Record ID should be positive integer"

    records = get_all_records()
    assert not records.empty, "Database should contain records"
    assert "predicted_crop" in records.columns

    stats = get_history_summary()
    assert stats["total_records"] >= 1
    print(f"  Total records in history: {stats['total_records']}")
    print("  ✅ Database manager passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 RUNNING COMPREHENSIVE PIPELINE VERIFICATION")
    print("=" * 60)
    test_data_loader()
    test_model_inference()
    test_crop_advisor()
    test_database()
    print("=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")
