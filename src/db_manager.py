"""
Database Manager for Cultivation History
Stores and manages crop recommendation logs using SQLite.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "cultivation_history.db")


def get_connection(db_path: str = DB_PATH):
    """Establishes and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def init_db(db_path: str = DB_PATH):
    """Initializes the database schema if it doesn't already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cultivation_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        nitrogen REAL NOT NULL,
        phosphorus REAL NOT NULL,
        potassium REAL NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        predicted_crop TEXT NOT NULL,
        confidence REAL NOT NULL,
        yield_estimate TEXT,
        soil_rating TEXT,
        notes TEXT
    );
    """)
    conn.commit()
    conn.close()


def save_record(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    predicted_crop: str,
    confidence: float,
    yield_estimate: str = "",
    soil_rating: str = "",
    notes: str = "",
    db_path: str = DB_PATH
) -> int:
    """
    Inserts a new cultivation recommendation record into the database.
    Returns the newly inserted record id.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO cultivation_records (
        timestamp, nitrogen, phosphorus, potassium,
        temperature, humidity, ph, rainfall,
        predicted_crop, confidence, yield_estimate, soil_rating, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall),
        str(predicted_crop).strip().lower(),
        float(confidence),
        str(yield_estimate),
        str(soil_rating),
        str(notes)
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_all_records(crop_filter: str = None, limit: int = 500, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Fetches cultivation history records as a pandas DataFrame.
    Supports optional filtering by crop name.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    query = "SELECT * FROM cultivation_records"
    params = []

    if crop_filter and crop_filter != "All":
        query += " WHERE LOWER(predicted_crop) = LOWER(?)"
        params.append(crop_filter)

    query += " ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_record(record_id: int, db_path: str = DB_PATH) -> bool:
    """Deletes a specific record by its ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cultivation_records WHERE id = ?", (record_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def clear_history(db_path: str = DB_PATH) -> bool:
    """Deletes all records from the cultivation records table."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cultivation_records")
    conn.commit()
    conn.close()
    return True


def get_history_summary(db_path: str = DB_PATH) -> dict:
    """Returns aggregated summary metrics from the saved cultivation records."""
    df = get_all_records(db_path=db_path)
    if df.empty:
        return {
            "total_records": 0,
            "top_crop": "None",
            "avg_confidence": 0.0,
            "crops_count": 0
        }

    return {
        "total_records": len(df),
        "top_crop": df["predicted_crop"].mode()[0] if not df["predicted_crop"].empty else "None",
        "avg_confidence": round(float(df["confidence"].mean()), 2),
        "crops_count": int(df["predicted_crop"].nunique())
    }


if __name__ == "__main__":
    init_db()
    test_id = save_record(
        nitrogen=90, phosphorus=42, potassium=43,
        temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.0,
        predicted_crop="rice", confidence=99.4, yield_estimate="4.5 t/ha",
        soil_rating="Good", notes="Automated test insertion"
    )
    print(f"Inserted record ID: {test_id}")
    history_df = get_all_records()
    print("History table preview:")
    print(history_df.head(2))
