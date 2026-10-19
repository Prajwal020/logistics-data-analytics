"""
Logistics Analytics Pipeline - Week 1 Strategic Implementation
Module: Data Ingestion, Metric Computation, and Feature Extraction
"""

import numpy as np
import pandas as pd


def calculate_haversine_distance(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Calculates great-circle distance between two geographic coordinates in kilometers."""
    r = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def load_data() -> pd.DataFrame:
    """Simulated shipment log (replace with pd.read_csv on Olist/DataCo in Week 2)."""
    data = {
        "shipment_id": [f"SHP_{i:04d}" for i in range(1, 6)],
        "carrier_id": ["Carrier_A", "Carrier_B", "Carrier_A", "Carrier_C", "Carrier_B"],
        "order_created_time": [
            "2026-10-01 07:45:00",
            "2026-10-01 08:20:00",
            "2026-10-01 09:10:00",
            "2026-10-01 10:30:00",
            "2026-10-01 12:00:00",
        ],
        "dispatch_time": [
            "2026-10-01 08:30:00",
            "2026-10-01 09:15:00",
            "2026-10-01 10:00:00",
            "2026-10-01 11:30:00",
            "2026-10-01 13:00:00",
        ],
        "promised_delivery_time": [
            "2026-10-01 12:30:00",
            "2026-10-01 14:15:00",
            "2026-10-01 13:00:00",
            "2026-10-01 18:30:00",
            "2026-10-01 16:00:00",
        ],
        "actual_delivery_time": [
            "2026-10-01 12:10:00",
            "2026-10-01 14:45:00",
            "2026-10-01 12:55:00",
            "2026-10-01 19:10:00",
            "2026-10-01 15:40:00",
        ],
        "origin_lat": [12.9716, 12.9716, 12.9352, 13.0827, 12.9716],
        "origin_lon": [77.5946, 77.5946, 77.6245, 80.2707, 77.5946],
        "dest_lat": [12.9279, 13.0358, 12.9716, 13.0405, 12.8995],
        "dest_lon": [77.6271, 77.5970, 77.5946, 80.2337, 77.5855],
        "operational_cost": [140.0, 220.0, 110.0, 310.0, 195.0],  # INR
    }
    return pd.DataFrame(data)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = [
        "order_created_time",
        "dispatch_time",
        "promised_delivery_time",
        "actual_delivery_time",
    ]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])

    hours = lambda delta: delta.dt.total_seconds() / 3600
    df["actual_transit_hours"] = hours(df["actual_delivery_time"] - df["dispatch_time"])
    df["promised_transit_hours"] = hours(df["promised_delivery_time"] - df["dispatch_time"])
    df["delay_hours"] = df["actual_transit_hours"] - df["promised_transit_hours"]
    df["ofct_hours"] = hours(df["actual_delivery_time"] - df["order_created_time"])  # KPI 3
    df["warehouse_wait_hours"] = hours(df["dispatch_time"] - df["order_created_time"])

    df["transit_distance_km"] = calculate_haversine_distance(
        df["origin_lat"], df["origin_lon"], df["dest_lat"], df["dest_lon"]
    )
    df["avg_speed_kmph"] = df["transit_distance_km"] / df["actual_transit_hours"]

    # Anomaly rules from Phase 1 of the roadmap
    valid = (df["actual_transit_hours"] > 0) & (df["avg_speed_kmph"] <= 150)
    dropped = (~valid).sum()
    if dropped:
        print(f"[cleaning] removed {dropped} anomalous record(s)")
    df = df[valid].copy()

    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["is_peak_hour"] = df["dispatch_hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    df["is_on_time"] = df["actual_delivery_time"] <= df["promised_delivery_time"]
    return {
        "shipments": len(df),
        "otd_pct": df["is_on_time"].mean() * 100,
        "cpd": df["operational_cost"].sum() / len(df),
        "mean_ofct_hours": df["ofct_hours"].mean(),
    }


def illustrate_ml_steps(df: pd.DataFrame) -> None:
    """Sketches of Phase 3. Five rows is far too few to train on; this only shows the API."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    # Clustering: group destinations for consolidated dispatch
    coords = df[["dest_lat", "dest_lon"]]
    df["dest_cluster"] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(coords)

    # Regression: predict actual transit time
    features = ["transit_distance_km", "dispatch_hour", "is_peak_hour"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["actual_transit_hours"], test_size=0.4, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"Baseline RF  RMSE: {rmse:.2f} h | MAE: {mean_absolute_error(y_test, pred):.2f} h")


def run_pipeline() -> pd.DataFrame:
    df = clean_and_engineer(load_data())
    k = compute_kpis(df)

    print("================== LOGISTICS KPI SUMMARY ==================")
    print(f"Total Shipments Analyzed      : {k['shipments']}")
    print(f"On-Time Delivery Rate (OTD)   : {k['otd_pct']:.2f}%")
    print(f"Average Cost Per Delivery     : ₹{k['cpd']:.2f}")
    print(f"Mean Order Fulfillment Cycle  : {k['mean_ofct_hours']:.2f} Hours")
    print("===========================================================")

    illustrate_ml_steps(df)
    return df


if __name__ == "__main__":
    processed_df = run_pipeline()