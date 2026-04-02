import pandas as pd
from src.corner_map import slice_corner
from src.metrics import get_braking_point, get_braking_distance, get_min_corner_speed, get_throttle_pickup


def run_diagnostics(driver1_df: pd.DataFrame, driver2_df: pd.DataFrame, d1_braking_zones, d2_braking_zones):
    '''
    Compares two drivers across all matching corners.
    Returns a DataFrame with raw metrics and delta columns for each zone.
    Positive deltas mean Driver 1 has the higher value.
    '''
    results = []

    for i in range(min(len(d1_braking_zones), len(d2_braking_zones))):
        d1_corner = slice_corner(driver1_df, d1_braking_zones, i)
        d2_corner = slice_corner(driver2_df, d2_braking_zones, i)

        d1_bp = get_braking_point(d1_corner)
        d2_bp = get_braking_point(d2_corner)
        d1_bd = get_braking_distance(d1_corner)
        d2_bd = get_braking_distance(d2_corner)
        d1_ms = get_min_corner_speed(d1_corner)
        d2_ms = get_min_corner_speed(d2_corner)
        d1_tp = get_throttle_pickup(d1_corner)
        d2_tp = get_throttle_pickup(d2_corner)

        def safe_delta(a, b):
            if a is None or b is None:
                return None
            try:
                return float(a) - float(b)
            except (TypeError, ValueError):
                return None

        metrics = {
            "Zone": i,
            "D1 Brakepoint": d1_bp,
            "D2 Brakepoint": d2_bp,
            "Brakepoint Delta": safe_delta(d1_bp, d2_bp),
            "D1 Brake Distance": d1_bd,
            "D2 Brake Distance": d2_bd,
            "Brake Distance Delta": safe_delta(d1_bd, d2_bd),
            "D1 Min Speed": d1_ms,
            "D2 Min Speed": d2_ms,
            "Min Speed Delta": safe_delta(d1_ms, d2_ms),
            "D1 Throttle Pickup": d1_tp,
            "D2 Throttle Pickup": d2_tp,
            "Throttle Pickup Delta": safe_delta(d1_tp, d2_tp),
        }

        results.append(metrics)
    return pd.DataFrame(results)
