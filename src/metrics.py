import numpy as np
import pandas as pd


def get_braking_point(corner_df: pd.DataFrame):
    '''
    Returns the distance where the driver begins braking
    '''
    braking_data = corner_df[corner_df['Brake']]
    if not braking_data.empty:
        return braking_data['Distance'].iloc[0]
    return None

def get_braking_distance(corner_df: pd.DataFrame):
    '''
    Measures the distance where the driver was braking
    '''
    braking_data = corner_df['Brake'].astype(bool).astype(int)
    brake_diff = braking_data.diff().fillna(0)

    start_point = brake_diff[brake_diff == 1].index
    end_point = brake_diff[brake_diff == -1].index

    if start_point.empty or end_point.empty:
        return 0

    valid_end_point = end_point[end_point > start_point[0]]
    if valid_end_point.empty:
        return 0


    start = corner_df.loc[start_point[0], 'Distance']

    end = corner_df.loc[valid_end_point[0], 'Distance']

    return max(0, end - start)

def get_min_corner_speed(corner_df: pd.DataFrame):
    '''
    Returns the slowest speed of the driver through the corner
    '''
    if not corner_df.empty:
        return corner_df['Speed'].min()
    return None


def get_throttle_pickup(
        corner_df: pd.DataFrame,
        threshold_high: float = 80.0,
        threshold_low: float = 60.0,
        min_stay: int = 2,
        smooth_window: int = 3,
        debug: bool = False
) -> float:
    """
    Detect first throttle pickup AFTER brake-off using hysteresis.
    Returns distance where throttle application begins, or np.nan if not found.
    """

    # Edge case: empty DataFrame or missing columns
    if corner_df.empty or 'Throttle' not in corner_df.columns or 'Brake' not in corner_df.columns:
        if debug:
            print("[TP DEBUG] Empty DataFrame or missing Throttle/Brake columns")
        return np.nan

    # Detect brake edges
    braking_data = corner_df['Brake'].astype(bool).astype(int)
    brake_diff = braking_data.diff().fillna(0)

    start_idxs = brake_diff[brake_diff == 1].index
    end_idxs = brake_diff[brake_diff == -1].index

    # Debug info
    if debug:
        thr_data = corner_df['Throttle'].astype(float)
        # Normalize if needed for display
        if thr_data.max() <= 1.0:
            thr_display = thr_data * 100.0
        else:
            thr_display = thr_data

        print(f"[TP DEBUG] Zone analysis:")
        print(f"  - Total rows: {len(corner_df)}")
        print(f"  - Brake starts: {len(start_idxs)}, Brake ends: {len(end_idxs)}")
        print(f"  - Throttle range: {thr_display.min():.1f}% - {thr_display.max():.1f}%")
        print(f"  - Distance range: {corner_df['Distance'].min():.2f} - {corner_df['Distance'].max():.2f}")

    # Need at least one complete brake cycle
    if start_idxs.empty or end_idxs.empty:
        if debug:
            print("[TP DEBUG] No complete brake on/off cycle → NaN")
        return np.nan

    # Find first brake-off after first brake-on
    valid_end = end_idxs[end_idxs > start_idxs[0]]
    if valid_end.empty:
        if debug:
            print("[TP DEBUG] No brake-off after first brake-on → NaN")
        return np.nan

    end_idx = valid_end[0]

    # CRITICAL FIX: Get post-brake window
    post = corner_df.loc[end_idx:].copy()

    if debug:
        print(f"  - Brake-off index: {end_idx}")
        print(f"  - Post-brake window: {len(post)} rows")

    if post.empty or len(post) < 2:
        if debug:
            print("[TP DEBUG] Post-brake window too small (need data after brake release) → NaN")
        return np.nan

    # Normalize throttle to 0-100%
    thr = post['Throttle'].astype(float).copy()
    if thr.max() <= 1.0:
        thr = thr * 100.0

    # Apply light smoothing to reduce noise
    if smooth_window > 1 and len(thr) >= smooth_window:
        thr_s = thr.rolling(window=smooth_window, min_periods=1, center=False).median()
    else:
        thr_s = thr

    if debug:
        print(f"  - Post-brake throttle range: {thr_s.min():.1f}% - {thr_s.max():.1f}%")
        print(f"  - Thresholds: high={threshold_high}%, low={threshold_low}%")

    # Hysteresis detection: find first point >= high with min_stay support
    idxs = thr_s.index.to_list()
    pick_idx = None

    for i, idx in enumerate(idxs):
        if thr_s.iloc[i] >= threshold_high:
            # Check if we have enough history
            start_i = max(0, i - (min_stay - 1))
            window_vals = thr_s.iloc[start_i:i + 1]

            if (window_vals >= threshold_low).all():
                pick_idx = idx
                if debug:
                    print(
                        f"[TP DEBUG] Found pickup at row {i}, distance={post.loc[idx, 'Distance']:.2f}, throttle={thr.loc[idx]:.1f}%")
                break

    # Fallback: find first point >= threshold_low
    if pick_idx is None:
        soft = thr_s[thr_s >= threshold_low]
        if not soft.empty:
            pick_idx = soft.index[0]
            if debug:
                print(
                    f"[TP DEBUG] Using fallback (≥{threshold_low}%) at distance={post.loc[pick_idx, 'Distance']:.2f}, throttle={thr.loc[pick_idx]:.1f}%")
        else:
            if debug:
                print(f"[TP DEBUG] No throttle pickup found (max={thr_s.max():.1f}%) → NaN")
            return np.nan

    return float(post.loc[pick_idx, 'Distance'])




