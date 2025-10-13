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
    CHANGED (docstring): Detect first pickup AFTER brake-off using hysteresis.
    We define pickup as the first sample >= threshold_high with the preceding
    (min_stay-1) samples >= threshold_low (after light smoothing).
    Returns np.nan if not found.
    """

    # UNCHANGED (edge detection setup)
    braking_data = corner_df['Brake'].astype(bool).astype(int)
    brake_diff   = braking_data.diff().fillna(0)

    # UNCHANGED (edges)
    start_idxs = brake_diff[brake_diff == 1].index
    end_idxs   = brake_diff[brake_diff == -1].index

    # ADDED (debug summary)
    if debug:
        tmin = float(corner_df['Throttle'].min()) if 'Throttle' in corner_df else float('nan')
        tmax = float(corner_df['Throttle'].max()) if 'Throttle' in corner_df else float('nan')
        print(f"[TP DEBUG] slice rows={len(corner_df)} starts={len(start_idxs)} ends={len(end_idxs)} "
              f"thr(min,max)=({tmin},{tmax})")

    # UNCHANGED (need on/off)
    if start_idxs.empty or end_idxs.empty:
        if debug:
            print("[TP DEBUG] no complete brake on/off in slice → np.nan")
        return np.nan

    # ADDED (pair the first off *after* first on)
    valid_end = end_idxs[end_idxs > start_idxs[0]]
    if valid_end.empty:
        if debug:
            print("[TP DEBUG] no brake-off AFTER first brake-on → np.nan")
        return np.nan
    end_idx = valid_end[0]

    # CHANGED (post-brake window only)
    post = corner_df.loc[end_idx:].copy()
    if post.empty or 'Throttle' not in post.columns:
        if debug:
            print("[TP DEBUG] post-brake window empty/missing Throttle → np.nan")
        return np.nan

    # ADDED (normalize to 0–100 if needed)
    thr = post['Throttle'].astype(float)
    if thr.max() <= 1.0:
        thr = thr * 100.0

    # ADDED (light smoothing to kill tiny spikes)
    if smooth_window > 1:
        thr_s = thr.rolling(window=smooth_window, min_periods=1).median()
    else:
        thr_s = thr

    # ADDED (hysteresis: first idx where thr_s >= high and last `min_stay` are >= low)
    # Example with your sequence: 0,20,35,49,64,76,98 → pickup at the 98 sample.
    idxs = thr_s.index.to_list()
    pick_idx = None
    for i, idx in enumerate(idxs):
        if thr_s.iloc[i] >= threshold_high:
            # require (min_stay-1) prior samples (including current span) ≥ threshold_low
            start_i = max(0, i - (min_stay - 1))
            window_ok = bool((thr_s.iloc[start_i:i+1] >= threshold_low).all())
            if window_ok:
                pick_idx = idx
                break

    if pick_idx is None:
        # ADDED (optional softer fallback: first >= threshold_low)
        soft = thr_s[thr_s >= threshold_low]
        if not soft.empty:
            pick_idx = soft.index[0]
            if debug:
                print(f"[TP DEBUG] no ≥{threshold_high}% with dwell; fallback to ≥{threshold_low}% at idx={pick_idx}")
        else:
            if debug:
                print(f"[TP DEBUG] no pickup found (max after = {float(thr_s.max())}) → np.nan")
            return np.nan

    # UNCHANGED (return distance at pickup)
    dist = float(post.loc[pick_idx, 'Distance'])
    if debug:
        val = float(thr.loc[pick_idx])
        print(f"[TP DEBUG] pickup@distance={dist} throttle={val} (high={threshold_high}, low={threshold_low}, stay={min_stay})")
    return dist

