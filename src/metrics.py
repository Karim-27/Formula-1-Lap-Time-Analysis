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
        threshold_high: float = 50.0,
        threshold_low: float = 20.0,
        min_stay: int = 2,
        smooth_window: int = 3,
        debug: bool = False
) -> float:
    """
    FIXED: Detect first throttle pickup AFTER brake-off using more realistic thresholds.
    Returns distance where throttle application begins, or np.nan if not found.
    """
    # Edge case: empty DataFrame or missing columns
    if corner_df.empty or 'Throttle' not in corner_df.columns or 'Brake' not in corner_df.columns:
        if debug:
            print("[TP DEBUG] Empty DataFrame or missing Throttle/Brake columns")
        return np.nan

    braking_data = corner_df['Brake'].astype(bool).astype(int)
    brake_diff = braking_data.diff().fillna(0)

    start_idxs = brake_diff[brake_diff == 1].index
    end_idxs = brake_diff[brake_diff == -1].index

    # Debug info
    # if debug:
    #     thr_data = corner_df['Throttle'].astype(float)
    #     if thr_data.max() <= 1.0:
    #         thr_display = thr_data * 100.0
    #     else:
    #         thr_display = thr_data
    #
    #     print(f"[TP DEBUG] Zone analysis:")
    #     print(f"  - Total rows: {len(corner_df)}")
    #     print(f"  - Brake starts: {len(start_idxs)}, Brake ends: {len(end_idxs)}")
    #     print(f"  - Throttle range: {thr_display.min():.1f}% - {thr_display.max():.1f}%")

    if start_idxs.empty or end_idxs.empty:
        if debug:
            print("[TP DEBUG] No braking detected - finding throttle pickup from minimum throttle point")

        thr = corner_df['Throttle'].astype(float).copy()
        if thr.max() <= 1.0:
            thr = thr * 100.0

        min_throttle_idx = thr.idxmin()
        post = corner_df.loc[min_throttle_idx:].copy()

        if post.empty or len(post) < 2:
            if debug:
                print("[TP DEBUG] No data after minimum throttle point → NaN")
            return np.nan
    else:
        # FIXED: Better brake cycle handling
        valid_end = end_idxs[end_idxs > start_idxs[0]]

        if valid_end.empty:
            # CHANGED: Handle braking that continues to segment end
            if braking_data.iloc[-1] == 1:  # Still braking at end
                last_20_percent = int(len(corner_df) * 0.2)
                start_search_idx = max(start_idxs[0], len(corner_df) - last_20_percent)
                post = corner_df.loc[start_search_idx:].copy()
                if debug:
                    print(f"[TP DEBUG] Braking continues to end - searching last {len(post)} points")
            else:
                if debug:
                    print("[TP DEBUG] No brake-off after first brake-on → NaN")
                return np.nan
        else:
            end_idx = valid_end[0]
            post = corner_df.loc[end_idx:].copy()

        if post.empty or len(post) < 2:
            if debug:
                print("[TP DEBUG] Search window too small → NaN")
            return np.nan

    # Normalize throttle to 0-100%
    thr = post['Throttle'].astype(float).copy()
    if thr.max() <= 1.0:
        thr = thr * 100.0

    # Apply smoothing
    if smooth_window > 1 and len(thr) >= smooth_window:
        thr_s = thr.rolling(window=smooth_window, min_periods=1, center=False).median()
    else:
        thr_s = thr

    if debug:
        print(f"  - Search window throttle range: {thr_s.min():.1f}% - {thr_s.max():.1f}%")
        print(f"  - Thresholds: high={threshold_high}%, low={threshold_low}%")

    # CHANGED: Multiple detection methods
    idxs = thr_s.index.to_list()
    pick_idx = None

    # Method 1: Original hysteresis detection
    for i, idx in enumerate(idxs):
        if thr_s.iloc[i] >= threshold_high:
            start_i = max(0, i - (min_stay - 1))
            window_vals = thr_s.iloc[start_i:i + 1]

            if (window_vals >= threshold_low).all():
                pick_idx = idx
                if debug:
                    print(
                        f"[TP DEBUG] Found pickup (method 1) at distance={post.loc[idx, 'Distance']:.2f}, throttle={thr.loc[idx]:.1f}%")
                break

    # ADDED: Fallback methods
    if pick_idx is None:
        # Method 2: Find 30% increase from minimum
        min_val = thr_s.min()
        increase_threshold = min_val + (thr_s.max() - min_val) * 0.3

        candidates = thr_s[thr_s >= increase_threshold]
        if not candidates.empty:
            pick_idx = candidates.index[0]
            if debug:
                print(
                    f"[TP DEBUG] Found pickup (method 2) at distance={post.loc[pick_idx, 'Distance']:.2f}, throttle={thr.loc[pick_idx]:.1f}%")

        # Method 3: First point above low threshold
        elif not thr_s[thr_s >= threshold_low].empty:
            pick_idx = thr_s[thr_s >= threshold_low].index[0]
            if debug:
                print(
                    f"[TP DEBUG] Found pickup (method 3) at distance={post.loc[pick_idx, 'Distance']:.2f}, throttle={thr.loc[pick_idx]:.1f}%")

        # Method 4: Derivative-based detection
        elif len(thr_s) > 3:
            thr_diff = thr_s.diff().fillna(0)
            for i in range(1, len(thr_diff) - 1):
                if thr_diff.iloc[i] > 2.0 and thr_diff.iloc[i + 1] >= 0:
                    pick_idx = thr_diff.index[i]
                    if debug:
                        print(
                            f"[TP DEBUG] Found pickup (method 4) at distance={post.loc[pick_idx, 'Distance']:.2f}, throttle={thr.loc[pick_idx]:.1f}%")
                    break

        if pick_idx is None:
            if debug:
                print(f"[TP DEBUG] No throttle pickup found → NaN")
            return np.nan

    return float(post.loc[pick_idx, 'Distance'])





