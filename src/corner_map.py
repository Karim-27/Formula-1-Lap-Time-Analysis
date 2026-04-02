from typing import Dict, Tuple

import pandas
import pandas as pd

def find_braking_points(df: pandas.DataFrame, min_zone_length = 2):
    '''
    Finds braking zones in the dataframe.
    '''
    if 'Brake' not in df.columns:
        return []

    braking = df['Brake'].astype(int)
    braking_diff = braking.diff().fillna(0)

    start_indices = braking_diff[braking_diff == 1].index
    end_indices = braking_diff[braking_diff == -1].index
    braking_zones = []

    for start in start_indices:
        end_candidates = end_indices[end_indices > start]
        if not end_candidates.empty:
            end = end_candidates[0]

            if end - start >= min_zone_length:
                braking_zones.append((int(start), int(end)))

    return braking_zones

def slice_corner(df: pd.DataFrame, braking_zones, zone) -> pd.DataFrame:
    if zone >= len(braking_zones):
        print(f"Braking zone {zone} is out of range")
        return df

    start, end = braking_zones[zone]

    start = max(start - 1, 0)
    end = min(end +1, len(df) - 1)

    return df.iloc[start:end+1]

