import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_delta(d1_df: pd.DataFrame, d2_df: pd.DataFrame, d1_name: str = "Driver 1", d2_name: str = "Driver 2"):
    """
    Plot speed delta (d1 - d2) across the lap distance.
    Positive values mean d1 is faster at that point; negative means d2 is faster.
    """
    d1_dist = d1_df['Distance'].values
    d1_speed = d1_df['Speed'].values
    d2_dist = d2_df['Distance'].values
    d2_speed = d2_df['Speed'].values

    # Interpolate d2 speed onto d1's distance axis so we can subtract
    d2_interp = np.interp(d1_dist, d2_dist, d2_speed)
    delta = d1_speed - d2_interp

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(d1_dist, delta, color='purple', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(d1_dist, delta, 0, where=(delta > 0), alpha=0.3, color='red', label=f'{d1_name} faster')
    ax.fill_between(d1_dist, delta, 0, where=(delta < 0), alpha=0.3, color='blue', label=f'{d2_name} faster')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed Delta (km/h)')
    ax.set_title(f'Speed Delta — {d1_name} minus {d2_name}')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_throttle_brake(d1_df: pd.DataFrame, d2_df: pd.DataFrame, d1_name: str = "Driver 1", d2_name: str = "Driver 2"):
    """
    Two-panel subplot: throttle overlay (top) and brake overlay (bottom) vs distance.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Normalise throttle to 0-100% if it comes in as 0-1
    d1_thr = d1_df['Throttle'].astype(float)
    d2_thr = d2_df['Throttle'].astype(float)
    if d1_thr.max() <= 1.0:
        d1_thr = d1_thr * 100.0
    if d2_thr.max() <= 1.0:
        d2_thr = d2_thr * 100.0

    ax1.plot(d1_df['Distance'], d1_thr, label=f'{d1_name}', linewidth=1.2)
    ax1.plot(d2_df['Distance'], d2_thr, label=f'{d2_name}', linewidth=1.2, alpha=0.8)
    ax1.set_ylabel('Throttle (%)')
    ax1.set_title(f'Throttle & Brake Comparison — {d1_name} vs {d2_name}')
    ax1.set_ylim(-5, 110)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    d1_brake = d1_df['Brake'].astype(int)
    d2_brake = d2_df['Brake'].astype(int)
    ax2.fill_between(d1_df['Distance'], d1_brake, alpha=0.5, label=f'{d1_name}', step='post')
    ax2.fill_between(d2_df['Distance'], d2_brake * -1, alpha=0.5, label=f'{d2_name} (inverted)', step='post')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Brake (1 = on)')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels([d2_name, 'off', d1_name])
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_corner_speed(d1_df: pd.DataFrame, d2_df: pd.DataFrame, d1_name: str = "Driver 1", d2_name: str = "Driver 2"):
    """
    Speed trace overlay vs distance for both drivers, with braking zones shaded.
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(d1_df['Distance'], d1_df['Speed'], label=d1_name, linewidth=1.5)
    ax.plot(d2_df['Distance'], d2_df['Speed'], label=d2_name, linewidth=1.5, alpha=0.8)

    # Shade regions where d1 is braking
    d1_brake = d1_df['Brake'].astype(bool)
    in_zone = False
    zone_start = None
    for i, (dist, braking) in enumerate(zip(d1_df['Distance'], d1_brake)):
        if braking and not in_zone:
            zone_start = dist
            in_zone = True
        elif not braking and in_zone:
            ax.axvspan(zone_start, dist, alpha=0.08, color='gray')
            in_zone = False
    if in_zone:
        ax.axvspan(zone_start, d1_df['Distance'].iloc[-1], alpha=0.08, color='gray')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title(f'Speed Trace — {d1_name} vs {d2_name} (shaded = braking zones)')
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
