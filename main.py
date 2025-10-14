from src.data_loader import load_session, get_fastest_lap
from src.diagnostics import run_diagnostics
from src.metrics import *
from src.corner_map import slice_corner, find_braking_points
import matplotlib.pyplot as plt
import src.diagnostics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
session = load_session(2024, "Saudi Arabia", "R")


# norris_df = get_fastest_lap(session, "NOR")
# ricciardo_df = get_fastest_lap(session, "RIC")
# nor_braking_zones = find_braking_points(norris_df)
# ric_braking_zones = find_braking_points(ricciardo_df)
#
# diagnostic = run_diagnostics(norris_df, ricciardo_df, nor_braking_zones, ric_braking_zones)
# print(diagnostic)

driver1_name_code = "PIA"
driver2_name_code = "VER"
d1_df = get_fastest_lap(session, driver1_name_code)
d2_df = get_fastest_lap(session, driver2_name_code)
d1_braking_zones = find_braking_points(d1_df)
d2_braking_zones = find_braking_points(d2_df)

driver1_name = d1_df["Driver"].iloc[0] if "Driver" in d1_df.columns else driver1_name_code
driver2_name = d1_df["Driver"].iloc[0] if "Driver" in d1_df.columns else driver2_name_code

diagnostic = run_diagnostics(d1_df, d2_df, d1_braking_zones, d2_braking_zones)
print(diagnostic)

plt.figure(figsize = (12, 6))

# plt.plot(norris_df['Distance'], norris_df['Throttle'], label = "Norris Throttle")
# plt.plot(ricciardo_df['Distance'], ricciardo_df['Throttle'], label = "Ricciardo Throtle")
# plt.plot(norris_df['Distance'], norris_df['Brake'].astype(int) * 300, label = 'Norris Brake (scaled)')
# plt.plot(ricciardo_df['Distance'], ricciardo_df['Brake'].astype(int) * 300, label = 'Ricciardo Brake (scaled)', linestyle = '--')
# plt.plot(norris_df['Distance'], norris_df['Throttle'], label = "Norris Throttle")

plt.plot(d1_df['Distance'], d1_df['Throttle'], label = f"{driver1_name} Throttle")
plt.plot(d2_df['Distance'], d2_df['Throttle'], label = f"{driver2_name} Throttle")
plt.plot(d1_df['Distance'], d1_df['Brake'].astype(int) * 300, label = f'{driver1_name} Brake (scaled)')
plt.plot(d2_df['Distance'], d2_df['Brake'].astype(int) * 300, label = f'{driver2_name} Brake (scaled)', linestyle = '--')

plt.plot(d1_df['Brake'].astype(int).values)
plt.title("Braking vs Index")
plt.xlabel("Index")
plt.ylabel("Braking")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
