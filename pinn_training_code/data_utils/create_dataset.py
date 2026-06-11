import json
import numpy as np
import pandas as pd

with open("6ms_mavros_imu_data.json", "r") as f:
    data = json.load(f)

rows = []
for timestamp, msg in data.items():
    row = {
        "timestamp": timestamp,
        "stamp_sec": msg["header"]["stamp"]["sec"],
        "stamp_nanosec": msg["header"]["stamp"]["nanosec"],
        "frame_id": msg["header"]["frame_id"],
        "orientation_x": msg["orientation"]["x"],
        "orientation_y": msg["orientation"]["y"],
        "orientation_z": msg["orientation"]["z"],
        "orientation_w": msg["orientation"]["w"],
        "angular_velocity_x": msg["angular_velocity"]["x"],
        "angular_velocity_y": msg["angular_velocity"]["y"],
        "angular_velocity_z": msg["angular_velocity"]["z"],
        "linear_acceleration_x": msg["linear_acceleration"]["x"],
        "linear_acceleration_y": msg["linear_acceleration"]["y"],
        "linear_acceleration_z": msg["linear_acceleration"]["z"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# Pitch and roll from accelerometer (static tilt estimation)
ax = df["linear_acceleration_x"]
ay = df["linear_acceleration_y"]
az = df["linear_acceleration_z"]

df["roll_rad"]  = np.arctan2(ay, az)
df["pitch_rad"] = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
df["roll_deg"]  = np.degrees(df["roll_rad"])
df["pitch_deg"] = np.degrees(df["pitch_rad"])

print(df.shape)
print(df[["roll_deg", "pitch_deg"]].head())

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(df.index, df["roll_deg"], color="tab:blue", linewidth=0.8)
axes[0].set_ylabel("Roll (deg)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df["pitch_deg"], color="tab:orange", linewidth=0.8)
axes[1].set_ylabel("Pitch (deg)")
axes[1].set_xlabel("Timestamp")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Roll & Pitch from Accelerometer")
plt.tight_layout()
plt.show()

# --- Force/Torque data from test_6ms.csv ---
ft_cols = ["Force X (N)", "Force Y (N)", "Force Z (N)",
           "Torque X (N-m)", "Torque Y (N-m)", "Torque Z (N-m)"]
df_ft = pd.read_csv("test_6ms.csv", header=0, usecols=range(6), names=ft_cols)

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

for col in ["Force X (N)", "Force Y (N)", "Force Z (N)"]:
    axes2[0].plot(df_ft.index, df_ft[col], linewidth=0.8, label=col)
axes2[0].set_ylabel("Force (N)")
axes2[0].legend(fontsize=8)
axes2[0].grid(True, alpha=0.3)

for col in ["Torque X (N-m)", "Torque Y (N-m)", "Torque Z (N-m)"]:
    axes2[1].plot(df_ft.index, df_ft[col], linewidth=0.8, label=col)
axes2[1].set_ylabel("Torque (N-m)")
axes2[1].set_xlabel("Sample")
axes2[1].legend(fontsize=8)
axes2[1].grid(True, alpha=0.3)

fig2.suptitle("Force & Torque — test_6ms.csv")
plt.tight_layout()
plt.show()

# --- DTW synchronisation: Force Y  <->  Pitch ---
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

pitch  = df["pitch_deg"].values
force_y = df_ft["Force Y (N)"].values

# Normalise to zero-mean unit-variance so DTW distance is scale-independent
def znorm(x):
    return (x - x.mean()) / (x.std() + 1e-9)

pitch_n   = znorm(pitch)
force_y_n = znorm(force_y)

print("Running fastdtw …")
distance, path = fastdtw(pitch_n.reshape(-1, 1), force_y_n.reshape(-1, 1),
                         dist=euclidean)
print(f"DTW distance: {distance:.4f}")

path = np.array(path)          # shape (K, 2)
pitch_aligned   = pitch[path[:, 0]]
force_y_aligned = force_y[path[:, 1]]

# Plot
fig3, ax_left = plt.subplots(figsize=(13, 5))
ax_right = ax_left.twinx()

ax_left.plot(pitch_aligned,   color="tab:orange", linewidth=0.7, label="Pitch (deg)")
ax_right.plot(force_y_aligned, color="tab:blue",  linewidth=0.7, label="Force Y (N)", alpha=0.8)

ax_left.set_xlabel("DTW sample index")
ax_left.set_ylabel("Pitch (deg)",   color="tab:orange")
ax_right.set_ylabel("Force Y (N)",  color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:orange")
ax_right.tick_params(axis="y", labelcolor="tab:blue")

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

fig3.suptitle("DTW-Synchronised: Pitch vs Force Y")
plt.tight_layout()
plt.show()

# --- Build synchronised CSV dataset ---
# Use the DTW path to align pitch (IMU side) with all F/T columns
imu_idx = path[:, 0]
ft_idx  = path[:, 1]

# Assign IMU timestamps to every DTW-matched row, then interpolate both
# signals onto a uniform time grid (median IMU sample rate).
df_sync_raw = pd.DataFrame({
    "timestamp_imu":  df.index.values[imu_idx],
    "pitch_deg":      df["pitch_deg"].values[imu_idx],
    "roll_deg":       df["roll_deg"].values[imu_idx],
    "Force X (N)":    df_ft["Force X (N)"].values[ft_idx],
    "Force Y (N)":    df_ft["Force Y (N)"].values[ft_idx],
    "Force Z (N)":    df_ft["Force Z (N)"].values[ft_idx],
    "Torque X (N-m)": df_ft["Torque X (N-m)"].values[ft_idx],
    "Torque Y (N-m)": df_ft["Torque Y (N-m)"].values[ft_idx],
    "Torque Z (N-m)": df_ft["Torque Z (N-m)"].values[ft_idx],
})

df_sync_raw["timestamp_imu"] = pd.to_datetime(df_sync_raw["timestamp_imu"])
df_sync_raw = df_sync_raw.set_index("timestamp_imu").sort_index()

# Average duplicated timestamps that arise from the many-to-one DTW mapping
df_sync_raw = df_sync_raw.groupby(level=0).mean()

# Build a uniform time grid at the median IMU sampling interval
dt_median = pd.Series(df_sync_raw.index).diff().median()
t_start   = df_sync_raw.index[0]
t_end     = df_sync_raw.index[-1]
uniform_idx = pd.date_range(start=t_start, end=t_end, freq=dt_median)

# Reindex and interpolate linearly onto the uniform grid
df_sync = (
    df_sync_raw
    .reindex(df_sync_raw.index.union(uniform_idx))
    .interpolate(method="time")
    .reindex(uniform_idx)
)
df_sync.index.name = "timestamp"

out_path = "dataset_synced.csv"
df_sync.to_csv(out_path)
print(f"Saved synchronised dataset → {out_path}  shape={df_sync.shape}")
print(f"Uniform sample interval : {dt_median}")
print(df_sync.head())

# --- Train / Validation split (first 2/3 train, last 1/3 val) ---
split_idx = int(len(df_sync) * 2 / 3)
df_train = df_sync.iloc[:split_idx]
df_val   = df_sync.iloc[split_idx:]

df_train.to_csv("dataset_train.csv")
df_val.to_csv("dataset_val.csv")
print(f"Train set  → dataset_train.csv  shape={df_train.shape}")
print(f"Val   set  → dataset_val.csv    shape={df_val.shape}")

# --- Plot train and validation datasets ---
def plot_dataset(data, title):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(data.index, data["pitch_deg"], color="tab:orange", linewidth=0.8, label="Pitch (deg)")
    axes[0].plot(data.index, data["roll_deg"],  color="tab:green",  linewidth=0.8, label="Roll (deg)")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for col, c in zip(["Force X (N)", "Force Y (N)", "Force Z (N)"],
                      ["tab:blue", "tab:red", "tab:purple"]):
        axes[1].plot(data.index, data[col], linewidth=0.8, label=col, color=c)
    axes[1].set_ylabel("Force (N)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # for col, c in zip(["Torque X (N-m)", "Torque Y (N-m)", "Torque Z (N-m)"],
    #                   ["tab:blue", "tab:red", "tab:purple"]):
    #     axes[2].plot(data.index, data[col], linewidth=0.8, label=col, color=c)
    # axes[2].set_ylabel("Torque (N-m)")
    # axes[2].set_xlabel("Timestamp")
    # axes[2].legend(fontsize=8)
    # axes[2].grid(True, alpha=0.3)

    # fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_dataset(df_train, "Part of airfoil training dataset")
plot_dataset(df_val,   "Validation Dataset (last 1/3)")
