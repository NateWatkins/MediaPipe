import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
CLIP = "rgb"   # folder prefix under output_data/
DATA_ROOT = "/Users/sehyr/Desktop/MediaPipe/output_data"
FPS = 7.5      # frames per second for velocity

# Pick any joints you want, just edit these lists:
POS_JOINTS = ["left_wrist", "right_wrist"]  # for the Position figure
VEL_JOINTS = ["left_index_tip"]  # for the Velocity figure

# One color family only (filtered vs unfiltered)
FILTERED_COLOR   = "#0072B2"  # deep blue (Okabe–Ito)
UNFILTERED_COLOR = "#A6CEE3"  # lighter companion blue

LINEWIDTH_FILTERED   = 1.8
LINEWIDTH_UNFILTERED = 2.2
ALPHA_FILTERED   = 1.0
ALPHA_UNFILTERED = 0.55

# ------------------ PATHS -------------------
raw_body  = os.path.join(DATA_ROOT, f"{CLIP}_data", "unfiltered", f"{CLIP}_body.csv")
filt_body = os.path.join(DATA_ROOT, f"{CLIP}_data", "filtered",   f"{CLIP}_body.csv")
raw_hand  = os.path.join(DATA_ROOT, f"{CLIP}_data", "unfiltered", f"{CLIP}_hand.csv")
filt_hand = os.path.join(DATA_ROOT, f"{CLIP}_data", "filtered",   f"{CLIP}_hand.csv")

# ------------------ LOAD --------------------
bd_raw  = pd.read_csv(raw_body)
bd_filt = pd.read_csv(filt_body)
hd_raw  = pd.read_csv(raw_hand)
hd_filt = pd.read_csv(filt_hand)

# Align lengths (assumes synchronous export)
n = min(len(bd_raw), len(bd_filt), len(hd_raw), len(hd_filt))
time = np.arange(n) / FPS
bd_raw  = bd_raw.iloc[:n].reset_index(drop=True)
bd_filt = bd_filt.iloc[:n].reset_index(drop=True)
hd_raw  = hd_raw.iloc[:n].reset_index(drop=True)
hd_filt = hd_filt.iloc[:n].reset_index(drop=True)

# ------------------ HELPERS -----------------
def _has_xy(df, joint):
    return f"{joint}_x" in df.columns and f"{joint}_y" in df.columns

def _xy(df, joint):
    x = df[f"{joint}_x"].astype(float).to_numpy()
    y = df[f"{joint}_y"].astype(float).to_numpy()
    return x, y

def pos_mag_any(joint, df_body, df_hand):
    """Return position magnitude for a joint from whichever DF contains it."""
    if _has_xy(df_body, joint):
        x, y = _xy(df_body, joint)
    elif _has_xy(df_hand, joint):
        x, y = _xy(df_hand, joint)
    else:
        return None
    return np.sqrt(x*x + y*y)

def vel_mag_any(joint, df_body, df_hand, fps):
    """Velocity magnitude using time-derivative of (x,y) from the DF that has the joint."""
    if _has_xy(df_body, joint):
        x, y = _xy(df_body, joint)
    elif _has_xy(df_hand, joint):
        x, y = _xy(df_hand, joint)
    else:
        return None
    dx = np.gradient(x) * fps
    dy = np.gradient(y) * fps
    return np.sqrt(dx*dx + dy*dy)

def _p98_ylim(*arrays):
    arrs = [a for a in arrays if a is not None and np.size(a) > 0]
    if not arrs:
        return None
    cat = np.concatenate(arrs)
    if cat.size == 0 or not np.isfinite(cat).any():
        return None
    ymax = np.nanpercentile(cat, 98) * 1.05
    return (0, ymax) if np.isfinite(ymax) and ymax > 0 else None

# ------------------ FIGURE 1: POSITION ------------------
pos_raw_list  = []
pos_filt_list = []
labels = []

for j in POS_JOINTS:
    raw_mag  = pos_mag_any(j, bd_raw,  hd_raw)
    filt_mag = pos_mag_any(j, bd_filt, hd_filt)
    if raw_mag is None or filt_mag is None:
        print(f"[warn] Joint '{j}' not found in body or hand CSVs; skipping.")
        continue
    pos_raw_list.append(raw_mag)
    pos_filt_list.append(filt_mag)
    labels.append(j)

ypos = _p98_ylim(*pos_filt_list)

plt.figure(figsize=(10, 4.5))
# unfiltered backdrop (same light color)
for y in pos_raw_list:
    plt.plot(time, y, color=UNFILTERED_COLOR, linewidth=LINEWIDTH_UNFILTERED, alpha=ALPHA_UNFILTERED, zorder=1)
# filtered foreground (single strong color)
for y, j in zip(pos_filt_list, labels):
    plt.plot(time, y, color=FILTERED_COLOR, linewidth=LINEWIDTH_FILTERED, alpha=ALPHA_FILTERED,
             label=f"{j} (filtered)", zorder=3)

if ypos: plt.ylim(*ypos)
plt.title("Position magnitude vs Time — Selected joints")
plt.xlabel("Time (s)")
plt.ylabel("Position (norm units)")
plt.grid(True, alpha=0.2)
plt.legend(loc="upper right", fontsize=9, frameon=False)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(os.path.join("plots", f"position_selected_{CLIP}.png"), dpi=220)
plt.savefig(os.path.join("plots", f"position_selected_{CLIP}.svg"))
plt.show()

# ------------------ FIGURE 2: VELOCITY ------------------
vel_raw_list  = []
vel_filt_list = []
labels = []

for j in VEL_JOINTS:
    raw_vel  = vel_mag_any(j, bd_raw,  hd_raw,  FPS)
    filt_vel = vel_mag_any(j, bd_filt, hd_filt, FPS)
    if raw_vel is None or filt_vel is None:
        print(f"[warn] Joint '{j}' not found in body or hand CSVs; skipping.")
        continue
    vel_raw_list.append(raw_vel)
    vel_filt_list.append(filt_vel)
    labels.append(j)

yvel = _p98_ylim(*vel_filt_list)

plt.figure(figsize=(10, 4.8))
for y in vel_raw_list:
    plt.plot(time, y, color=UNFILTERED_COLOR, linewidth=LINEWIDTH_UNFILTERED, alpha=ALPHA_UNFILTERED, zorder=1)
for y, j in zip(vel_filt_list, labels):
    plt.plot(time, y, color=FILTERED_COLOR, linewidth=LINEWIDTH_FILTERED, alpha=ALPHA_FILTERED,
             label=f"{j} (filtered)", zorder=3)

if yvel: plt.ylim(*yvel)
plt.title("Velocity magnitude vs Time — Selected joints")
plt.xlabel("Time (s)"); plt.ylabel("Speed (norm units/s)")
plt.grid(True, alpha=0.2)
plt.legend(loc="upper right", fontsize=9, ncol=1, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join("plots", f"velocity_selected_{CLIP}.png"), dpi=220)
plt.savefig(os.path.join("plots", f"velocity_selected_{CLIP}.svg"))
plt.show()
