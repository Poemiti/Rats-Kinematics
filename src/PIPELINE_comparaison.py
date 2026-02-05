#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from utils.trajectory_metrics import Trajectory

names = ["average velocity", "peak velocity", "tortuosity"]
METRIC =  names[2]
THRESHOLD = 0.5
BODYPART = 'left_hand'
SINGLE_TRAJ = False
SHOW = True
RAT_NAME = "#525"

# plot titles 
title = f"{METRIC} distribution, for {RAT_NAME} CHR rat population\n" \
        "bodypart observed : left hand, L1 task"
ylabel = f"{METRIC}" # (cm.s$^{-1}$)"
if "velocity" in METRIC : 
     ylabel += " (cm.s$^{-1}$)"

saving_name = f"Violin_{METRIC}_DETAIL_CHR_L1.png"


# define cm per pixel
frame_width_m = 0.12 # m
frame_width_px = 512 # pixel
M_PER_PIXEL = frame_width_m / frame_width_px

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
ANALYSIS_DIR = GENERATED_DATA_DIR / "analysis_results" / RAT_NAME
OUTPUT_DIR = GENERATED_DATA_DIR / "analysis_results" / RAT_NAME / "comparative_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------ velocity comparaison ---------------------------------------

# =================================================================== CHR CONDITION

#default
beta_OFF_1_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Beta_RightHemi_CueL1_H001_LaserOff/metrics_per_clips.csv"
conti_OFF_1_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Conti_RightHemi_CueL1_H001_LaserOff/metrics_per_clips.csv"

beta_ON_1_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Beta_RightHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"
conti_ON_1_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Conti_RightHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"

# greater
beta_OFF_2_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Beta_RightHemi_CueL1_H001_LaserOff_2,5mW/metrics_per_clips.csv"
conti_OFF_2_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Conti_RightHemi_CueL1_H001_LaserOff_0,75mW/metrics_per_clips.csv"

beta_ON_2_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Beta_RightHemi_CueL1_H001_LaserOn_2,5mW/metrics_per_clips.csv"
conti_ON_2_csv = f"../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CHR_Conti_RightHemi_CueL1_H001_LaserOn_0,75mW/metrics_per_clips.csv"



# default
beta_OFF_1_velo = pd.read_csv(beta_OFF_1_csv)[METRIC]
conti_OFF_1_velo = pd.read_csv(conti_OFF_1_csv)[METRIC]

beta_ON_1_velo = pd.read_csv(beta_ON_1_csv)[METRIC]
conti_ON_1_velo = pd.read_csv(conti_ON_1_csv)[METRIC]

# greater
beta_OFF_2_velo = pd.read_csv(beta_OFF_2_csv)[METRIC]
conti_OFF_2_velo = pd.read_csv(conti_OFF_2_csv)[METRIC]

beta_ON_2_velo = pd.read_csv(beta_ON_2_csv)[METRIC]
conti_ON_2_velo = pd.read_csv(conti_ON_2_csv)[METRIC]

for data in [beta_OFF_1_velo, beta_OFF_2_velo, conti_OFF_1_velo, conti_OFF_2_velo] : 
     print(len(data))

def _pack(values, condition, intensity):
        return pd.DataFrame({
            "Condition": condition,
            "LaserIntensity": intensity,
            "Value": values.dropna().values
        })

df_long = pd.concat([
    _pack(beta_OFF_1_velo, "Beta-NoStim", "conti=0.5, beta=1"),
    _pack(conti_OFF_1_velo, "Conti-NoStim", "conti=0.5, beta=1"),
    _pack(beta_ON_1_velo, "Beta", "conti=0.5, beta=1"),
    _pack(conti_ON_1_velo, "Conti", "conti=0.5, beta=1"),

    _pack(beta_OFF_2_velo, "Beta-NoStim", "conti=0.75, beta=2.5"),
    _pack(conti_OFF_2_velo, "Conti-NoStim", "conti=0.75, beta=2.5"),
    _pack(beta_ON_2_velo, "Beta", "conti=0.75, beta=2.5"),
    _pack(conti_ON_2_velo, "Conti", "conti=0.75, beta=2.5"),
])

# df_long = pd.concat([
#     _pack(beta_OFF_1_velo, "NoStim", "conti=0.5, beta=1"),
#     _pack(conti_OFF_1_velo, "NoStim", "conti=0.5, beta=1"),
#     _pack(beta_ON_1_velo, "Beta", "conti=0.5, beta=1"),
#     _pack(conti_ON_1_velo, "Conti", "conti=0.5, beta=1"),

#     _pack(beta_OFF_2_velo, "NoStim", "conti=0.75, beta=2.5"),
#     _pack(conti_OFF_2_velo, "NoStim", "conti=0.75, beta=2.5"),
#     _pack(beta_ON_2_velo, "Beta", "conti=0.75, beta=2.5"),
#     _pack(conti_ON_2_velo, "Conti", "conti=0.75, beta=2.5"),
# ])

# remove extreme values with the interquarile range method (IQR)
def trim_extremes_iqr(df, value_col="Value", group_cols=("Condition", "LaserIntensity"), k=1.5):
    def _trim(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        return group[(group[value_col] >= low) & (group[value_col] <= high)]

    return (
        df
        .groupby(list(group_cols), group_keys=False)
        .apply(_trim)
        .reset_index(drop=True)
    )

df_long_trimed = trim_extremes_iqr(df_long, k=1.5)
print(f"\nNumber of removed outliers : {len(df_long) - len(df_long_trimed)}")
print(df_long_trimed)

# plotting
fig, ax = plt.subplots()


sns.violinplot(
    x="Condition",
    y="Value",
    hue="LaserIntensity",
    data=df_long_trimed,
    ax=ax,
    split=True,
    gap= .1,
    inner="quart",
    order=["Conti-NoStim", "Beta-NoStim", "Conti", "Beta"],
    palette={"conti=0.5, beta=1": "lightblue", "conti=0.75, beta=2.5": "salmon"},
)

# sns.violinplot(
#     x="Condition",
#     y="Value",
#     hue="LaserIntensity",
#     data=df_long_trimed,
#     ax=ax,
#     split=True,
#     gap= .1,
#     inner="quart",
#     order=["NoStim", "Conti", "Beta"],
#     palette={"conti=0.5, beta=1": "lightblue", "conti=0.75, beta=2.5": "salmon"},
# )

sns.stripplot(
    x="Condition",
    y="Value",
    hue="LaserIntensity",
    data=df_long_trimed,
    ax=ax,
    dodge=True,
    color="black",
    size=2.5,
    alpha=0.5,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Laser intensity", 
          loc='lower right', fontsize=7)

# ---- add counts ----
counts = df_long_trimed.groupby(["Condition", "LaserIntensity"]).size().reset_index(name="N")
y_max = df_long_trimed["Value"].max()
y_offset = 0.05 * y_max

for _, row in counts.iterrows():
    x = ["Conti-NoStim", "Beta-NoStim", "Conti", "Beta"].index(row["Condition"])
    # x = ["NoStim", "Conti", "Beta"].index(row["Condition"])
    x += -0.15 if row["LaserIntensity"] == "conti=0.5, beta=1" else 0.20

    ax.text(
        x,
        y_max + y_offset,
        f"{row['N']}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color = "lightblue" if row["LaserIntensity"] == "conti=0.5, beta=1" else "salmon"
    )

ax.set_ylabel(ylabel)
ax.set_title(title)
ax.grid(axis="y", alpha=0.3)

plt.savefig(OUTPUT_DIR / saving_name)

if SHOW : 
    plt.show()

plt.close(fig)



# =================================================================== CTRL CONDITION


# noStim_csv = "../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CTRL_Conti_LeftHemi_CueL1_H001_LaserOff_2,5mW/metrics_per_clips.csv"
# beta_ON_1_csv = "../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CTRL_Beta_LeftHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"
# conti_ON_1_csv = "../data/analysis_results/{RAT_NAME}/{RAT_NAME}_CTRL_Conti_LeftHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"

# noStim_velo = pd.read_csv(noStim_csv)[METRIC]
# beta1_velo = pd.read_csv(beta_ON_1_csv)[METRIC]
# conti1_velo = pd.read_csv(conti_ON_1_csv)[METRIC]


# fig, ax = plt.subplots()
# title = "Velocity distribution, for CTRL rat population\n" \
#         "bodypart observed : left hand, L1 task"

# ylabel = "velocity (m.$s^{-1}$)"

# def build_long_df(noStim, conti1, beta1):
#     def _pack(values, condition, intensity):
#         return pd.DataFrame({
#             "Condition": condition,
#             "LaserIntensity": intensity,
#             "Value": values.dropna().values
#         })

#     df = pd.concat([
#         _pack(noStim, "NoStim", "conti=0.5, beta=1"),
#         _pack(conti1, "Conti", "conti=0.5, beta=1"),
#         _pack(beta1, "Beta", "conti=0.5, beta=1"),
#     ], ignore_index=True)

#     return df

# df_long = build_long_df(
#     noStim_velo,
#     conti1_velo,
#     beta1_velo,
# )


# sns.violinplot(
#     x="Condition",
#     y="Value",
#     hue="LaserIntensity",
#     data=df_long,
#     ax=ax,
#     split=True,
#     gap= .1,
#     inner="quart",
#     palette={"conti=0.5, beta=1": "lightblue"},
# )

# sns.stripplot(
#     x="Condition",
#     y="Value",
#     hue="LaserIntensity",
#     data=df_long,
#     ax=ax,
#     dodge=True,
#     color="black",
#     size=2.5,
#     alpha=0.5,
# )

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], title="Laser intensity", 
#           loc='lower right', fontsize=7)

# # ---- add counts ----
# counts = df_long.groupby(["Condition", "LaserIntensity"]).size().reset_index(name="N")
# y_max = df_long["Value"].max()
# y_offset = 0.05 * y_max

# for _, row in counts.iterrows():
#     x = ["NoStim", "Conti", "Beta"].index(row["Condition"])
#     x += -0.15 if row["LaserIntensity"] == "conti=0.5, beta=1" else 0.15

#     ax.text(
#         x,
#         y_max + y_offset,
#         f"n={row['N']}",
#         ha="center",
#         va="bottom",
#         fontsize=9,
#         fontweight="bold",
#         color = "lightblue" if row["LaserIntensity"] == "conti=0.5, beta=1" else "salmon"
#     )

# ax.set_ylabel(ylabel)
# ax.set_title(title)
# ax.grid(axis="y", alpha=0.3)

# plt.savefig(OUTPUT_DIR / "Violin_CTRL_L1.png")

# if SHOW : 
#     plt.show()

# plt.close(fig)

