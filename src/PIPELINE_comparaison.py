#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from utils.trajectory_metrics import Trajectory


THRESHOLD = 0.5
BODYPART = 'left_hand'
SINGLE_TRAJ = False
SHOW = True

# define cm per pixel
frame_width_m = 0.12 # m
frame_width_px = 512 # pixel
M_PER_PIXEL = frame_width_m / frame_width_px

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
ANALYSIS_DIR = GENERATED_DATA_DIR / "analysis_results" / "#517"
OUTPUT_DIR = GENERATED_DATA_DIR / "analysis_results" / "comparative_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------ velocity comparaison ---------------------------------------

# =================================================================== CTRL CONDITION
#default
beta_OFF_1_csv = "../data/analysis_results/#517/#517_CHR_Beta_RightHemi_CueL1_H001_LaserOff/metrics_per_clips.csv"
conti_OFF_1_csv = "../data/analysis_results/#517/#517_CHR_Conti_RightHemi_CueL1_H001_LaserOff/metrics_per_clips.csv"

beta_ON_1_csv = "../data/analysis_results/#517/#517_CHR_Beta_RightHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"
conti_ON_1_csv = "../data/analysis_results/#517/#517_CHR_Conti_RightHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"

# greater
beta_OFF_2_csv = "../data/analysis_results/#517/#517_CHR_Beta_RightHemi_CueL1_H001_LaserOff_2,5mW/metrics_per_clips.csv"
conti_OFF_2_csv = "../data/analysis_results/#517/#517_CHR_Conti_RightHemi_CueL1_H001_LaserOff_0,75mW/metrics_per_clips.csv"

beta_ON_2_csv = "../data/analysis_results/#517/#517_CHR_Beta_RightHemi_CueL1_H001_LaserOn_2,5mW/metrics_per_clips.csv"
conti_ON_2_csv = "../data/analysis_results/#517/#517_CHR_Conti_RightHemi_CueL1_H001_LaserOn_0,75mW/metrics_per_clips.csv"

# default
beta_OFF_1_velo = pd.read_csv(beta_OFF_1_csv)['velocity']
conti_OFF_1_velo = pd.read_csv(conti_OFF_1_csv)['velocity']

beta_ON_1_velo = pd.read_csv(beta_ON_1_csv)['velocity']
conti_ON_1_velo = pd.read_csv(conti_ON_1_csv)['velocity']

# greater
beta_OFF_2_velo = pd.read_csv(beta_OFF_2_csv)['velocity']
conti_OFF_2_velo = pd.read_csv(conti_OFF_2_csv)['velocity']

beta_ON_2_velo = pd.read_csv(beta_ON_2_csv)['velocity']
conti_ON_2_velo = pd.read_csv(conti_ON_2_csv)['velocity']



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

# plotting
fig, ax = plt.subplots()
title = "Velocity distribution, for CHR rat population\n" \
        "bodypart observed : left hand, L1 task"

ylabel = "average velocity (m.$s^{-1}$)"

sns.violinplot(
    x="Condition",
    y="Value",
    hue="LaserIntensity",
    data=df_long,
    ax=ax,
    split=True,
    gap= .1,
    inner="quart",
    palette={"conti=0.5, beta=1": "lightblue", "conti=0.75, beta=2.5": "salmon"},
)

sns.stripplot(
    x="Condition",
    y="Value",
    hue="LaserIntensity",
    data=df_long,
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
counts = df_long.groupby(["Condition", "LaserIntensity"]).size().reset_index(name="N")
y_max = df_long["Value"].max()
y_offset = 0.05 * y_max

for _, row in counts.iterrows():
    x = ["Conti-NoStim", "Beta-NoStim", "Conti", "Beta"].index(row["Condition"])
    x += -0.15 if row["LaserIntensity"] == "conti=0.5, beta=1" else 0.20

    ax.text(
        x,
        y_max + y_offset,
        f"n={row['N']}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color = "lightblue" if row["LaserIntensity"] == "conti=0.5, beta=1" else "salmon"
    )

ax.set_ylabel(ylabel)
ax.set_title(title)
ax.grid(axis="y", alpha=0.3)

plt.savefig(OUTPUT_DIR / "Violin_mean_velocity_CHR_L1.png")

if SHOW : 
    plt.show()

plt.close(fig)


# =================================================================== CTRL CONDITION


# noStim_csv = "../data/analysis_results/#517/#517_CTRL_Conti_LeftHemi_CueL1_H001_LaserOff_2,5mW/metrics_per_clips.csv"
# beta_ON_1_csv = "../data/analysis_results/#517/#517_CTRL_Beta_LeftHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"
# conti_ON_1_csv = "../data/analysis_results/#517/#517_CTRL_Conti_LeftHemi_CueL1_H001_LaserOn/metrics_per_clips.csv"

# noStim_velo = pd.read_csv(noStim_csv)['velocity']
# beta1_velo = pd.read_csv(beta_ON_1_csv)['velocity']
# conti1_velo = pd.read_csv(conti_ON_1_csv)['velocity']


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

