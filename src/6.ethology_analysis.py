#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
import time
import seaborn as sns
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path, load_trial_data
import rats_kinematics_utils.analysis.clustering as c
from rats_kinematics_utils.preprocessing.preprocess import crop_xy


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("paper", style="ticks", rc=custom_params, palette="pastel")

LASER_COLOR = "coral"
LINE_COLOR = "gray"
AVG_LINE_COLOR = "navy"
LINE_TRANSPARANCY = 0.3

REWARD_PALETTE = {"no": "black",
                  "yes": "green"}
LASER_INTENSITY_PALETTE = {"low" : "lightblue",
                            "high" : "salmon",
                            "NOstim" : "gray"}
LASER_INTENSITY_PALETTE_DARK = {"low" : "steelblue",
                            "high" : "tomato",
                            "NOstim" : "black"}
HUE_ORDER = ["low", "high"]


LASER_STATE_MARKER = {
    "LaserOff" :"X",
    "LaserOn" : "o"
}

LASER_STATE_PALETTE = {
    "LaserOff" :"slategray",
    "LaserOn" : "tomato"
}

dash = {
    "LaserOff" : (4,2),
    "LaserOn" : ""
}


behavior_palette = {
    "none": "k",
    "reach": "gold",
    "open": "orange",
    "grasp": "cornflowerblue",
    "press": "yellowgreen"
}

behavior_to_int = {
    "none": 0,
    "reach": 1,
    "open": 2,
    "grasp": 3,
    "press": 4
}
int_to_behavior = {v: k for k, v in behavior_to_int.items()}



condition_palette = {
    'Conti_LaserOff_low': "deepskyblue",
    'Conti_LaserOff_high': "salmon", 
    'Conti_LaserOn_high': "orangered", 
    'Beta_LaserOff_low': "steelblue", 
    'Beta_LaserOn_low': "springgreen", 
    'Beta_LaserOn_high': "deeppink", 
    'Beta_LaserOff_high': "orchid",
    'Conti_LaserOn_low': "limegreen", 
}

condition_to_int = {
    'Conti_LaserOff_low': 0,
    'Conti_LaserOff_high': 1, 
    'Conti_LaserOn_high': 2, 
    'Conti_LaserOn_low': 3, 
    'Beta_LaserOff_low': 4, 
    'Beta_LaserOn_low': 5, 
    'Beta_LaserOn_high': 6, 
    'Beta_LaserOff_high': 7,
}
int_to_condition = {v: k for k, v in condition_to_int.items()}




# ---------------------------- function ----------------------------------



def add_pad_off(ax, x_pad_off) -> None: 

    ax.axvline(
        x=x_pad_off,
        color="k",
        lw=1,
        ls="--",
        label="Pad off"
    )



def add_laser_period(ax, y, x_min, x_max, color: str, show_limits: bool=False) -> None: 

    ax.hlines(
        y=y,
        xmin=x_min ,
        xmax=x_max,
        color=color,
        linewidth=4,
        label="Laser period",
        clip_on=False
    )

    if show_limits: 
        ax.axvline(
            x=x_min,
            color=color,
            alpha=0.7,
            lw=0.5,
            ls="--",
        )
        ax.axvline(
            x=x_max,
            color=color,
            alpha=0.7,
            lw=0.5,
            ls="--",
        )


# ---------------------------- setup ----------------------------------

SHOW = False
NORMALIZE = False
cfg = load_config()
print_analysis_info(cfg, "Ethology Analysis")

filenames = list((cfg.paths.metrics).glob("*.joblib"))
filenames = sorted(filenames)
data = pd.DataFrame()
n = 0


bad_trials = pd.DataFrame()
behavior = pd.DataFrame()

condition_index = {}
overall_condition = ""


for i, metrics_path in enumerate(filenames) :
    metrics = load_trial_data(Path(metrics_path))
    bad_trials = 0

    print(f"\nLoading : {metrics_path.stem}")

    for j, trial in enumerate(metrics) : 

        if not trial[cfg.bodypart]["trial_success"] or trial["date"].month == 5: 
            bad_trials += 1
            continue
        
        condition = trial["condition"]
        laser_state = trial["laser_state"]

        if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
        elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
        else : laser_intensity = "high"

        overall_condition = f"{condition}_{laser_state}_{laser_intensity}"

        etho = trial[cfg.bodypart]["xy_etho"]
        x = etho["x"] * cfg.cm_per_pixel
        y = (512 - etho["y"]) * cfg.cm_per_pixel

        velo = trial[cfg.bodypart]["instant_velocity"]
        pad_off = trial["pad_off"]
        frame_pad_off = pad_off * cfg.fps

        # crop around the pad off
        velocity = crop_xy(velo, 
                        start=pad_off - 0.1, 
                        end=pad_off + 0.4)
        croped_etho = crop_xy(etho, 
                        start=pad_off - 0.1, 
                        end=pad_off + 0.4)
        relative_time = velocity["t"] - pad_off


        # behavior time line
        etho_labels = etho["label"].map(behavior_to_int).to_list()
        etho_condition = condition_to_int[overall_condition]

        df_beha = pd.DataFrame({
            "t" : etho["t"] - pad_off,
            "behavior": etho_labels,
            "id": n,
            "condition": etho_condition
        })



        behavior = pd.concat([behavior, df_beha], ignore_index=True)


        df = pd.DataFrame({
                # "x": x,
                # "y": y,
                # "t": etho["t"],
                # "behavior": etho["label"],

                "y_pos": (512 - croped_etho["y"]) * cfg.cm_per_pixel,
                "velocity": velocity["velocity"],
                "velocity_behavior": croped_etho["label"],
                "velocity_t": relative_time,

                "condition": condition,
                "laser_state": laser_state,
                "laser_intensity": laser_intensity,
                "id": n,
            })

        data = pd.concat([data, df], ignore_index=True)

        if j == 0+bad_trials : 
            condition_index[overall_condition] = {"min": n, "max": None}

        n += 1

    condition_index[overall_condition]["max"] = n-1


# -------------------- plot ----------------------------------


# g = sns.FacetGrid(
#         data=data,
#         col="condition",
#         row="laser_state",
#         margin_titles=True,
#         height=4,      # size of each facet
#         aspect=1.2     # width/height ratio
#     )


# g.map_dataframe(
#         sns.lineplot, 
#         x="x",
#         y="y", 
#         estimator=None, 
#         errorbar=None,
#         color="black",
#         alpha=0.1,
#         units="id",
#         sort=False,
#         lw=.5
# )


# g.map_dataframe(
#         sns.scatterplot, 
#         x="x",
#         y="y", 
#         hue="behavior",
#         palette=behavior_palette,
#         s=8
# )



# g.add_legend(title="Behavior")
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# g.set_axis_labels("x (cm)", "y (cm)")
# g.figure.subplots_adjust(top=0.88)
# g.figure.suptitle(f"Behavior analysis{cfg.rat_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')

# sns.move_legend(g,
#                     borderpad=1,
#                     loc='upper right', 
#                     facecolor="lightgray")

# for ax in g.axes.flat:
#     ax.set_ylim(0, cfg.frame_width_cm)
#     ax.set_xlim(0, cfg.frame_width_cm)


# g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"trajectories.png"))

# fig = g.figure
# fig.tight_layout()
# plt.show()




# # -------------------------- distplot of the proportion of time in box -------------------


# count = (
#     data.groupby(["condition", "laser_state", "behavior"])
#         .size()
#         .groupby(level=[0, 1])          # within each facet
#         .transform(lambda x: ((x / x.sum()) * 100).round(1))
#         .reset_index(name="proportion"))

# g = sns.FacetGrid(
#         data=count,
#         col="condition",
#         row="laser_state",
#         margin_titles=True,
#         height=4,      # size of each facet
#         aspect=1.2     # width/height ratio
# )

# g.map_dataframe(
#         sns.barplot,
#         x="behavior",
#         y="proportion",
#         hue="behavior",
#         palette=behavior_palette,
#         order=behavior_palette.keys(),
#         legend=True,
# )



# g.add_legend(title="Behavior")
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# g.figure.subplots_adjust(top=0.88)
# g.figure.suptitle(f"Behavior proportion of rat {cfg.rat_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')

# sns.move_legend(g,
#                     borderpad=1,
#                     loc='upper right', 
#                     facecolor="lightgray")

# for ax in g.axes.flat:
#     for i in range(5): 
#         ax.bar_label(ax.containers[i], fontsize=10)
#     ax.set_ylabel("proportion (%)")


# g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"time_spend_per_behavior.png"))


# fig = g.figure
# fig.tight_layout()
# plt.show()




# -------------------------- velocity and behavior -------------------


g = sns.FacetGrid(
        data=data,
        col="condition",
        row="laser_state",
        margin_titles=True,
        height=4,      # size of each facet
        aspect=1.2     # width/height ratio
)

g.map_dataframe(
        sns.lineplot, 
        x="velocity_t",
        y="velocity", 
        estimator=None, 
        errorbar=None,
        color="black",
        alpha=0.05,
        units="id",
        sort=False,
        lw=.8
)


g.map_dataframe(
        sns.scatterplot, 
        x="velocity_t",
        y="velocity", 
        hue="velocity_behavior",
        palette=behavior_palette,
        s=8
)

laser_color = "royalblue"

for row_i, laser_intensity in enumerate(g.row_names):
    for col_j, condition in enumerate(g.col_names):

        ax = g.axes[row_i, col_j]
        ax.set_ylim(-5, 80)

        # Vertical line at pad off
        ax.axvline(
            0,
            color="k",
            alpha=0.5,
            lw=0.8,
            ls="--",
        )


        # Laser period annotation
        y = ax.get_ylim()[1] * 0.95

        ax.hlines(
            y=y,
            xmin=0.025,
            xmax=0.325,
            color=laser_color,
            linewidth=3
        )

        ax.text(
            0.175,
            y,
            "Laser period",
            ha="center",
            va="bottom",
            color=laser_color,
            fontsize=10
        )

g.add_legend(title="Behavior")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.figure.subplots_adjust(top=0.88)
g.figure.suptitle(f"Behavioral analysis of the veloctiy during the laser stimulation of rat {cfg.rat_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')

sns.move_legend(g,
                    borderpad=1,
                    loc='upper right', 
                    facecolor="lightgray")

g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"instant_velocity.png"))


fig = g.figure
fig.tight_layout()
plt.show()




# -------------------------- velocity and behavior -------------------


g = sns.FacetGrid(
        data=data,
        col="condition",
        row="laser_state",
        margin_titles=True,
        height=4,      # size of each facet
        aspect=1.2     # width/height ratio
)

g.map_dataframe(
        sns.lineplot, 
        x="velocity_t",
        y="y_pos", 
        estimator=None, 
        errorbar=None,
        color="black",
        alpha=0.05,
        units="id",
        sort=False,
        lw=.8
)


g.map_dataframe(
        sns.scatterplot, 
        x="velocity_t",
        y="y_pos", 
        hue="velocity_behavior",
        palette=behavior_palette,
        s=8
)

laser_color = "royalblue"

for row_i, laser_intensity in enumerate(g.row_names):
    for col_j, condition in enumerate(g.col_names):

        ax = g.axes[row_i, col_j]

        # Vertical line at pad off
        ax.axvline(
            0,
            color="k",
            alpha=0.5,
            lw=0.8,
            ls="--",
        )


        # Laser period annotation
        y = ax.get_ylim()[1] * 0.95

        ax.hlines(
            y=y,
            xmin=0.025,
            xmax=0.325,
            color=laser_color,
            linewidth=3
        )

        ax.text(
            0.175,
            y,
            "Laser period",
            ha="center",
            va="bottom",
            color=laser_color,
            fontsize=10
        )

g.add_legend(title="Behavior")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.figure.subplots_adjust(top=0.88)
g.figure.suptitle(f"Behavioral analysis of the y position during the laser stimulation of rat {cfg.rat_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')
g.set_axis_labels("time (s)", "y (cm)")

sns.move_legend(g,
                    borderpad=1,
                    loc='upper right', 
                    facecolor="lightgray")

g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"position_y.png"))


fig = g.figure
fig.tight_layout()
plt.show()



# # -------------------------- ethograme -------------------

behavior = behavior.sort_values(["id", "condition"]).reset_index(drop=True)
behavior["t"] = behavior["t"].round(3)

mat_behavior = behavior.pivot(
    index="t",
    columns="id",
    values="behavior"
).sort_index()

int_behavior_palette = ["lightgray", "gold", "orange", "cornflowerblue", "yellowgreen", ]
                        
fig, ax = plt.subplots(figsize=(15, 8))

# Heatmap
sns.heatmap(
    mat_behavior.T,
    cmap=ListedColormap(int_behavior_palette),
    cbar=False,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    linewidths=0,
)

ax.set_title("Ethogram")
ax.set_xlabel("time (s)")
ax.set_ylabel("trial")


# Axis limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
right_pad = 55
ax.set_xlim(xmin, xmax + right_pad)

# set tick positions

time_values = mat_behavior.index.to_numpy()
target = np.array([-1, 0, 1, 2, 3])
tick_idx = np.where(np.isin(time_values, target))[0]

ax.set_xticks(tick_idx)
ax.set_xticklabels(np.round(time_values[tick_idx], 2))
ax.spines['bottom'].set_visible(True) 
ax.spines['bottom'].set_bounds(xmin, xmax) 

# show pad off and laser period
x0 = np.argmin(np.abs(time_values))
add_pad_off(ax, x0)

laser_start = np.argmin(np.abs(time_values - 0.025))
laser_end   = np.argmin(np.abs(time_values - 0.325))
add_laser_period(ax, -3, laser_start, laser_end, color="red", show_limits=True)


# Condition bars on right side
for condition, idx in condition_index.items():

    y0 = idx["min"]
    y1 = idx["max"] + 1
    ymid = (y0 + y1) / 2

    ax.vlines(
        x=xmax + 5,
        ymin=y0,
        ymax=y1,
        color=condition_palette[condition],
        linewidth=14,
        clip_on=False
    )

    ax.text(
        xmax + 12,
        ymid,
        condition,
        va="center",
        ha="left",
        fontsize=9,
        color=condition_palette[condition]
    )


# Final layout

ax.legend(loc="upper left")

plt.suptitle(f"Ethogram across frames for rat {cfg.rat_name}\nNumber of trials: {len(behavior.groupby('id'))}", y=0.98)
plt.tight_layout()

plt.savefig(make_output_path(cfg.paths.analysis / "behavior","timeline.png"))

plt.show()







# --------------------------------------- probability of changing behavior -------------------------


mat_behavior = mat_behavior.fillna(0)

arr = mat_behavior.to_numpy()
time_values = mat_behavior.index.to_numpy()

records = []

for state, label in int_to_behavior.items():

    if state == 0 :     # don't show the "none" behavior
        continue

    prob = (arr == state).mean(axis=1)

    tmp = pd.DataFrame({
        "t": time_values,
        "behavior": label,
        "probability": prob
    })

    records.append(tmp)

df_prob = pd.concat(records, ignore_index=True)

fig, ax = plt.subplots(figsize=(12,8))

sns.lineplot(
    data=df_prob,
    x="t",
    y="probability",
    hue="behavior",
    palette=behavior_palette,
    linewidth=1,
    ax=ax,
)
ax.set_ylim(-0.05, 1)
ax.set_ybound(-0.05, 1)

ax.set_xlim(-2, 3)
ax.set_xbound(-2, 3)

add_pad_off(ax, 0)
add_laser_period(ax, y=ax.get_ylim()[1], x_min=0.025, x_max=0.325, color="red", show_limits=True)

ax.legend()

plt.xlabel("time (s)")
plt.ylabel("Probability")
plt.title(f"Behavior probability over time of rat {cfg.rat_name}\nAll condition mixed - Number of trials: {len(behavior.groupby('id'))}")
plt.tight_layout()

plt.savefig(make_output_path(cfg.paths.analysis / "behavior", "behavior_probability_all_condition.png"))




for int_cond, subset in behavior.groupby("condition"): 
    condition = int_to_condition[int_cond]
    print("\n", condition)

    mat_subset = subset.pivot(
        index="t",
        columns="id",
        values="behavior"
    ).sort_index()


    arr = mat_subset.to_numpy()
    time_values = mat_subset.index.to_numpy()

    records = []

    for state, label in int_to_behavior.items():

        if state == 0 :     # don't show the "none" behavior
            continue

        prob = (arr == state).mean(axis=1)

        tmp = pd.DataFrame({
            "t": time_values,
            "behavior": label,
            "probability": prob
        })

        records.append(tmp)

    df_prob = pd.concat(records, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12,8))

    sns.lineplot(
        data=df_prob,
        x="t",
        y="probability",
        hue="behavior",
        palette=behavior_palette,
        linewidth=1,
        ax=ax,
    )

    ax.set_ylim(-0.05, 1)
    ax.set_ybound(-0.05, 1)

    ax.set_xlim(-2, 3)
    ax.set_xbound(-2, 3)


    add_pad_off(ax, 0)
    add_laser_period(ax, y=1, x_min=0.025, x_max=0.325, color="red", show_limits=True)

    ax.legend()

    plt.xlabel("time (s)")
    plt.ylabel("Probability")
    plt.title(f"Behavior probability over time of rat {cfg.rat_name}\nCondition: {condition} - Number of trial: {len(subset.groupby('id'))}")
    plt.tight_layout()

    plt.savefig(make_output_path(cfg.paths.analysis / "behavior", f"behavior_probability_{condition}.png"))

    



print("Done !")