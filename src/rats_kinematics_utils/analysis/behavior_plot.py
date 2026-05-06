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



def add_pad_off(ax, x_pad_off, label: str = 'Pad off') -> None: 

    ax.axvline(
        x=x_pad_off,
        color="k",
        lw=1,
        ls="--",
        label=label
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




def get_biggest_condition(cfg, filenames: list[Path]) -> int: 
    bad_trials = {}
    biggest_condition = 0

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))
        bad_trials[metrics_path.stem] = 0

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["date"].month == 5 or \
                trial[cfg.bodypart]["behavior_state"] != "keep_all": 
                bad_trials[metrics_path.stem] += 1
                continue

        nb_good_trials = len(metrics) - bad_trials[metrics_path.stem]
        if nb_good_trials > biggest_condition : 
            biggest_condition = nb_good_trials

    return biggest_condition, bad_trials




def preprocess_trial_behavior(cfg, filenames: list[Path]): 

    behavior = pd.DataFrame()
    time_stamps = pd.DataFrame()
    overall_condition = ""
    n = 0 

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["date"].month == 5 or \
                trial[cfg.bodypart]["behavior_state"] != "keep_all": 
                continue
            
            # extract information 
            condition = trial["condition"]
            laser_state = trial["laser_state"]

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            overall_condition = f"{condition}_{laser_state}_{laser_intensity}"

            etho = trial[cfg.bodypart]["xy_etho"]
            pad_off = trial["pad_off"]

            # behavior time line
            etho_labels = etho["label"].map(behavior_to_int).to_list()
            etho_condition = condition_to_int[overall_condition]

            df_beha = pd.DataFrame({
                "t" : etho["t"],
                "behavior": etho_labels,
                "id": n,
                "condition": etho_condition
            })
            behavior = pd.concat([behavior, df_beha], ignore_index=True)

            first_reach = etho.loc[etho["label"] == "reach"].reset_index(drop=True)
            first_open = etho.loc[etho["label"] == "open"].reset_index(drop=True)
            first_grasp = etho.loc[etho["label"] == "grasp"].reset_index(drop=True)
            first_press = etho.loc[etho["label"] == "press"].reset_index(drop=True)

            # get timestamps for specific time (to align later)
            df_time = pd.DataFrame({
                "id": [n],
                "pad_off": [pad_off],
                "first_reach": [first_reach["t"][0]],
                "first_open": [first_open["t"][0] if not first_open.empty else 0],
                "first_grasp": [first_grasp["t"][0]],
                "first_press": [first_press["t"][0] if not first_press.empty else 0],
            })
            time_stamps = pd.concat([time_stamps, df_time], ignore_index=True)
            n += 1

    return behavior, time_stamps





def preprocess_velocity_behavior(cfg, filenames: list[Path]):

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["date"].month == 5 or \
                trial[cfg.bodypart]["behavior_state"] != "keep_all": 
                continue
            
            condition = trial["condition"]
            laser_state = trial["laser_state"]

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            etho = trial[cfg.bodypart]["xy_etho"]
            velo = trial[cfg.bodypart]["instant_velocity"]
            pad_off = trial["pad_off"]

            # crop around the pad off
            velocity = crop_xy(velo, 
                            start=pad_off - 0.1, 
                            end=pad_off + 0.4)
            croped_etho = crop_xy(etho, 
                            start=pad_off - 0.1, 
                            end=pad_off + 0.4)
            relative_time = velocity["t"] - pad_off


            df = pd.DataFrame({
                    "y_pos": (512 - croped_etho["y"]) * cfg.cm_per_pixel,
                    "velocity": velocity["velocity"],
                    "velocity_behavior": croped_etho["label"],
                    "relative_time": relative_time,

                    "condition": condition,
                    "laser_state": laser_state,
                    "laser_intensity": laser_intensity,
                    "id": n,
                })

            data = pd.concat([data, df], ignore_index=True)

            n += 1

    return data



def preprocess_trajectory_behavior(cfg, filenames: list[Path]):
    n = 0
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["date"].month == 5 or \
                trial[cfg.bodypart]["behavior_state"] != "keep_all": 
                continue
            
            condition = trial["condition"]
            laser_state = trial["laser_state"]

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            etho = trial[cfg.bodypart]["xy_etho"]
            x = etho["x"] * cfg.cm_per_pixel
            y = (512 - etho["y"]) * cfg.cm_per_pixel

            df = pd.DataFrame({
                    "x": x,
                    "y": y,
                    "t": etho["t"],
                    "behavior": etho["label"],

                    "condition": condition,
                    "laser_state": laser_state,
                    "laser_intensity": laser_intensity,
                    "id": n,
                })

            data = pd.concat([data, df], ignore_index=True)


            n += 1

    return data




def plot_metric_behavior(cfg, data: pd.DataFrame, metric: str) :

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
            x="relative_time",
            y=metric, 
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
            x="relative_time",
            y=metric, 
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
    g.figure.suptitle(f"Behavioral analysis of the {metric} during the laser stimulation of rat {cfg.rat_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')

    sns.move_legend(g,
                        borderpad=1,
                        loc='upper right', 
                        facecolor="lightgray")

    g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"{metric}.png"))


    fig = g.figure
    fig.tight_layout()
    plt.show()




def plot_trajectory_behavior(cfg, data, crop: bool = False): 
    
    # crop 
    if crop: 
        data = data[(data["t"] >= 0.3) & (data["t"] <= 1)] 
        crop_name = "_cropped"
    else : 
        crop_name = ""

    for laser_intensity, subset in data.groupby("laser_intensity"): 

        mean_traj = (
            subset
            .groupby(["condition", "laser_state", "t"], as_index=False)
            .agg({"x": "mean", "y": "mean"})
        )

        g = sns.FacetGrid(
                data=subset,
                col="condition",
                row="laser_state",
                row_order=["LaserOn", "LaserOff"],
                margin_titles=True,
                height=4,      # size of each facet
                aspect=1.2     # width/height ratio
            )

        g.map_dataframe(
                sns.lineplot, 
                x="x",
                y="y", 
                estimator=None, 
                errorbar=None,
                color="black",
                alpha=0.08,
                units="id",
                sort=False,
                lw=.5
        )

        g.map_dataframe(
                sns.scatterplot, 
                x="x",
                y="y", 
                hue="behavior",
                palette=behavior_palette,
                s=8
        )


        g.add_legend(title="Behavior")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels("x (cm)", "y (cm)")
        g.figure.subplots_adjust(top=0.88)
        g.figure.suptitle(f"Behavior analysis{cfg.rat_name}\nLaser intensity: {laser_intensity} - Number of trials: {len(subset.groupby('id'))}", ha='center')

        sns.move_legend(g,
                            borderpad=1,
                            loc='upper right', 
                            facecolor="lightgray")

        if not crop: 
            for ax in g.axes.flat:
                ax.set_ylim(0, cfg.frame_width_cm)
                ax.set_xlim(0, cfg.frame_width_cm)


        g.savefig(make_output_path(cfg.paths.analysis / f"behavior" / "trajectories", f"trajectories_{laser_intensity}{crop_name}.png"))
        plt.close()




def plot_time_in_behavior_space(cfg, data): 

    for laser_intensity, subset in data.groupby("laser_intensity"): 

        count = (
            subset.groupby(["condition", "laser_state", "behavior"])
                .size()
                .groupby(level=[0, 1])          # within each facet
                .transform(lambda x: ((x / x.sum()) * 100).round(1))
                .reset_index(name="proportion"))

        g = sns.FacetGrid(
                data=count,
                col="condition",
                row="laser_state",
                margin_titles=True,
                height=4,      # size of each facet
                aspect=1.2     # width/height ratio
        )

        g.map_dataframe(
                sns.barplot,
                x="behavior",
                y="proportion",
                hue="behavior",
                palette=behavior_palette,
                order=behavior_palette.keys(),
                legend=True,
        )



        g.add_legend(title="Behavior")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.figure.subplots_adjust(top=0.88)
        g.figure.suptitle(f"Behavior proportion of rat {cfg.rat_name}\nLaser intensity: {laser_intensity} - Number of trials: {len(subset.groupby('id'))}", ha='center')

        sns.move_legend(g,
                            borderpad=1,
                            loc='upper right', 
                            facecolor="lightgray")

        for ax in g.axes.flat:
            for i in range(5): 
                ax.bar_label(ax.containers[i], fontsize=10)
            ax.set_ylabel("proportion (%)")
            ax.set_ylim(0, 53)


        g.savefig(make_output_path(cfg.paths.analysis / f"behavior", f"time_spend_per_behavior_{laser_intensity}.png"))

        plt.close()




def longest_run(series, target):
    # True where behavior == target
    mask = series == target
    
    # Identify consecutive groups
    groups = (mask != mask.shift()).cumsum()
    
    # Count lengths of True groups only
    run_lengths = mask.groupby(groups).sum()
    
    return run_lengths.max() if not run_lengths.empty else 0






def ethogram_by_condition(cfg, behavior, biggest_condition, align_by: str = "pad_off"):     
    """
    behavior dataframe must contain the folowing columns: 
        id, t, behavior, id, condition, 
        pad_off, first_reach, first_open, first_grasp, first_press 
    """
    new_behavior = behavior.copy()

    # align to a specific time stamps
    new_behavior["t_aligned"] = behavior["t"] - behavior[align_by]
    new_behavior["t_aligned"] = new_behavior["t_aligned"].round(3)

    print(f"\nMaking ethogram aligned to: {align_by}")
    int_behavior_palette = ["lightgray", "gold", "orange", "cornflowerblue", "yellowgreen", ]
                            

    for int_cond, subset in new_behavior.groupby("condition"): 
        condition = int_to_condition[int_cond]
        print("  Ethogram of:", condition)

        mat_subset = subset.pivot(
            index="t_aligned",
            columns="id",
            values="behavior"
        ).sort_index()
        
        if align_by == "pad_off" : 
            target_behavior = 1
        else :
            target_behavior = behavior_to_int[align_by.split("_")[1]]

        scores = {
            col: longest_run(mat_subset[col], target_behavior)
            for col in mat_subset.columns
        }

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        mat_subset = mat_subset[sorted_ids]


        fig, ax = plt.subplots(figsize=(15, 8))

        # Heatmap
        sns.heatmap(
            mat_subset.T,
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

        # set tick positions
        time_values = mat_subset.index.to_numpy()
        target = np.array([-1, 0, 1, 2, 3])
        tick_idx = np.where(np.isin(time_values, target))[0]

        # Axis limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # set limits
        lim_min = np.argmin(np.abs(time_values)) - 125
        lim_max = np.argmin(np.abs(time_values - 3))

        # set ticks
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(np.round(time_values[tick_idx], 2))
        ax.spines['bottom'].set_bounds(lim_min, lim_max) 
        ax.spines['bottom'].set_visible(True) 
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(0, biggest_condition)

        # show 0 
        x0 = np.argmin(np.abs(time_values))
        add_pad_off(ax, x0, label=align_by.replace("_", " "))


        if align_by == "pad_off": 
            laser_start = np.argmin(np.abs(time_values - 0.025))
            laser_end   = np.argmin(np.abs(time_values - 0.325))
            add_laser_period(ax, ymax-1, laser_start, laser_end, color="red", show_limits=True)

        ax.legend(loc="upper left")

        plt.suptitle(f"Ethogram across frames for rat {cfg.rat_name}\nCondition: {condition}, Number of trials: {len(subset.groupby('id'))}", y=0.98)
        plt.tight_layout()

        plt.savefig(make_output_path(cfg.paths.analysis / "behavior" / f"ethogram" / f"_align_to_{align_by}", f"ethogram_{condition}.png"))
        plt.close()





def behavior_proba_all(cfg, behavior): 

    mat_behavior = behavior.pivot(
        index="t",
        columns="id",
        values="behavior"
    ).sort_index()

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
        linewidth=3,
        ax=ax,
    )
    ax.set_ylim(-0.05, 1)
    ax.set_ybound(-0.05, 1)

    ax.set_xlim(-1, 3)
    ax.set_xbound(-1, 3)

    add_pad_off(ax, 0)
    add_laser_period(ax, y=ax.get_ylim()[1], x_min=0.025, x_max=0.325, color="red", show_limits=True)

    ax.legend()

    plt.xlabel("time (s)")
    plt.ylabel("Probability")
    plt.title(f"Behavior probability over time of rat {cfg.rat_name}\nAll condition mixed - Number of trials: {len(behavior.groupby('id'))}")
    plt.tight_layout()

    plt.savefig(make_output_path(cfg.paths.analysis / "behavior", "behavior_probability_all_condition.png"))



def behavior_proba_per_condition(cfg, behavior, align_by: str = "pad_off"): 
    from scipy.ndimage import gaussian_filter1d

    print(f"\nMaking probability aligned to: {align_by}")

    new_behavior = behavior.copy()

    # align to a specific time stamps
    new_behavior["t_aligned"] = behavior["t"] - behavior[align_by]
    new_behavior["t_aligned"] = new_behavior["t_aligned"].round(3)

    for int_cond, subset in new_behavior.groupby("condition"): 
        condition = int_to_condition[int_cond]
        print("  ", condition)

        mat_subset = subset.pivot(
            index="t_aligned",
            columns="id",
            values="behavior"
        ).sort_index()


        arr = mat_subset.to_numpy()
        time_values = mat_subset.index.to_numpy()

        records = []
        records_per_id = []

        for state, label in int_to_behavior.items():

            if state == 0 :     # don't show the "none" behavior
                continue

            prob_per_id = (arr == state) 
            prob = prob_per_id.mean(axis=1)

            cdf = np.cumsum(prob)
            if cdf[-1] != 0:
                cdf = cdf / cdf[-1]

            tmp = pd.DataFrame({
                "t_aligned": time_values,
                "behavior": label,
                "probability": prob,
                "CDF": cdf,
                "density": gaussian_filter1d(prob, sigma=2)
            })

            records.append(tmp)

            df_long = pd.DataFrame(prob_per_id, index=time_values)
            df_long.index.name = "t_aligned"
            df_long = df_long.reset_index().melt(id_vars="t_aligned", var_name="id", value_name="probability")
            df_long["behavior"] = label

            records_per_id.append(df_long)

        df_prob = pd.concat(records, ignore_index=True)
        df_prob_per_id = pd.concat(records_per_id, ignore_index=True)

        fig, ax = plt.subplots(figsize=(12,8))

        sns.lineplot(
            data=df_prob_per_id,
            x="t_aligned",
            y="probability",
            hue="behavior",
            palette=behavior_palette,
            linewidth=3,
            estimator="mean",
            errorbar=("ci", 50),
            ax=ax,
        )

        # for beha, sub_df in df_prob.groupby("behavior"):
        #     plt.fill_between(
        #         sub_df["t_aligned"].values,
        #         sub_df["probability"].values,
        #         color=behavior_palette[beha],
        #         alpha=0.3
        #     )

        ax.set_ylim(-0.05, 1)
        ax.set_ybound(-0.05, 1)

        ax.set_xlim(-1, 3)
        ax.set_xbound(-1, 3)


        add_pad_off(ax, 0)
        add_laser_period(ax, y=1, x_min=0.025, x_max=0.325, color="red", show_limits=True)

        ax.legend()

        plt.xlabel("time (s)")
        plt.ylabel("Probability")
        plt.title(f"Behavior probability over time of rat {cfg.rat_name}\nCondition: {condition} - Number of trial: {len(subset.groupby('id'))}")
        plt.tight_layout()

        plt.savefig(make_output_path(cfg.paths.analysis / "behavior" / "probability" / f"ci-50", f"behavior_probability_{condition}.png"))

        plt.close()
