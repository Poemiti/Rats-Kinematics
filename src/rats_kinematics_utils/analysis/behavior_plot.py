#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
import joblib
import time
import seaborn as sns
import sys
import matplotlib.colors as mcolors


from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path, load_trial_data, check_exclusion_rules
import rats_kinematics_utils.analysis.clustering as c
from rats_kinematics_utils.preprocessing.preprocess import crop_xy


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("paper", style="ticks", rc=custom_params, palette="pastel")

def on_off_palette(data) : 
    # define colors of each rats
    cond = data["condition"].unique()
    cond = sorted(cond)
    base_colors = sns.color_palette("tab10", len(cond))

    def lighten(color, amount=0.45):
        c = mcolors.to_rgb(color)
        return tuple(1 - amount*(1 - x) for x in c)

    pal = {}

    for c, base in zip(cond, base_colors):
        pal[f"{c}_LaserOn"] = base
        pal[f"{c}_LaserOff"] = lighten(base)

    return pal


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
CONDITION_PALETTE = {
    "Beta": "forestgreen",
    "Conti": "orange" 
}

dash = {
    "LaserOff" : (2,2),
    "LaserOn" : ""
}


behavior_palette = {
    "none": "lightgray",
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



def add_laser_period(ax, y, x_min, x_max, color: str= "red", show_limits: bool=False) -> None: 

    ax.hlines(
        y=y,
        xmin=x_min ,
        xmax=x_max,
        color=color,
        linewidth=5,
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




def get_biggest_condition(cfg, data) -> str:

    count = (
        data.groupby("condition")["id"]
        .nunique()
    )

    biggest_condition = count.idxmax()
    n_trials = count.max()

    return int_to_condition[biggest_condition], n_trials


# -------------------------------------------------------- preprocessing --------------------------------------


def preprocess_trial_behavior(cfg, filenames: list[Path], remove_may: bool= True): 

    behavior = pd.DataFrame()
    time_stamps = pd.DataFrame()
    overall_condition = ""
    n = 0 

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            if remove_may and trial["date"].month == 5 : 
                continue

            if not check_exclusion_rules(trial) : 
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
                "first_grasp": [first_grasp["t"][0] if not first_grasp.empty else 0],
                "first_press": [first_press["t"][0] if not first_press.empty else 0],
            })
            time_stamps = pd.concat([time_stamps, df_time], ignore_index=True)
            n += 1

    return behavior, time_stamps





def preprocess_metric_behavior(cfg, filenames: list[Path]):
    n=0
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["date"].month == 5 or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            if not check_exclusion_rules(trial) : 
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



def preprocess_trajectory_behavior(cfg, filenames: list[Path], remove_may: bool = False):
    n = 0
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            if remove_may and trial["date"].month == 5: 
                continue

            if not check_exclusion_rules(trial) : 
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
                    "pad_off": trial["pad_off"],

                    "condition": condition,
                    "laser_state": laser_state,
                    "laser_intensity": laser_intensity,
                    "id": n,
                })

            data = pd.concat([data, df], ignore_index=True)


            n += 1

    return data




def preprocess_proba(cfg, filenames, remove_may: bool = True, reward: str= "both"): 
    data = pd.DataFrame()
    n = 0 

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            if remove_may and trial["date"].month == 5: 
                continue

            if reward == "reward_only" and trial["reward"] is None: 
                continue
            elif reward == "non_reward_only" and trial["reward"] is not None: 
                continue

            
            if not check_exclusion_rules(trial) : 
                continue

            # extract information 
            condition = trial["condition"]
            laser_state = trial["laser_state"]

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            etho = trial[cfg.bodypart]["xy_etho"]
            pad_off = trial["pad_off"]

            # behavior time line
            etho_labels = etho["label"].map(behavior_to_int).to_list()

            df_beha = pd.DataFrame({
                "t" : etho["t"] - pad_off,
                "behavior": etho_labels,
                "id": n,

                "condition": condition,
                "laser_state": laser_state,
                "laser_intensity": laser_intensity,
            })
            data = pd.concat([data, df_beha], ignore_index=True)

            n+=1

    return data


def preprocess_reward(cfg, filenames, remove_may: bool = False): 
    
    data = pd.DataFrame()
    n=0

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in enumerate(metrics) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            if remove_may and trial["date"].month == 5: 
                continue
            
            if not check_exclusion_rules(trial) : 
                continue

            condition = trial["condition"]
            laser_state = trial["laser_state"]
            pad_off = trial["pad_off"]

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            reward = trial["reward"] is not None
            reward_during_laser = trial["reward"] is not None and pad_off < trial["reward"] < pad_off + 0.325 

            df = pd.DataFrame({
                "rewarded": [reward],
                "rewarded_during_laser": [reward_during_laser],
                "cond_state": [condition + "_" + laser_state],
                "condition": [condition],
                "laser_state": [laser_state],
                "laser_intensity": [laser_intensity],
                "id": [n],
            })

            data = pd.concat([data, df], ignore_index=True)

            n+=1

    return data


# -------------------------------------------------------- plotting --------------------------------------


def plot_metric_behavior(cfg, data: pd.DataFrame, metric: str) :

    for laser_intensity, subset in data.groupby("laser_intensity"): 

        g = sns.FacetGrid(
                data=subset,
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

        for row_i, laser_state in enumerate(g.row_names):
            for col_j, condition in enumerate(g.col_names):

                ax = g.axes[row_i, col_j]
                # ax.set_ylim(-5, 80)

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
        g.figure.suptitle(f"Behavioral analysis of the {metric} during the laser stimulation of rat {cfg.rat_name}\nNumber of trials: {len(subset.groupby('id'))}", ha='center')

        sns.move_legend(g,
                            borderpad=1,
                            loc='upper right', 
                            facecolor="lightgray")

        g.savefig(make_output_path(cfg.paths.analysis / f"{cfg.rat_type}_behavior" / "metrics_across_time", f"{cfg.rat_type}_{metric}_{laser_intensity}.png"))

        plt.close()




def plot_trajectory_behavior(cfg, data, crop: bool = False): 
    
    # crop 
    if crop: 
        data = data[(data["t"] >= 0.3) & (data["t"] <= data["pad_off"] + 0.325)] 
        crop_name = "_cropped"
    else : 
        crop_name = ""

        mean_traj = (
            data
            .groupby(["condition", "laser_state", "t"], as_index=False)
            .agg({"x": "mean", "y": "mean"})
        )

        g = sns.FacetGrid(
                data=data,
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

        sns.move_legend(g,
                            borderpad=1,
                            loc='upper right', 
                            facecolor="lightgray")

        if not crop: 
            for ax in g.axes.flat:
                ax.set_ylim(0, cfg.frame_width_cm)
                ax.set_xlim(0, cfg.frame_width_cm)
        
        return g, crop_name




def plot_time_in_behavior_space(cfg, data): 
        
        print(data)

        # Raw counts
        count = (
            data.groupby(["condition", "laser_state", "behavior"])
            .size()
            .reset_index(name="n")
        )

        # Proportions within each condition + laser_state
        count["proportion"] = (
            count.groupby(["condition", "laser_state"])["n"]
            .transform(lambda x: (x / x.sum()) * 100)
            .round(1)
        )

        # Remove unwanted behavior
        count = count[count["behavior"] != 0].reset_index(drop=True)

        print(count)

        g = sns.catplot(
            data=count,
            kind="bar",
            col="condition",

            x="behavior",
            y="proportion",
            hue="laser_state",
            palette=LASER_STATE_PALETTE,
            # order=[1, 2, 3, 4],
            order=["reach", "open", "grasp", "press"],
            hue_order=["LaserOff", "LaserOn"],
        )

        for ax in g.axes.flat:
            for container in ax.containers:

                labels = []

                for bar in container:
                    height = bar.get_height()

                    # Match bar -> dataframe row
                    x = bar.get_x() + bar.get_width() / 2

                    labels.append(f"{height:.1f}%")

                ax.bar_label(container, labels=labels, fontsize=10)

            ax.set_ylabel("proportion (%)")
            ax.set_ylim(0,100)
            ax.set_xlabel(["reach", "open", "grasp", "press"])

        return g




def longest_run(series, target):
    # True where behavior == target
    mask = series == target
    
    # Identify consecutive groups
    groups = (mask != mask.shift()).cumsum()
    
    # Count lengths of True groups only
    run_lengths = mask.groupby(groups).sum()
    
    return run_lengths.max() if not run_lengths.empty else 0






def plot_ethogram_by_condition(cfg, behavior, biggest_condition, output_dir, align_by: str = "pad_off"):     
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
        lim_max = np.argmin(np.abs(time_values - 2))

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
            add_laser_period(ax, biggest_condition, laser_start, laser_end, color="red", show_limits=True)

        ax.legend(loc="upper left")

        plt.suptitle(f"Ethogram across frames\nCondition: {condition}, Number of trials: {len(subset.groupby('id'))}", y=0.98)
        plt.tight_layout()

        plt.savefig(make_output_path(output_dir / f"{cfg.rat_type}_behavior" / f"ethogram" / f"_align_to_{align_by}", f"{cfg.rat_type}_ethogram_{condition}.png"))
        plt.savefig(make_output_path(output_dir / f"{cfg.rat_type}_behavior" / f"ethogram" / f"_align_to_{align_by}", f"{cfg.rat_type}_ethogram_{condition}.svg"))

        plt.close()





def plot_behavior_proba_all(cfg, behavior): 

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

    plt.savefig(make_output_path(cfg.paths.analysis / f"{cfg.rat_type}_behavior", f"{cfg.rat_type}_behavior_probability_all_condition.png"))



def plot_behavior_proba_per_condition(cfg, behavior, output_dir: Path, align_by: str = "pad_off"): 
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

        ax.set_ylim(-0.02, 1)
        ax.set_ybound(-0.02, 1)

        ax.set_xlim(-0.2, 2)
        ax.set_xbound(-0.2, 2)


        add_pad_off(ax, 0)
        add_laser_period(ax, y=1, x_min=0.025, x_max=0.325, color="red", show_limits=True)

        # ax.legend(loc="upper right")

        plt.xlabel("time (s)")
        plt.ylabel("Probability")
        plt.title(f"Behavior probability over time\nCondition: {condition} - Number of trial: {len(subset.groupby('id'))}")
        plt.tight_layout()

        plt.savefig(make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / f"ci-50", f"{cfg.rat_type}_behavior_probability_{condition}.png"))
        plt.savefig(make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / f"ci-50", f"{cfg.rat_type}_behavior_probability_{condition}.svg"))

        plt.close()




def _behavior_probability_over_time_per_id(state, df: pd.DataFrame) -> pd.DataFrame: 

    mat_subset = df.pivot(
        index="t",
        columns="id",
        values="behavior"
    ).sort_index()


    arr = mat_subset.to_numpy()
    time_values = mat_subset.index.to_numpy()

    prob_per_id = (arr == state) 

    df_prob_per_id = pd.DataFrame(
        prob_per_id,
        index=time_values,
        columns=mat_subset.columns
    )

    return (
        df_prob_per_id
        .reset_index(names="t")
        .melt(
            id_vars="t",
            var_name="id",
            value_name="probability"
        )
    )



def plot_behavior_proba_per_behavior(cfg, data, output_dir: Path): 

    print(data)
    data["t"] = data["t"].round(3)

    pal = on_off_palette(data)

    for state, beha in int_to_behavior.items():

        if state == 0 : 
            continue

        records = []

        for (cond, l_state, l_intensity), subset in data.groupby(["condition", "laser_state", "laser_intensity"]): 

            df_prob_per_id = _behavior_probability_over_time_per_id(state, subset)

            df_prob_per_id["behavior"] = beha
            df_prob_per_id["condition"] = cond + "_" + l_state
            df_prob_per_id["laser state"] = l_state
            df_prob_per_id["l_intensity"] = l_intensity

            records.append(df_prob_per_id)

        df_records = pd.concat(records, ignore_index=True)
        print(df_records)

        # count the number of trials
        n_trials = df_records["id"].nunique()
        trial_counts = (df_records.groupby("l_intensity")["id"].nunique())

        # plot
        g = sns.relplot(
            kind="line",
            data=df_records,
            col="l_intensity",
            col_order=["low", "high"],

            x="t",
            y="probability",
            hue="condition",
            style="laser state",

            palette=pal,
            dashes=dash,
            estimator="mean",
            errorbar=("ci", 50),
        )

        axes = np.ravel(g.axes)

        # add pad off, laser period and labels
        for ax in axes:

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.2, 2)

            add_pad_off(ax, 0)
            add_laser_period(ax, y=1.01, x_min=0.025, x_max=0.325, color="royalblue")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Probability")

        # add laser intensity label and counts
        for ax, intensity in zip(g.axes.flat, g.col_names):

            n = trial_counts.get(intensity, 0)
            ax.set_title(f"{intensity} - N trials: {n}" )



        g.figure.suptitle(
            f"{beha} probability over time\n"
            f"Trials: {n_trials}",
        )

        g.tight_layout()
        g.figure.subplots_adjust(top=0.78)

        output = make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / "per_behavior",f"{cfg.rat_type}_probability_{beha}.png")
        g.savefig(output, dpi=300)
        output = make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / "per_behavior",f"{cfg.rat_type}_probability_{beha}.svg")
        g.savefig(output, dpi=300)

        plt.close(g.figure)




def plot_mean_proba_per_behavior(cfg, data, output_dir: Path): 

    print(data)
    data["t"] = data["t"].round(3)

    for state, beha in int_to_behavior.items():

        if state == 0 : 
            continue

        records = []

        for (cond, l_state, l_intensity), subset in data.groupby(["condition", "laser_state", "laser_intensity"]): 

            df_prob_per_id = _behavior_probability_over_time_per_id(state, subset)

            mean_per_id = (
                df_prob_per_id.groupby("id")["probability"]
                .mean()
                .reset_index(name="mean_proba")
            )

            print(mean_per_id)

            mean_per_id["behavior"] = beha
            mean_per_id["condition"] = cond 
            mean_per_id["laser state"] = l_state
            mean_per_id["l_intensity"] = l_intensity

            records.append(mean_per_id)

        df_records = pd.concat(records, ignore_index=True)
        print(df_records)

                # count the number of trials
        n_trials = df_records["id"].nunique()
        trial_counts = (df_records.groupby("l_intensity")["id"].nunique())

        # plot
        g = sns.catplot(
            kind="bar",
            data=df_records,
            col="l_intensity",
            col_order=["low", "high"],

            x="condition",
            y="mean_proba",
            hue="laser state",
            palette=LASER_STATE_PALETTE,

            estimator="mean",
            errorbar=("ci", 50),
        )

        # add laser intensity label and counts
        for ax, intensity in zip(g.axes.flat, g.col_names):

            n = trial_counts.get(intensity, 0)
            ax.set_title(f"{intensity} - N trials: {n}" )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Probability")

            ax.set_ylim(0,1)

            for c in ax.containers: 
                ax.bar_label(c, labels=np.round(c.datavalues, 2), fontsize=11, padding=5,)



        g.figure.suptitle(
            f"Mean probability during laser stim : {beha}\n"
            f"Trials: {n_trials}",
        )

        g.tight_layout()
        g.figure.subplots_adjust(top=0.78)

        output = make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / "mean_proba_per_behavior",f"{cfg.rat_type}_probability_{beha}.png")
        g.savefig(output, dpi=300)
        output = make_output_path(output_dir / f"{cfg.rat_type}_behavior" / "probability" / "mean_proba_per_behavior",f"{cfg.rat_type}_probability_{beha}.svg")
        g.savefig(output, dpi=300)

        plt.close(g.figure)



def _draw_transition_graph(
            transition_matrix,
            threshold=0.0,
        ):
    """
    Draw a directed transition graph from a transition matrix.

    Parameters
    ----------
    transition_matrix : pd.DataFrame
        Rows = previous behaviors
        Columns = next behaviors
        Values = transition probabilities

    threshold : float
        Minimum probability to display an edge.

    layout : str
        'circular', 'spring', or 'kamada_kawai'
    """
    import networkx as nx

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for node in transition_matrix.index:
        G.add_node(node)

    # Add weighted edges
    for prev_behavior in transition_matrix.index:
        for next_behavior in transition_matrix.columns:

            prob = transition_matrix.loc[prev_behavior, next_behavior]

            if prob >= threshold:
                G.add_edge(
                    prev_behavior,
                    next_behavior,
                    weight=prob
                )


    #  layout
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots()

    # Node colors
    node_colors = [behavior_palette.get(node, "lightgray")
                    for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        node_color=node_colors,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    # Edge weights
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [w * 10 for w in edge_weights]


    # Group edges between same node pairs
    seen = {}

    for (u, v, d), w, width in zip(G.edges(data=True), edge_weights, edge_widths):

        # key ignoring direction (to detect A<->B pairs)
        key = tuple(sorted([u, v]))
        seen.setdefault(key, 0)

        # curvature: opposite directions get opposite bend
        rad = 0.2 * (seen[key] + 1)

        # draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            width=width,
            node_size=2000,
            edgelist=[(u, v)],
            # width=3,
            edge_color=behavior_palette.get(u, "lightgray"),
            # edge_cmap=plt.cm.Reds,
            # edge_vmin=min(edge_weights),
            # edge_vmax=max(edge_weights),
            arrows=True,
            arrowstyle="->",
            arrowsize=23,
            connectionstyle=f"arc3,rad={rad}",
            ax=ax,
        )

        # draw corresponding labels
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): f"{d['weight']:.2f}"},
            font_size=12,
            connectionstyle=f"arc3,rad={rad}",
            ax=ax
        )


    # Optional colorbar
    # sm = plt.cm.ScalarMappable(
    #     cmap=plt.cm.Reds,
    #     norm=plt.Normalize(
    #         vmin=min(edge_weights),
    #         vmax=max(edge_weights)
    #     )
    # )

    # sm.set_array([])
    # plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.85)
    # plt.subplots_adjust(top=0.9, right=0.88)
    
    return ax



def plot_transition_matrix(cfg, data): 

    # Ensure proper ordering
    data = data.sort_values(["id", "t"])

    # Detect behavior changes within each trial
    data["behavior_change"] = (
        data.groupby("id")["behavior"]
        .diff()
        .ne(0)
    )

    # Keep only rows where behavior changes
    transitions = data[data["behavior_change"]].copy()

    # Previous behavior before the change
    transitions["prev_behavior"] = (
        transitions.groupby("id")["behavior"]
        .shift(1)
    )

    # Current behavior after the change
    transitions["next_behavior"] = transitions["behavior"]

    # Remove first rows of trials (where no previous behavior exists)
    transitions = transitions.dropna(subset=["prev_behavior"])

    # Convert to int if needed
    transitions["prev_behavior"] = transitions["prev_behavior"].astype(int)
    transitions["next_behavior"] = transitions["next_behavior"].astype(int)

    print(transitions)

    # Count transitions
    trans_mat = pd.crosstab(
        transitions["prev_behavior"],
        transitions["next_behavior"],
        normalize="index"   # row-wise probabilities
    )

    print(trans_mat)

    # Ensure all behaviors exist
    behaviors = [0, 1, 2, 3, 4]

    trans_mat = trans_mat.reindex(
        index=behaviors,
        columns=behaviors,
        fill_value=0
    )

    # rename behavior with real name
    trans_mat = trans_mat.rename(
        index=int_to_behavior,
        columns=int_to_behavior,
    )

    print(trans_mat)

    # Plot
    ax = sns.heatmap(
        trans_mat,

        annot=True,
        fmt=".2f",

        cmap="Reds",

        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={
            "label": "Transition probability"
        }
    )

    return ax, trans_mat




def plot_proportion_behavior_combinaison(cfg ,data): 

    # ---------- unique behaviors per trial ----------
    combo = (
        data.groupby(["id", "laser_state"])["behavior"]
        .unique()
        .apply(lambda x: sorted(set(x) - {0}))
        .reset_index(name="behaviors")
    )

    # Remove empty trials
    combo = combo[
        combo["behaviors"].apply(len) > 0
    ]

    # Convert to readable string
    combo["combination"] = combo["behaviors"].apply(
        lambda x: "+".join(map(str, x))
    )

    # ---------- counts ----------
    count = (
        combo.groupby(["laser_state", "combination"])
        .size()
        .reset_index(name="n")
    )

    # Create all possible combinations
    all_idx = pd.MultiIndex.from_product(
        [
            combo["laser_state"].unique(),
            combo["combination"].unique()
        ],
        names=["laser_state", "combination"]
    )

    # Reindex and fill missing with 0
    count = (
        count.set_index(["laser_state", "combination"])
        .reindex(all_idx, fill_value=0)
        .reset_index()
    )


    # ---------- proportions ----------

    count["proportion"] = (
            count.groupby("laser_state")["n"]
            .transform(lambda x: (x / x.sum()) * 100)
            .round(1)
        )
    

    # ---------- complexity ----------
    count["n_behaviors"] = (
        count["combination"]
        .str.count(r"\+") + 1
    )

    # ---------- sorting ----------
    count = count.sort_values(
        ["n_behaviors", "n"],
        ascending=[True, True]
    )

    print(count)

    # ---------- plotting ----------
    g = sns.catplot(
        kind="bar",
        data=count,

        x="combination",
        y="proportion",
        order=["1", "1+2", "1+3", "1+3+4", "1+2+3", "1+2+3+4"],

        hue="laser_state",
        palette=LASER_STATE_PALETTE,
        hue_order=["LaserOff", "LaserOn"],

        height=6,
        aspect=1.5,
    )

    # ---------- labels ----------
    for ax in g.axes.flat:

        for c, (_, row_subset) in zip(
            ax.containers,
            count.groupby("laser_state")
        ):
            labels = [
                f"{p:.1f}%\n(n: {int(n)})"
                for p, n in zip(
                    c.datavalues,
                    row_subset["n"].values
                )
            ]
            print(labels)

            ax.bar_label(
                c,
                labels=labels,
                fontsize=11,
                padding=3,
            )

        ax.set_xlabel("Behavior combinations")
        ax.set_ylabel("Proportion (%)")

        ax.set_ylim(0, 60)

    return ax