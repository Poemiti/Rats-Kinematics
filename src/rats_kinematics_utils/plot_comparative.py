import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import time

from rats_kinematics_utils.trajectory_metrics import crop_xy


# ==================================== Plots for comparative analysis ===========================================


def _relative_metric(metric_list: pd.Series, 
                     relative_time: pd.Series,
                     ax: plt.axes,
                     color: str,
                     laser_on: bool = False,
                     show_pad_off: bool = False,
                     transparancy: float=0.7) -> plt.axes : 

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(relative_time, metric_list, color= color, alpha=transparancy)
    if show_pad_off :
        # add line for pad off 
        ax.axvline(0, color='k', lw=0.8, ls='--', label="pad off")
        ax.legend()

        # show laser on
        if laser_on :
            laser_on = 0 + 0.025
            laser_off = laser_on +  0.3 # sec or 37.5 frame
            ax.axvspan(laser_on, laser_off, color='red', alpha=0.3, label="laser on")
            ax.legend()
        
    return ax



def plot_stacked_velocity(cfg, metrics: dict) :
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    fig, axs = plt.subplots(figsize=(9, 7))

    aligned_trials = []

    for trial in metrics:
        if not check_trial_success(trial):
            continue

        trial_name = trial["filename_clips"].stem
        velo = trial["instant_velocity"]
        relative_time = velo["t"] - trial["pad_off"]

        _relative_metric(velo["velocity"],
                         relative_time=relative_time,
                         ax=axs, 
                         color="green",
                         transparancy=0.33)

        df = pd.DataFrame({
            "velocity": velo["velocity"].values
        }, index=relative_time.values)

        aligned_trials.append(df)

    common_time = np.linspace(-0.5, 1.0, 500)
    aligned_resampled = []

    for df in aligned_trials:
        interp_values = np.interp(
            common_time,
            df.index.values,
            df["velocity"].values,
            left=np.nan,
            right=np.nan
        )
        aligned_resampled.append(interp_values)

    aligned_matrix = np.vstack(aligned_resampled)
    avg_velocity = np.nanmean(aligned_matrix, axis=0)

    _relative_metric(
                    metric_list=avg_velocity,
                    relative_time=common_time,
                    ax=axs,
                    show_pad_off=True,
                    laser_on="LaserOn" in trial_name,
                    color="blue",
                    transparancy=1
                )

    
    return axs



def plot_stacked_Yposition(cfg, metrics: dict) :
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    aligned_pos = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        # print(f"\n[{t}/{len(metrics)}]")

        if not check_trial_success(trial) : 
            continue 

        trial_name = trial["filename_clips"].stem
        y_pos = trial["xy_filtered"]
        relative_time = y_pos["t"] - trial["pad_off"]

        _relative_metric(metric_list=y_pos["y"] * cfg.cm_per_pixel,
                        relative_time=relative_time,
                        ax = axs, 
                        laser_on=False, 
                        color="green",
                        transparancy=0.3)
        
        df = pd.DataFrame({"y_pos": y_pos["y"] * cfg.cm_per_pixel}, index=relative_time.values)

        aligned_pos.append(df)

    common_time = np.linspace(-0.5, 1.0, 500)
    aligned_resampled = []

    for df in aligned_pos:
        interp_values = np.interp(
            common_time,
            df.index.values,
            df["y_pos"].values,
            left=np.nan,
            right=np.nan
        )
        aligned_resampled.append(interp_values)

    aligned_matrix = np.vstack(aligned_resampled)
    avg_y_pos = np.nanmean(aligned_matrix, axis=0)

    _relative_metric(metric_list=avg_y_pos,
                    relative_time = common_time,
                    ax = axs, 
                    show_pad_off=True,
                    laser_on="LaserOn" in trial_name,
                    color="blue",
                    transparancy=1)

    return axs




def plot_stacked_trajectories(cfg, metrics, ax: plt.axes = None) : 
    from rats_kinematics_utils.plot import plot_single_bodypart_trajectories
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    all_coords = []
    if not ax :
        fig, ax = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        if not check_trial_success(trial) : 
            continue

        trial_name = trial['filename_clips'].stem
        print(f"\nMaking figures of {trial_name}")

        coords = trial["xy_pad_off"] 

        if trial["laser_on"] is not None:
            frame_laser_on = coords.index[coords["t"] >= trial["laser_on"]][0]
            print(
                f"trial laser on: {trial['laser_on']},\n"
                f"frame laser on: {frame_laser_on}"
            )
        else:
            frame_laser_on = None

        plot_single_bodypart_trajectories(
                coords=coords,
                cm_per_pixel=cfg.cm_per_pixel,
                frame_laser_on=frame_laser_on,
                ax=ax,
                color="green",
                transparancy=0.5,
            )
        all_coords.append(coords)

    # avg_coords = (pd.concat(all_coords, axis=1)
    #                 .T
    #                 .groupby(level=0)
    #                 .mean()
    #                 .T)

    # plot_single_bodypart_trajectories(
    #         coords=avg_coords,
    #         cm_per_pixel=cfg.cm_per_pixel,
    #         frame_laser_on=None,
    #         ax=ax,
    #         color="blue",
    #         transparancy=1,
    #     )

    return ax






















############################################################################# violin ################################################""



# remove extreme values with the interquarile range method (IQR)
def _trim_extremes_iqr(df, value_col="value", group_cols=("condition", "laser_intensity"), k=1.5):
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


def _add_text(ax, x, y, text, color) :
    ax.text(
        x, y, text,
        ha="center", va="bottom", fontsize=8,
        fontweight="bold", color=color, alpha=0.7)


def _display_counts(g, data, data_trimmed, order) : 
    counts_trimmed = (
        data_trimmed
        .groupby(["laser_state", "condition", "laser_intensity", "reward"])
        .size()
        .reset_index(name="N_trim")
    )

    counts_initial = (
        data
        .groupby(["laser_state", "condition", "laser_intensity"])
        .size()
        .reset_index(name="N_initial")
    )

    l_states = data_trimmed["laser_state"].unique()
    l_intensities = data_trimmed["laser_intensity"].unique()

    for ax, laser_state in zip(g.axes.flatten(), l_states):

        subset_trim = counts_trimmed[counts_trimmed["laser_state"] == laser_state]
        subset_init = counts_initial[counts_initial["laser_state"] == laser_state]

        y_max = data_trimmed["value"].max()
        y_offset = 0.05 * y_max

        _add_text(ax, -0.4, y_max + y_offset*3, "tot:", "blue")
        _add_text(ax, -0.4, y_max + y_offset*2, "yes:", "green")
        _add_text(ax, -0.4, y_max + y_offset, "no:", "black")

        for condition in order:
            for intensity in l_intensities:

                # --- X position with dodge ---
                x = order.index(condition)
                hue_index = list(l_intensities).index(intensity)
                total_hue = len(l_intensities)

                dodge_amount = 0.6
                x_offset = (hue_index - (total_hue - 1) / 2) * dodge_amount / total_hue
                x_pos = x + x_offset

                # --- Get counts ---
                init_row = subset_init[
                    (subset_init["condition"] == condition) &
                    (subset_init["laser_intensity"] == intensity)
                ]

                yes_row = subset_trim[
                    (subset_trim["condition"] == condition) &
                    (subset_trim["laser_intensity"] == intensity) &
                    (subset_trim["reward"] == "yes")
                ]

                no_row = subset_trim[
                    (subset_trim["condition"] == condition) &
                    (subset_trim["laser_intensity"] == intensity) &
                    (subset_trim["reward"] == "no")
                ]

                if init_row.empty:
                    continue

                N_initial = int(init_row["N_initial"].values[0])
                N_yes = int(yes_row["N_trim"].values[0]) if not yes_row.empty else 0
                N_no = int(no_row["N_trim"].values[0]) if not no_row.empty else 0

                _add_text(ax, x_pos, y_max + y_offset*3, N_initial, "blue")
                _add_text(ax, x_pos, y_max + y_offset*2, N_yes, "green")
                _add_text(ax, x_pos, y_max + y_offset, N_no, "black")


def _plot_violin_distribution(cfg, data) : 

    data_trimmed = _trim_extremes_iqr(data, k=1.5)
    print(f"\nNumber of removed outliers : {len(data) - len(data_trimmed)}")
    if len(data_trimmed[data_trimmed["condition"]=="NOstim"]) == 0 : 
        data_trimmed = data_trimmed[data_trimmed["condition"]!="NOstim"]
        order = ["Conti", "Beta"]
    else : 
        order = ["NOstim", "Conti", "Beta"]

    print(data_trimmed)
    
    reward_palette = {"no": "black",
                      "yes": "green"}
    laser_intensity_palette = {"low" : "lightblue",
                               "high" : "salmon",
                               "NOstim" : "gray"}

    g = sns.FacetGrid(
        data=data_trimmed,
        col="laser_state",
        margin_titles=True,
        height=4,
        aspect=1
    )

    # VIOLIN
    g.map_dataframe(
        sns.violinplot,
        x="condition",
        y="value",
        hue="laser_intensity",
        split=True,
        inner="quart",
        order=order,
        gap= .1,
        palette=laser_intensity_palette,
        legend=True,
    )

    # STRIP
    g.map_dataframe(
        sns.stripplot,
        x="condition",
        y="value",
        hue="reward",
        palette=reward_palette,
        marker="X",
        size=3,
        alpha=0.7,
        legend=True,
    )
    g.add_legend(title="Laser intensity        Rewarded trial", ncol=2)

    # --------------------------- display counting --------------------------

    _display_counts(g, data, data_trimmed, order)

    return g


def plot_violin_distribution_velocity() : 
    pass

def plot_violin_distribution_tortuosity() : 
    pass

def plot_violin_distribution_peak() : 
    pass







########################################### displot ###############################################


def _plot_displot(data):

    data_trimmed = _trim_extremes_iqr(data, k=1.5)
    print(f"\nNumber of removed outliers: {len(data) - len(data_trimmed)}")

    if (data_trimmed["condition"] == "NOstim").sum() == 0:
        data_trimmed = data_trimmed[data_trimmed["condition"] != "NOstim"]
        row_order = ["Conti", "Beta"]
    else:
        row_order = ["NOstim", "Conti", "Beta"]

    laser_intensity_palette = {"low" : "lightblue",
                               "high" : "salmon",
                               "NOstim" : "gray"}

    # Remove empty combinations to avoid KDE crash
    data_trimmed = data_trimmed.dropna(subset=["value"])

    g = sns.displot(
        data=data_trimmed,
        x="value",
        col="laser_state",
        row="condition",
        hue="laser_intensity",
        kind="kde",
        height=2,
        aspect=2,
        fill=True,
        palette=laser_intensity_palette,
        row_order=row_order,
        facet_kws=dict(margin_titles=True),
        common_norm=False  # prevents normalization issues across subsets
    )

    g.add_legend()

    return g

def plot_displot_velocity(data) : 
    pass

def plot_displot_peak(data) : 
    pass

def plot_displot_tortuosity(data) : 
    pass



########################################### velocity accross trials ###############################################






def plot_velocity_over_cliptime(data) : 

    data_trimmed = _trim_extremes_iqr(data,
                                      value_col="velocity",
                                      group_cols=["condition", "date"],
                                      k=1.5)
    print(f"\nNumber of removed outliers : {len(data) - len(data_trimmed)}")

    g = sns.FacetGrid(
        data=data_trimmed,
        row="condition",
        col="date",
        height=2,
        aspect=2,
        margin_titles=True,
        palette="pastel",
        sharex=False,
    )

    g.map_dataframe(
        sns.lineplot,
            x="clip",
            y="velocity",
            hue="condition",
            data=data_trimmed,
            alpha=0.7,
            estimator=None,
            style="condition",
            hue_order=data["condition"].unique(),
            markers=True
        )
            
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Trial order", "Velocity (cm.s$^{-1}$)")
    g.tight_layout()


    return g