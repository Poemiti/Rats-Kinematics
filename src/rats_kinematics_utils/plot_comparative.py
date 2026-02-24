import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
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



def _plot_violin_distribution(cfg, data) : 

    data_trimmed = _trim_extremes_iqr(data, k=1.5)
    print(f"\nNumber of removed outliers : {len(data) - len(data_trimmed)}")
    # print(data_trimmed)

    g = sns.FacetGrid(data_trimmed, 
                      col='laser_state',
                      col_order=["LaserOn", "LaserOff"],
                      hue="laser_intensity",    
                       
                      palette="pastel", 
                      margin_titles=True,
                    #   sharey=False,
                      height=4, aspect=0.8)
    g.map_dataframe(
        sns.stripplot,
        x="condition",
        y="value",
        data=data_trimmed,
        dodge=True,
        color="black",
        size=2.5,
        alpha=0.5,
        order=["NOstim", "Conti", "Beta"],
        legend=False, 
    )

    g.map_dataframe(
        sns.violinplot,
        x="condition",
        y="value",
        data=data_trimmed,
        hue_order=data["condition"].unique(),
        split=True,
        gap= .1,
        inner="quart",
        order=["NOstim", "Conti", "Beta"],
    )
    
    

    counts = (
        data_trimmed
        .groupby(["laser_state", "condition", "laser_intensity"])
        .size()
        .reset_index(name="N")
    )

    order = ["NOstim", "Conti", "Beta"]
    intensities = data_trimmed["laser_intensity"].unique()

    for ax, laser_state in zip(g.axes.flatten(), ["On", "Off"]):
        subset = counts[counts["laser_state"] == laser_state]
        y_max = data_trimmed["value"].max()
        y_offset = 0.05 * y_max

        for _, row in subset.iterrows():
            x = order.index(row["condition"])

            # dodge offset (adjust depending on how many intensities you have)
            if row["laser_intensity"] == intensities[0]:
                x -= 0.2
            else:
                x += 0.2

            ax.text(
                x,
                y_max + y_offset,
                f"{row['N']}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )
    g.add_legend(title="Laser intensity")

    return g



def plot_violin_distribution_velocity() : 
    pass

def plot_violin_distribution_tortuosity() : 
    pass

def plot_violin_distribution_peak() : 
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