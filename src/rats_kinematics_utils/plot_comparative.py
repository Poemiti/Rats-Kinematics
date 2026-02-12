import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# ==================================== Plots for comparative analysis ===========================================


def relative_metric(velocity_list: pd.Series, 
                     time: pd.Series,
                     ax: plt.axes,
                     color: str,
                     laser_on: bool = False,
                     show_pad_off: bool = False,
                     transparancy: float=0.7) -> plt.axes : 

    if ax is None:
        fig, ax = plt.subplots()

    # relative time
    relative_time = time - time[0]

    ax.plot(relative_time, velocity_list, color= color, alpha=transparancy)
    if show_pad_off :
        # add line for pad off 
        ax.axvline(relative_time[0], color='k', lw=0.8, ls='--', label="pad off")
        ax.legend()

        # show laser on
        if laser_on :
            laser_on = relative_time[0] + 0.025
            laser_off = laser_on +  0.3 # sec or 37.5 frame
            ax.axvspan(laser_on, laser_off, color='red', alpha=0.3, label="laser on")
            ax.legend()
        
    return ax



def plot_stacked_velocity(cfg, metrics: dict) :
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    all_velocity = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        if not check_trial_success(trial) : 
            continue 


        trial_name = trial['filename_clips'].stem
        print(f"Making figures of {trial_name}")

        velo = trial["instant_velocity"]
        all_velocity.append(velo)

        relative_metric(velocity_list=velo["velocity"],
                            time=velo['t'],
                            ax = axs, 
                            laser_on=False, 
                            color="green",
                            transparancy=0.3)
            

    avg_velocity = (
        pd.concat(all_velocity, axis=1)
        .T
        .groupby(level=0)
        .mean()
        .T
    )

    relative_metric(velocity_list=avg_velocity["velocity"],
                    time = avg_velocity["t"],
                    ax = axs, 
                    show_pad_off=True,
                    laser_on=True,
                    color="blue",
                    transparancy=1)

    
    return axs



def plot_stacked_Yposition(cfg, metrics: dict) :
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    all_pos = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        # print(f"\n[{t}/{len(metrics)}]")

        if not check_trial_success(trial) : 
            continue 

        trial_name = trial['filename_clips'].stem
        print(f"Making figures of {trial_name}")

        y_pos = trial["xy_filtered"] 
        all_pos.append(y_pos)

        relative_metric(velocity_list=y_pos["y"] * cfg.cm_per_pixel,
                        time=y_pos['t'],
                        ax = axs, 
                        laser_on=trial["laser_on"], 
                        color="green",
                        transparancy=0.3)

    # avg_y_pos = (
    #     pd.concat(all_pos, axis=1)
    #     .T
    #     .groupby(level=0)
    #     .mean()
    #     .T
    # )

    # relative_metric(velocity_list=avg_y_pos["y"],
    #                 time = avg_y_pos["t"],
    #                 ax = axs, 
    #                 show_pad_off=True,
    #                 laser_on=True,
    #                 color="blue",
    #                 transparancy=0)

    
    return axs




def plot_stacked_trajectories(cfg, metrics) : 
    from rats_kinematics_utils.plot import plot_single_bodypart_trajectories
    from rats_kinematics_utils.pipeline_maker import check_trial_success

    all_coords = []
    fig, ax = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        if not check_trial_success(trial) : 
            continue

        trial_name = trial['filename_clips'].stem
        print(f"\nMaking figures of {trial_name}")

        coords = trial["xy_filtered"] 

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
    #         rat_background=False, # display a rat in the background
    #     )

    return ax






















############################################################################# violin ################################################""



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



def plot_violin_distribution_velocity(cfg, velocity_list) : 
    def _pack(values, condition, intensity):
        return pd.DataFrame({
            "Condition": condition,
            "LaserIntensity": intensity,
            "Value": values.dropna().values
        })
    # default
    beta_OFF_1_velo = velocity_list[0]
    conti_OFF_1_velo = velocity_list[4]

    beta_ON_1_velo = velocity_list[2]
    conti_ON_1_velo = velocity_list[6]

    # greater
    beta_OFF_2_velo = velocity_list[1]
    conti_OFF_2_velo = velocity_list[5]

    beta_ON_2_velo = velocity_list[3]
    conti_ON_2_velo = velocity_list[7]


    df_long = pd.concat([
        _pack(beta_OFF_1_velo, "Beta-OFF", "conti=0.5, beta=1"),
        _pack(conti_OFF_1_velo, "Conti-OFF", "conti=0.5, beta=1"),
        _pack(beta_ON_1_velo, "Beta", "conti=0.5, beta=1"),
        _pack(conti_ON_1_velo, "Conti", "conti=0.5, beta=1"),

        _pack(beta_OFF_2_velo, "Beta-OFF", "conti=0.75, beta=2.5"),
        _pack(conti_OFF_2_velo, "Conti-OFF", "conti=0.75, beta=2.5"),
        _pack(beta_ON_2_velo, "Beta", "conti=0.75, beta=2.5"),
        _pack(conti_ON_2_velo, "Conti", "conti=0.75, beta=2.5"),
    ])

    # df_long = pd.concat([
    #     _pack(beta_OFF_1_velo, "OFF", "conti=0.5, beta=1"),
    #     _pack(conti_OFF_1_velo, "OFF", "conti=0.5, beta=1"),
    #     _pack(beta_ON_1_velo, "Beta", "conti=0.5, beta=1"),
    #     _pack(conti_ON_1_velo, "Conti", "conti=0.5, beta=1"),

    #     _pack(beta_OFF_2_velo, "OFF", "conti=0.75, beta=2.5"),
    #     _pack(conti_OFF_2_velo, "OFF", "conti=0.75, beta=2.5"),
    #     _pack(beta_ON_2_velo, "Beta", "conti=0.75, beta=2.5"),
    #     _pack(conti_ON_2_velo, "Conti", "conti=0.75, beta=2.5"),
    # ])


    df_long_trimed = trim_extremes_iqr(df_long, k=1.5)
    print(f"\nNumber of removed outliers : {len(df_long) - len(df_long_trimed)}")
    print(df_long_trimed)

    # plotting
    fig, ax = plt.subplots()


    sns.violinplot(
        x="Condition",
        y="Value",
        hue="LaserIntensity",
        velocity_list=df_long_trimed,
        ax=ax,
        split=True,
        gap= .1,
        inner="quart",
        order=["Conti-OFF", "Beta-OFF", "Conti", "Beta"],
        palette={"conti=0.5, beta=1": "lightblue", "conti=0.75, beta=2.5": "salmon"},
    )

    # sns.violinplot(
    #     x="Condition",
    #     y="Value",
    #     hue="LaserIntensity",
    #     velocity_list=df_long_trimed,
    #     ax=ax,
    #     split=True,
    #     gap= .1,
    #     inner="quart",
    #     order=["OFF", "Conti", "Beta"],
    #     palette={"conti=0.5, beta=1": "lightblue", "conti=0.75, beta=2.5": "salmon"},
    # )

    sns.stripplot(
        x="Condition",
        y="Value",
        hue="LaserIntensity",
        velocity_list=df_long_trimed,
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
        x = ["Conti-OFF", "Beta-OFF", "Conti", "Beta"].index(row["Condition"])
        # x = ["OFF", "Conti", "Beta"].index(row["Condition"])
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

    return ax







#####################




def prepare_df(velocity_series, date_series):

    df = pd.DataFrame({
        "date": date_series.values,
        "velocity": velocity_series.values
    }).sort_values("date")

    return df


def plot_one_setting(ax, title, beta_OFF, conti_OFF, beta_ON, conti_ON):


    # combine OFF velocities and dates
    # vel_off = pd.concat([beta_OFF[0], conti_OFF[0]])
    # date_off = pd.concat([beta_OFF[1], conti_OFF[1]])

    # # create a single DataFrame with a 'Condition' column
    # df_off = prepare_df(vel_off, date_off)
    # df_off["Condition"] = "OFF"

    df_beta_OFF = prepare_df(beta_OFF[0], beta_OFF[1])
    df_beta_OFF["Condition"] = "beta-off"

    df_conti_OFF = prepare_df(conti_OFF[0], conti_OFF[1])
    df_conti_OFF["Condition"] = "conti-off"

    df_beta_ON = prepare_df(beta_ON[0], beta_ON[1])
    df_beta_ON["Condition"] = "beta"

    df_conti_ON = prepare_df(conti_ON[0], conti_ON[1])
    df_conti_ON["Condition"] = "conti"

    df_all = pd.concat([df_conti_OFF, df_beta_OFF, df_beta_ON, df_conti_ON])
    df_all["date"] = pd.to_datetime(df_all["date"])

    # seaborn stripplot
    sns.stripplot(
        x="date",
        y="velocity",
        hue="Condition",
        data=df_all,
        jitter=0.2,
        size=4,
        alpha=0.7,
        ax=ax
    )

    # for condition, color in zip(["OFF", "beta ON", "conti ON"], ["gray", "green", "orange"]):
    #     df_subset = df_all[df_all["Condition"] == condition]
    #     df_subset["date_num"] = df_subset["date"].map(datetime.toordinal)
        
    #     sns.regplot(
    #         x="date",
    #         y="velocity",
    #         data=df_subset,
    #         scatter=False,
    #         ax=ax,
    #         line_kws={"color": color, "lw": 2, "alpha": 0.7}
    #     )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Velocity (cm.s$^{-1}$)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Condition")
    ax.tick_params(axis='x', rotation=45)


def plot_velocity_over_sessiontime(cfg, velocity_list, date_list):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # default
    beta_OFF_1_velo = (velocity_list[0], date_list[0])
    conti_OFF_1_velo = (velocity_list[4], date_list[4])

    beta_ON_1_velo = (velocity_list[2], date_list[2])
    conti_ON_1_velo = (velocity_list[6], date_list[6])

    # greater
    beta_OFF_2_velo = (velocity_list[1], date_list[1])
    conti_OFF_2_velo = (velocity_list[5], date_list[5])

    beta_ON_2_velo =(velocity_list[3], date_list[3])
    conti_ON_2_velo = (velocity_list[7], date_list[7])

    # DEFAULT
    plot_one_setting(
        axs[0],
        "Settings : beta=1mW, conti=0.5mW",
        beta_OFF_1_velo,
        conti_OFF_1_velo,
        beta_ON_1_velo,
        conti_ON_1_velo,
    )

    # GREATER
    plot_one_setting(
        axs[1],
        "Settings : beta=2.5mW, conti=0.75mW",
        beta_OFF_2_velo,
        conti_OFF_2_velo,
        beta_ON_2_velo,
        conti_ON_2_velo,
    )
    
    return fig





def plot_velocity_over_cliptime(data) : 

    nb_day = len(data["date"].unique())
    fig, axs = plt.subplots(1, nb_day, figsize=(15, 5), sharey=True)
    
    for i, d in enumerate(data["date"].unique()) : 

        df = data[data["date"] == d]
        print(len(df))

        sns.lineplot(
            x="clip",
            y="velocity",
            hue="condition",
            data=df,
            # jitter=0.2,
            # size=4,
            alpha=0.7,
            ax=axs[i],
            style="condition",
            hue_order=["Conti_Off", 'Beta_Off', "Conti_On", "Beta_On"],
            markers=True
        )
            
        axs[i].set_title(d.date())
        axs[i].set_xlabel("Trial order")
        axs[i].set_ylabel("Velocity (cm.s$^{-1}$)")
        axs[i].grid(axis="y", alpha=0.3)
        axs[i].legend(title="Condition")
        axs[i].tick_params(axis='x', rotation=45)


    return fig