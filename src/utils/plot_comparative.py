import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ==================================== Plots for comparative analysis ===========================================


def relative_metric(data: pd.Series, 
                     time: pd.Series,
                     ax: plt.axes,
                     color: str,
                     laser_on: bool = False,
                     show_pad_off: bool = False,
                     transparancy: float=0.7) -> plt.axes : 
    fps = 125

    if ax is None:
        fig, ax = plt.subplots()

    # relative time
    relative_time = time - time[0]

    ax.plot(relative_time, data, color= color, alpha=transparancy)
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

    all_velocity = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        # print(f"\n[{t}/{len(metrics)}]")

        if trial["trial_success"] : 
            trial_name = trial['filename_clips'].stem
            # print(f"Making figures of {trial_name}")

            velo = trial["instant_velocity"]
            all_velocity.append(velo)

            relative_metric(data=velo["velocity"],
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

    relative_metric(data=avg_velocity["velocity"],
                    time = avg_velocity["t"],
                    ax = axs, 
                    show_pad_off=True,
                    laser_on=True,
                    color="blue",
                    transparancy=1)

    
    return axs



def plot_stacked_Yposition(cfg, metrics: dict) :

    all_pos = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        # print(f"\n[{t}/{len(metrics)}]")

        if trial["trial_success"] : 
            trial_name = trial['filename_clips'].stem
            # print(f"Making figures of {trial_name}")

            y_pos = trial["xy_filtered"] 
            all_pos.append(y_pos)

            relative_metric(data=y_pos["y"] * cfg.cm_per_pixel,
                                time=y_pos['t'],
                                ax = axs, 
                                laser_on=False, 
                                color="green",
                                transparancy=0.3)

    # avg_y_pos = (
    #     pd.concat(all_pos, axis=1)
    #     .T
    #     .groupby(level=0)
    #     .mean()
    #     .T
    # )

    # relative_metric(data=avg_y_pos["y"],
    #                 time = avg_y_pos["t"],
    #                 ax = axs, 
    #                 show_pad_off=True,
    #                 laser_on=True,
    #                 color="blue",
    #                 transparancy=1)

    
    return axs





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



def plot_violin_distribution_velocity(cfg, data) : 
    def _pack(values, condition, intensity):
        return pd.DataFrame({
            "Condition": condition,
            "LaserIntensity": intensity,
            "Value": values.dropna().values
        })
    # default
    beta_OFF_1_velo = data[0]
    conti_OFF_1_velo = data[4]

    beta_ON_1_velo = data[2]
    conti_ON_1_velo = data[6]

    # greater
    beta_OFF_2_velo = data[1]
    conti_OFF_2_velo = data[5]

    beta_ON_2_velo = data[3]
    conti_ON_2_velo = data[7]


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

    return ax

