import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


# ==================================== display hyperparameter ===========================================


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

    ax.plot(relative_time, metric_list, color= color, alpha=transparancy, label="average" if show_pad_off else None)
    if show_pad_off :
        # add line for pad off 
        ax.axvline(0, color='k', lw=0.8, ls='--', label="pad off")
        ax.legend()

        # show laser on
        if laser_on :
            laser_on = 0 + 0.025
            laser_off = laser_on +  0.3 # sec or 37.5 frame
            ax.axvspan(laser_on, laser_off, color=LASER_COLOR, alpha=0.3, label="laser on")
            ax.legend()
        
    return ax



def plot_stacked_acceleration(cfg, metrics: dict) :
    from rats_kinematics_utils.core.file_utils import check_trial_success

    fig, axs = plt.subplots(figsize=(9, 7))

    aligned_trials = []

    for trial in metrics:
        if not check_trial_success(cfg, trial):
            continue
        laser_state = trial["laser_state"]
        trial_name = trial["filename_clips"].stem
        acc = trial[cfg.bodypart]["acceleration"]
        relative_time = acc["t"] - trial["pad_off"]

        _relative_metric(acc["acceleration"],
                         relative_time=relative_time,
                         ax=axs, 
                         color=LINE_COLOR,
                         transparancy=LINE_TRANSPARANCY)

        df = pd.DataFrame({
            "acceleration": acc["acceleration"].values
        }, index=relative_time.values)

        aligned_trials.append(df)

    common_time = np.linspace(-0.5, 1.0, 500)
    aligned_resampled = []

    for df in aligned_trials:
        interp_values = np.interp(
            common_time,
            df.index.values,
            df["acceleration"].values,
            left=np.nan,
            right=np.nan
        )
        aligned_resampled.append(interp_values)

    aligned_matrix = np.vstack(aligned_resampled)
    avg_acc = np.nanmean(aligned_matrix, axis=0)

    _relative_metric(
                    metric_list=avg_acc,
                    relative_time=common_time,
                    ax=axs,
                    show_pad_off=True,
                    laser_on="On" in laser_state,
                    color=AVG_LINE_COLOR,
                    transparancy=1
                )

    
    return axs


def plot_stacked_velocity(cfg, metrics: dict) :
    from rats_kinematics_utils.core.file_utils import check_trial_success

    fig, axs = plt.subplots(figsize=(9, 7))

    aligned_trials = []

    for trial in metrics:
        if not check_trial_success(cfg, trial):
            continue
        laser_state = trial["laser_state"]
        trial_name = trial["filename_clips"].stem
        velo = trial[cfg.bodypart]["instant_velocity"]
        relative_time = velo["t"] - trial["pad_off"]

        _relative_metric(velo["velocity"],
                         relative_time=relative_time,
                         ax=axs, 
                         color=LINE_COLOR,
                         transparancy=LINE_TRANSPARANCY)

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
                    laser_on="On" in laser_state,
                    color=AVG_LINE_COLOR,
                    transparancy=1
                )

    
    return axs



def plot_stacked_Yposition(cfg, metrics: dict) :
    from rats_kinematics_utils.core.file_utils import check_trial_success

    aligned_pos = []
    fig, axs = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        # print(f"\n[{t}/{len(metrics)}]")

        if not check_trial_success(cfg, trial) : 
            continue 

        laser_state = trial["laser_state"]
        trial_name = trial["filename_clips"].stem
        y_pos = trial[cfg.bodypart]["xy_raw"]
        relative_time = y_pos["t"] - trial["pad_off"]

        _relative_metric(metric_list=y_pos["y"] * cfg.cm_per_pixel,
                        relative_time=relative_time,
                        ax = axs, 
                        laser_on=False, 
                        color=LINE_COLOR,
                        transparancy=LINE_TRANSPARANCY)
        
        df = pd.DataFrame({"y_pos": (y_pos["y"] * cfg.cm_per_pixel).values}, index=relative_time.values)

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
                    laser_on="On" in laser_state,
                    color=AVG_LINE_COLOR,
                    transparancy=1)

    return axs




def plot_stacked_trajectories(cfg, metrics, ax: plt.axes = None) : 
    from rats_kinematics_utils.analysis.plot import plot_single_bodypart_trajectories
    from rats_kinematics_utils.core.file_utils import check_trial_success

    all_coords = []
    if not ax :
        fig, ax = plt.subplots(figsize=(9, 7))


    for t, trial in enumerate(metrics) : 

        if not check_trial_success(cfg, trial) : 
            continue

        trial_name = trial['filename_clips'].stem
        print(f"\nMaking figures of {trial_name}")

        coords = trial[cfg.bodypart]["xy_pad_off"] 

        if trial_name == "Rat_#525Ambidexter_20240527_ContiMT300_RightHemiCHR_L1L25050_C001H001S0001_NO_TRUST_DATA_clip_55" : 
            continue

        if trial["laser_state"] == "LaserOn":
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
                color=LINE_COLOR,
                transparancy=LINE_TRANSPARANCY,
            )
        all_coords.append(coords)

    avg_coords = (pd.concat(all_coords, axis=1)
                    .T
                    .groupby(level=0)
                    .mean()
                    .T)

    plot_single_bodypart_trajectories(
            coords=avg_coords,
            cm_per_pixel=cfg.cm_per_pixel,
            frame_laser_on=None,
            ax=ax,
            color=AVG_LINE_COLOR,
            transparancy=1,
        )

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
        .apply(_trim, include_groups=True)
        .reset_index(drop=True)
    )


def plot_violin_distribution_velocity() : 
    pass

def plot_violin_distribution_tortuosity() : 
    pass

def plot_violin_distribution_peak() : 
    pass




######################################## statistics violin ######################################



def parse_group(g):
    cond, intensity = g.split(".")
    return (cond, intensity)

def _add_stat_annotations(ax, data, statistics, order):

    # Filter stats for this facet (adapt if needed)
    stats_subset = statistics.copy()
    l_intensities = data["laser_intensity"].unique()

    # Build pairs list
    pairs = [
        (parse_group(row["group1"]),
        parse_group(row["group2"]))
        for _, row in stats_subset.iterrows()
    ]

    if len(pairs) == 0:
        return

    annotator = Annotator(
            ax,
            pairs,
            data=data,
            x="condition",
            y="value",
            hue="laser_intensity",
            order=order,
            hue_order=HUE_ORDER if len(l_intensities) > 1 else None
        )

    # We already computed p-values → no test
    annotator.configure(
        test=None,
        text_format="star",   # or "simple"
        loc="inside",
        verbose=0
    )

    # Use adjusted p-values
    pvalues = stats_subset["p_value"].values

    annotator.set_pvalues(pvalues)
    annotator.annotate()



def _plot_violin_statistic(cfg, data: pd.DataFrame, statistics: pd.DataFrame = None, strip: bool = True) : 

    print(len(data))
    data_trimmed = _trim_extremes_iqr(data, k=1.5)
    print(f"\nNumber of removed outliers : {len(data) - len(data_trimmed)}")

    if data_trimmed["condition"].str.contains("NOstim").any() and statistics is not None:
        data_trimmed = data_trimmed.loc[~data_trimmed["condition"].str.contains("NOstim")]
        data = data.loc[~data["condition"].str.contains("NOstim")]

    order = ["Conti_LaserOff", "Beta_LaserOff", "Conti_LaserOn", "Beta_LaserOn"]

    fig, ax = plt.subplots(figsize=[10,5])

    # VIOLIN
    sns.violinplot(
        data=data_trimmed,
        x="condition",
        y="value",
        hue="laser_intensity",
        split=True,
        inner="quart",
        order=order,
        hue_order=HUE_ORDER,
        gap= .1,
        palette=LASER_INTENSITY_PALETTE,
        legend=True,
    )

    # STRIP
    if strip : 
        sns.stripplot(
            data=data_trimmed,
            x="condition",
            y="value",
            # hue="reward",
            # palette=REWARD_PALETTE,
            hue="laser_intensity",
            palette=LASER_INTENSITY_PALETTE_DARK,
            hue_order=HUE_ORDER,
            marker="X",
            size=3,
            alpha=0.7,
            # legend=True,
            dodge=True,  # to split like the violin
        )

    # --------------------------- display counting --------------------------

    count = (data
            .groupby(["condition", "laser_intensity"])
            .size()
            .reset_index(name="N")
            )
    
    # Get category positions
    x_positions = {cond: i for i, cond in enumerate(order)}

    # Small horizontal offset for split violins
    offset = 0.15
    ymin = data["value"].min()

    for _, row in count.iterrows():
        cond = row["condition"]
        intensity = row["laser_intensity"]
        N = row["N"]

        x = x_positions[cond]

        # Shift left/right depending on hue level
        if intensity == HUE_ORDER[0]:
            x_shifted = x - offset
        else:
            x_shifted = x + offset

        ax.text(
            x_shifted,
            ymin-0.5,
            f"{N}",
            ha="center",
            va="bottom",
            color="steelblue" if intensity=="low" else "tomato",
            fontsize=9,
            fontweight="bold"
        )

    if statistics is not None : 
        _add_stat_annotations(ax, data_trimmed, statistics, order)

    ax.legend(loc="upper right")

    return fig




def plot_violin_stat_velocity() : 
    pass

def plot_violin_stat_tortuosity() : 
    pass








########################################### displot ###############################################


def _plot_displot(data):
    from rats_kinematics_utils.analysis.statistics import transform_data

    print(f"data size before trim = {len(data)}")
    # log_val = np.log(data["value"])
    # data["value"] = log_val

    transformed_val = transform_data(data["value"])
    data["value"] = transformed_val

    # data_trimmed = data
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
    g = _plot_displot(data) 
    return g

def plot_displot_peak(data) : 
    g = _plot_displot(data) 
    return g

def plot_displot_tortuosity(data) : 
    g = _plot_displot(data) 
    return g



########################################### velocity accross trials ###############################################






def plot_velocity_over_cliptime(data) : 

    data_trimmed = _trim_extremes_iqr(data,
                                      value_col="velocity",
                                      group_cols=["condition", "laser_state", "date"],
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
        color="grey",
        estimator=None,
        alpha=0.5,
        sort=True
    )

    g.map_dataframe(
        sns.scatterplot,
        x="clip",
        y="velocity",
        hue="laser_state",
        # sort=True,
    )
            
    g.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Trial order", "Velocity (cm.s$^{-1}$)")
    g.tight_layout()


    return g












def _metric_at_padOff(data, type: str = "boxplot") :

    multiple_intensity =  len(data["laser_intensity"].unique()) > 1 
    print(multiple_intensity)
    
    g = sns.FacetGrid(
        data=data,
        row="laser_state",
        col="condition",
        margin_titles=True,
        sharex=False,
    )

    if type == "boxplot" : 
        g.map_dataframe(
            sns.boxplot,
            x="event",
            y="value",
            hue="laser_intensity" if multiple_intensity else None,
            hue_order= HUE_ORDER if multiple_intensity else None,
            palette=LASER_INTENSITY_PALETTE if multiple_intensity else None,
        )

    elif type == "violin" : 
        g.map_dataframe(
            sns.violinplot,
            x="event",
            y="value",
            hue="laser_intensity" if multiple_intensity else None,
            hue_order= HUE_ORDER if multiple_intensity else None,
            palette=LASER_INTENSITY_PALETTE if multiple_intensity else None,
            split=True if multiple_intensity else None,
            inner="quart",
            gap= .1,
        )

    # --- Add counts ---
    for (laser_state, condition), ax in g.axes_dict.items():
        subset = data[
            (data["laser_state"] == laser_state) &
            (data["condition"] == condition)
        ]

        # count per event + intensity
        counts = (
            subset
            .groupby(["event", "laser_intensity"])
            .size()
            .reset_index(name="n")
        )

        for i, event in enumerate(subset["event"].unique()):
            sub_counts = counts[counts["event"] == event]
            ymax = subset["value"].max()

            # build label (e.g. n=10 / n=12)
            label = "        ".join([f"{n}" for n in sub_counts["n"]])

            ax.text(
                i,
                ymax,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if multiple_intensity :
            g.add_legend(title="Laser intensity")

    return g


def plot_velocity_at_padOff(): 
    return