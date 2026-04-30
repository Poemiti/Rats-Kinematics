#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re, yaml
import seaborn as sns
import matplotlib.colors as mcolors

import rats_kinematics_utils.analysis.plot_comparative as plot_comparative
from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.analysis.plot_comparative import _plot_violin_statistic
from rats_kinematics_utils.core.file_utils import load_trial_data, make_output_path, check_analysis_choice, print_analysis_info, dataframe_report, parse_filename
from rats_kinematics_utils.analysis.statistics import compute_statistics, save_stat_results, LMM, compute_permutation_effect_size, transform_data


# ==================================== display hyperparameter ===========================================


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("paper", style="ticks", rc=custom_params, palette="pastel")


def rat_colorpalette(data, rat_only: bool = False) : 
    # define colors of each rats
    rats = data["rat"].unique()
    rats = sorted(rats)
    base_colors = sns.color_palette("tab10", len(rats))

    def lighten(color, amount=0.45):
        c = mcolors.to_rgb(color)
        return tuple(1 - amount*(1 - x) for x in c)

    pal = {}

    if rat_only : 
        for rat, base in zip(rats, base_colors):
            pal[f"{rat}"] = base
    else :
        for rat, base in zip(rats, base_colors):
            pal[f"{rat}_LaserOn"] = base
            pal[f"{rat}_LaserOff"] = lighten(base)

    return pal

dash = {
    "LaserOff" : (2,1),
    "LaserOn" : ""
}

laser_color = "royalblue"


# ==================================== Plots for inter analysis ===========================================




def filter_contra_trials(cfg, single_rat: bool = False): 

    if single_rat: 
        file_to_process = []
    else : 
        file_to_process = {}

    print(cfg.inter_rat_metrics_paths)
    for rat, root_path in cfg.inter_rat_metrics_paths.items() : 
        print()
        print(rat)
        if rat not in file_to_process.keys(): 
            file_to_process[rat] = []

        filenames = list(root_path.glob("*.joblib"))

        for file in filenames: 

            meta = parse_filename(file.stem)

            view = "Left" if meta["view"]=="H001" else "Right"
            hemi = meta["stim_location"]
            l_intensity = meta["laser_intensity"]
            condition = meta["condition"]

            if l_intensity not in cfg.laser_intensities[condition]: 
                print("  ", file.stem, "-NOT RIGHT LASER INTENSITY")
                continue

            if cfg.inter_rat.contra_hemi[rat] != hemi: 
                print("  ", file.stem, "-NOT CONTRA")
                continue

            if view in hemi :       # view must be contra lateral to the camera view
                print("  ", file.stem, "-NOT RIGHT VIEW")
                continue
            
            print("  ", file.stem)
            file_to_process[rat].append(file)

    return file_to_process



def _preprocess(cfg, filenames: list[Path], METRIC: str, split_condition: bool = False, 
                merge_laserOff: bool = False) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for trial in metrics : 
            if not trial[cfg.bodypart]["trial_success"] : 
                continue
            
            condition = trial["condition"]
            laser_state = trial["laser_state"]
            name = trial["filename_clips"].stem
            rat = name[4:8]

            if merge_laserOff and not split_condition: 
                if laser_state == "LaserOff" : 
                    condition = "LaserOff"
                else : 
                    condition = condition

            elif not merge_laserOff and not split_condition : 
                condition = condition + "_" + laser_state

            else : 
                raise ValueError(f"ERROR : impossible combinaison of split condition: {split_condition}, and merge laserOff: {merge_laserOff}")

            reward = "yes" if trial["reward"] else "no"

            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            value = trial[cfg.bodypart][METRIC]

            df = pd.DataFrame({
                "value": [value],
                "rat": [rat],
                "condition": [condition],
                "laser_state": [laser_state if split_condition else None],  
                "laser_intensity": [laser_intensity],
                "reward" : [reward]
            })
            data = pd.concat([data, df])

    return data.sort_values(by="condition").reset_index(drop=True)


# --------------------------------------- plotting -------------------------------------------


def inter_rat_metadata_report(cfg, rat_types: list[str], joblib_filenames: list[Path]) : 
    from rats_kinematics_utils.preprocessing.plot_preprocess import _plot_metadata_report

    output_dir = cfg.paths.inter_rat

    noCue_video = pd.DataFrame()
    report = pd.DataFrame()

    for r_type in rat_types : 

        for f in joblib_filenames:
                
            trials = load_trial_data(f)

            for t in trials: 

                if t['rat_type'] != r_type: 
                    continue

                view_num = "H001" if t["camera_view"]=="left" else "H002"

                df = pd.DataFrame({
                        "rat_name": [t["rat_name"]],
                        "condition": [t["condition"]],
                        "view": [t["camera_view"] + f" ({view_num})"],
                        "stim": [t["stim_location"]],
                        "cue": [t["cue_type"]],
                        "laser": [t["laser_state"]],
                        "intensity": [t["laser_intensity"]],
                        "date": [t["date"]]

                    }
                )
                report = pd.concat([report, df], ignore_index=True)

                if t["cue_type"] == "NoCue" : 
                    df = pd.DataFrame({
                        "rat type": [t["rat_type"]],
                        "pad off": [t["pad_off"]],
                        "laser state": [t["laser_state"]],
                        "date": [t["date"]],
                        "clip path": [t["filename_clips"]],
                    })
                    noCue_video = pd.concat([noCue_video, df], ignore_index=True)

        print("Number of trials :", len(report))
            
        if len(report) == 0  : 
            print(f"\”No data for this rat type : {r_type}")
            break

        # plot reports
        # _plot_metadata_report(report, output_dir / f"{r_type}_trial_metadata_report", 
        #                       rat_type=r_type)
        
        # _plot_metadata_report(report, output_dir / f"{r_type}_rat_individual_proportion", 
        #                       groups=["condition", "rat_name"], 
        #                       rat_type=r_type)
        
        # _plot_metadata_report(report, output_dir / f"{r_type}_rat_proportion", 
        #                       groups=["rat_name", "condition", "laser"], 
        #                       rat_type=r_type)

        _plot_metadata_report(report, output_dir / f"{r_type}_rat_date", 
                              groups=["date", "rat_name"], 
                              rat_type=r_type)

        noCue_filename = output_dir / f"{r_type}_NoCue_video.csv"
        noCue_video.to_csv(noCue_filename)

        print(f"\nNumber of NoCue videos : {len(noCue_video)}")
        print(f"No Cue video are stored in {noCue_filename}")




def plot_likelihood_distri_interrat(cfg, joblib_filenames: list[Path]) : 
    from rats_kinematics_utils.preprocessing.preprocess import open_DLC_results
    from tqdm import tqdm

    likelihood_distri = []
    n=0

    for f in tqdm(joblib_filenames, "loading bodypart likelihood"):
        
        trials_list = load_trial_data(f)
        for trial in trials_list: 

            # loading
            coords_path = Path(trial['filename_coords'])
            raw_coords = open_DLC_results(coords_path)
            bodyparts = raw_coords.columns.get_level_values(0).unique()

            for bp in bodyparts[1:]:
                likelihoods = raw_coords[bp]["likelihood"]

                for val in likelihoods:
                    likelihood_distri.append({
                        "bodypart": bp,
                        "likelihood": val
                    })
            n+=1

    data = pd.DataFrame(likelihood_distri)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    sns.violinplot(
        data=data,
        x="bodypart",
        y="likelihood",
        inner="quart",
        ax=ax
    )

    ax.axhline(cfg.threshold, linestyle="--", color="red", label="likelihood threshold", lw="0.8")

    ax.set_title(f"Distribution of likelihood across bodyparts of all rats\nlikelihood, threshold={cfg.threshold:.2f}\nNumber of trials: {n}")
    ax.set_xlabel("Bodyparts")
    ax.set_ylabel("Likelihood")
    ax.legend(loc="lower right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.inter_rat, f"bodypart_likelihood_distribution.png"))
    plt.show()
    plt.close()



def get_rat_proportion(rat_filenames: dict[str, Path]): 
    import joblib

    rat_proportion = {}
    n = 0

    for rat, joblib_filenames in rat_filenames.items() : 
        rat_proportion[rat] = 0

        for file in joblib_filenames : 
            trials = joblib.load(file)
            n += len(trials)

            rat_proportion[rat] += len(trials)

    
    print(n)
    return rat_proportion






def plot_statistics(cfg, filenames: list[Path], metric: str, comparisons: list[tuple[str, str]], 
                    merge_laserOff: bool = False, per_rats: bool = False) -> None: 
    
    data = _preprocess(cfg, filenames, metric, split_condition=False, # has to stay false
                       merge_laserOff=merge_laserOff) 

    if merge_laserOff : 
        merged = "_laserOff_merged" 
        order = ["LaserOff", "Beta", "Conti"]
    else : 
        merged = "" 
        order = ["Conti_LaserOff", "Beta_LaserOff", "Conti_LaserOn", "Beta_LaserOn"]

    if per_rats: 
        per_rats = "_per_rats"
        fig = plot_violin_per_rat(cfg, data, strip=False, order=order)
        fig.subplots_adjust(top=0.88)
        fig.suptitle(f"{metric} distribution across all trials of rat :\n{cfg.inter_rat.rats}")
        fig.savefig(make_output_path(cfg.paths.inter_rat / "analysis_distribution", f"violin_{metric}{merged}{per_rats}.png"))

    else : 
        per_rats = ""

        stats_res = compute_statistics(data, comparisons)
        save_stat_results(stats_res, cfg.paths.inter_rat / "metric" / f"{metric}.joblib")

        if "mann_whitney" in stats_res.keys() :
            pairwise_results = stats_res["mann_whitney"] 
            significant_pair = pairwise_results[pairwise_results["p_value"] < 0.05]

            if len(significant_pair) > 0 : 
                fig = _plot_violin_statistic(cfg, data, significant_pair, strip=True, 
                                            order=order)
                fig.suptitle(f"{metric} distribution across all trials of rat :\n{cfg.inter_rat.rats}")
                fig.savefig(make_output_path(cfg.paths.inter_rat / "analysis_distribution", f"violin_{metric}{merged}{per_rats}.png"))

                plt.show()
                plt.close()

        else : 
            print("stop!")


def _displot_stat(perm_data) : 

    rows = []
    for cond in perm_data:
        diffs = np.array(cond["permutation differences"])

        # Mirror the permutation distribution
        mirrored_diffs = np.concatenate([-diffs, diffs])
        
        for diff in mirrored_diffs:
            rows.append({
                "Condition": cond["Condition"],
                "permutation difference": diff
            })
    df = pd.DataFrame(rows)

    order = ["Beta vs Conti",
             "Conti vs NOstim", 
             "Beta vs NOstim", ]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(
        data=df,
        x="permutation difference",
        hue="Condition",
        ax=ax,
        hue_order=order,
        common_norm=False,  # keeps separate densities normalized
        alpha=0.5,
        kde=True
    )

    kde_lines = [line for line in ax.lines]
    print(f"n kde lines {len(kde_lines)}")

    for i, line in enumerate(kde_lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        
        # Find peak of this KDE
        max_idx = np.argmax(y_data)
        x_peak = x_data[max_idx]
        y_peak = y_data[max_idx]
        
        ax.text(
            perm_data[i]['observed mean difference'],
            y_peak,
            f"          {perm_data[i]['observed mean difference']:.2f}",
            color=line.get_color(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.axvline(perm_data[i]["observed mean difference"], 
                   color=line.get_color(), 
                   lw=1, ls='--', label="conti observed")


    return ax



def plot_permutation(cfg, filenames: list[Path], metric: str, intensity: str, n_perm: int) : 
    raw_data = _preprocess(cfg, filenames, metric, split_condition=True)  # has to stay true

    data = raw_data[raw_data["laser_intensity"] == intensity]
    res = compute_permutation_effect_size(data, n_perm)
    ax = _displot_stat(res)
    ax.set_title(f"Effect size Distributions of {metric} (n_perm={n_perm})\n({intensity} intensity)")
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("Density")

    fig = ax.figure
    fig.savefig(make_output_path(cfg.paths.inter_rat / "analysis_permutation", f"NostimePerCondition_notransform_{metric}_{intensity}Intensities_{n_perm}.png"))
    plt.show()
    plt.close()









def _preprocess_tendency(cfg, filenames: list[Path], metric: str, metric_vector: str = None) : 
    from tqdm import tqdm
    from rats_kinematics_utils.preprocessing.preprocess import crop_xy

    if metric_vector is None: 
        metric_vector = metric

    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for j, trial in tqdm(enumerate(metrics)) : 

            if not trial[cfg.bodypart]["trial_success"] or \
                trial["laser_intensity"] == "NOstim": 
                continue

            condition = trial["condition"]
            laser_state = trial["laser_state"]
            pad_off = trial["pad_off"]
            value = trial[cfg.bodypart][metric]
            rat = trial["rat_name"]

            # crop around the pad off
            val = crop_xy(value, 
                           start=pad_off - 0.1, 
                           end=pad_off + 0.4)
            # val = value
            relative_time = val["t"] - pad_off


            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            else : laser_intensity = "high"


            df = pd.DataFrame({
                "t": round(relative_time, 2),
                "rat": rat,
                "rat_laser": rat + "_" + laser_state,
                "value": val[metric_vector],
                "condition": condition,
                "laser_state": laser_state,
                "laser_intensity": laser_intensity,
                "id": f"{i}_{j}",
            })

            data = pd.concat([data, df], ignore_index=True)

    data.to_csv(make_output_path(cfg.paths.inter_rat / f"tendency", f"{metric}_data.csv"))

    return data



def plot_tendency_per_rat(cfg, data, error_function: str) :
    from scipy.stats import sem
    
    if error_function is None: 
        error_function = "pi" 

    pal = rat_colorpalette(data)

    g = sns.relplot(
        kind="line",
        data=data,
        col="condition",
        row="laser_intensity",
        x="t",
        y="value",
        hue="rat_laser",
        palette=pal,
        style="laser_state",
        dashes=dash,
        row_order=["low", "high"],
        estimator="mean" ,
        errorbar=("pi", 50) if error_function == "pi" else None,   
        # percentile interval (non parametric) https://seaborn.pydata.org/tutorial/error_bars.html
        # drawstyle="steps-pre"
    )


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

            # Manual SEM shading
            if error_function == "sem":

                # subset data for this facet
                facet_df = data[
                    (data["condition"] == condition) &
                    (data["laser_intensity"] == laser_intensity)
                ]

                # do SEM separately for each hue group
                for (rat, laser_state), sub in facet_df.groupby(["rat", "laser_state"]):

                    grouped = (
                        sub.groupby("t")["value"]
                        .agg(
                            mean="mean",
                            sem=lambda x: sem(x, nan_policy="omit")
                        )
                        .reset_index()
                        .sort_values("t")
                    )

                    lower = grouped["mean"] - grouped["sem"]
                    upper = grouped["mean"] + grouped["sem"]

                    color = pal[f"{rat}_{laser_state}"]

                    ax.fill_between(
                        grouped["t"].values,
                        lower.values,
                        upper.values,
                        color=color,
                        alpha=0.20
                    )

    sns.move_legend(g,
                    borderpad=1,
                    loc='upper right', 
                    facecolor="lightgray")

    return g



def plot_violin_per_rat(cfg, data: pd.DataFrame, strip: bool = True,
                           order: list[str] = ["Conti_LaserOff", "Beta_LaserOff", "Conti_LaserOn", "Beta_LaserOn"]) : 
    from rats_kinematics_utils.analysis.plot_comparative import _trim_extremes_iqr

    pal = rat_colorpalette(data, rat_only=True)

    print(len(data))
    data_trimmed = _trim_extremes_iqr(data, k=1.5)
    print(f"\nNumber of removed outliers : {len(data) - len(data_trimmed)}")

    if data_trimmed["condition"].str.contains("NOstim").any() :
        data_trimmed = data_trimmed.loc[~data_trimmed["condition"].str.contains("NOstim")]
        data = data.loc[~data["condition"].str.contains("NOstim")]

    g = sns.catplot(
        kind="violin",
        data=data_trimmed,
        row="laser_intensity",
        row_order=["low", "high"],
        x="condition",
        y="value",
        hue="rat",
        order=order,
        hue_order=cfg.inter_rat.rats,
        palette=pal,
        legend=True,
        dodge=True,
        height=4,
        aspect=1.6,
        inner="quartile",   
        linewidth=1,
    )

    if strip:
        for i, intensity in enumerate(["low", "high"]):
            ax = g.axes[i, 0]

            sub = data_trimmed[data_trimmed["laser_intensity"] == intensity]

            sns.stripplot(
                data=sub,
                x="condition",
                y="value",
                hue="rat",
                order=order,
                hue_order=cfg.inter_rat.rats,
                palette=pal,
                marker="X",
                size=3,
                alpha=0.7,
                dodge=True,
                ax=ax,
                legend=False,
            )


    count = (
        data.groupby(["laser_intensity", "condition", "rat"])
        .size()
        .reset_index(name="N")
    )

    x_positions = {cond: i for i, cond in enumerate(order)}
    rats = list(cfg.inter_rat.rats)

    total_width = 0.8
    step = total_width / len(rats)

    for i, intensity in enumerate(["low", "high"]):

        ax = g.axes[i, 0]

        sub_count = count[count["laser_intensity"] == intensity]

        ymin, ymax = ax.get_ylim()
        y_text = ymin + 0.02 * (ymax - ymin)

        for _, row in sub_count.iterrows():

            cond = row["condition"]
            rat = row["rat"]
            N = row["N"]

            if cond not in x_positions or rat not in rats:
                continue

            x = x_positions[cond]
            j = rats.index(rat)

            x_shifted = x - total_width / 2 + step / 2 + j * step

            ax.text(
                x_shifted,
                y_text,
                f"{N}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=8,
                fontweight="bold",
            )
    
    return g.figure