#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re, yaml
import seaborn as sns

import rats_kinematics_utils.analysis.plot_comparative as plot_comparative
from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.analysis.plot_comparative import _plot_violin_statistic
from rats_kinematics_utils.core.file_utils import load_trial_data, make_output_path, check_analysis_choice, print_analysis_info, dataframe_report, parse_filename
from rats_kinematics_utils.analysis.statistics import compute_statistics, save_stat_results, LMM, compute_permutation_effect_size, transform_data


def filter_contra_trials(cfg): 
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



def _preprocess(cfg, filenames: list[Path], METRIC: str, split_condition: bool = False) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_trial_data(Path(metrics_path))

        for trial in metrics : 
            if not trial[cfg.bodypart]["trial_success"] : 
                continue

            name = trial["filename_clips"].stem
            rat = name[4:8]
            
            if split_condition : 
                condition = trial["condition"]
                laser_state = trial["laser_state"]
            else : 
                condition = trial["condition"] + "_" + trial["laser_state"]
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


def plot_statistics(cfg, filenames: list[Path], metric: str, comparisons: list[tuple[str, str]]) -> None: 
    data = _preprocess(cfg, filenames, metric, split_condition=False)  # has to stay false

    stats_res = compute_statistics(data, comparisons)
    save_stat_results(stats_res, cfg.paths.inter_rat / "metric" / f"{metric}.joblib")

    if "mann_whitney" in stats_res.keys() :
        pairwise_results = stats_res["mann_whitney"] 
        significant_pair = pairwise_results[pairwise_results["p_value"] < 0.05]

        if len(significant_pair) > 0 : 
            fig = _plot_violin_statistic(cfg, data, significant_pair, strip=True)
            fig.suptitle(f"{metric} distribution across all trials of rat :\n{cfg.inter_rat.rats}")
            fig.savefig(make_output_path(cfg.paths.inter_rat / "analysis_distribution", f"violin_{metric}_dots.png"))

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