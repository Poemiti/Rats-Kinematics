#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re

import rats_kinematics_utils.plot_comparative as plot
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, _plot_violin_statistic
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, make_output_path, check_analysis_choice, print_analysis_info, print_interRat_analysis_info, dataframe_report
from rats_kinematics_utils.statistics import compute_statistics, save_stat_results, LMM, compute_permutation_effect_size


# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()
print_analysis_info(cfg, "Making inter-rats figures")


root = cfg.paths.metrics
pattern = re.compile(r"#\d{3}")

filenames = sorted(
    p for p in root.glob("#*/")      # only first-level folders
    if pattern.fullmatch(p.name)
    for p in p.glob("*.joblib")      # only files directly inside
)

available_functions = {name: func
                for name, func in inspect.getmembers(plot, inspect.isfunction)
                if name.startswith("plot")}

check_analysis_choice(filenames, available_functions)

contra_hemi = {
    "#517" : "RightHemi",
    "#521" : "LeftHemi",
    "#525" : "RightHemi"
}

contra_filenames = []
for f in filenames :
    rat = f.parent.name
    file = f.stem

    if contra_hemi[rat] in file : 
        contra_filenames.append(f)

print_interRat_analysis_info(contra_filenames, available_functions)


# ------------------------------ util function -------------------------------------------



def _preprocess(filenames, METRIC: str, split_condition: bool = False) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial["trial_success"] : 
                continue

            name = trial["filename_clips"].stem
            rat = name[4:8]
            # condition = trial["condition"]
            if split_condition : 
                condition, laser_state = trial["condition"].split("_")
            else : 
                condition = trial["condition"]
            reward = "yes" if trial["reward"] else "no"

            
            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            df = pd.DataFrame({
                "value": [trial[METRIC]],
                "rat": [rat],
                "condition": [condition],
                "laser_state": [laser_state if split_condition else None],  
                "laser_intensity": [laser_intensity],
                "reward" : [reward]
            })
            data = pd.concat([data, df])

    return data.sort_values(by="condition")



# --------------------------------------- plotting -------------------------------------------

def plot_statistics(metric) : 
    data = _preprocess(contra_filenames, metric)
    print(data.iloc[:5])
    stats_res = compute_statistics(data, "value ~ condition * laser_intensity")
    save_stat_results(stats_res, cfg.paths.metrics / "statistics" / "inter_rat" / f"{metric}.joblib")

    if "mann_whitney" in stats_res.keys() :
        pairwise_results = stats_res["mann_whitney"] 
        significant_pair = pairwise_results[pairwise_results["p_value"] < 0.05]

        if len(significant_pair) > 0 : 
            fig = _plot_violin_statistic(cfg, data, significant_pair, strip=True)
            fig.suptitle(f"{metric} distribution across all trials of rat :\n{[r for r in contra_hemi.keys()]}")
            fig.savefig(make_output_path(cfg.paths.figures / "inter_rat", f"violin_{metric}.png"))

            if SHOW : 
                    plt.show()
            plt.close()

    else : 
         print("Not significant, stop!")



# --------------------------------------- main -------------------------------------------


plot_statistics("average_velocity")
plot_statistics("tortuosity")


# data = _preprocess(filenames, "average_velocity", split_condition=True)

# result = LMM(data, "value ~ condition * laser_state * laser_intensity")
# print(result.summary())


# results = dataframe_report(data)

# for col, info in results.items():
#     print(f"\nColumn: {col}")
#     print(info['summary'])


# low_data = data[data["laser_intensity"] == "low"]
# high_data = data[data["laser_intensity"] == "high"]

# print("="*60)
# print("\nSize effect of LOW laser intensity")
# low_res = compute_permutation_effect_size(low_data)


# print("="*60)
# print("\nSize effect of HIGH laser intensity")
# high_res = compute_permutation_effect_size(high_data)
