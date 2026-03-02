#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re

import rats_kinematics_utils.plot_comparative as plot
from rats_kinematics_utils.file_management import verify_exist, get_session
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, _plot_violin_statistic
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice, print_analysis_info, print_interRat_analysis_info
from rats_kinematics_utils.statistics import compute_statistics


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



def _preprocess(filenames, METRIC: str) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial["trial_success"] : 
                continue

            name = trial["filename_clips"].as_posix()
            condition = trial["condition"]
            reward = "yes" if trial["reward"] else "no"

            
            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            df = pd.DataFrame({
                "value": [trial[METRIC]],
                "condition": [condition],
                "laser_intensity": [laser_intensity],
                "reward" : [reward]
            })
            data = pd.concat([data, df])

    return data.sort_values(by="condition")



# --------------------------------------- plotting -------------------------------------------

metric= "average_velocity"
data = _preprocess(contra_filenames, metric)
print(data.iloc[:5])
significant_pair = compute_statistics(data, "value ~ condition * laser_intensity")


fig = _plot_violin_statistic(cfg, data, significant_pair, strip=True)
fig.suptitle(f"{metric} distribution across all trials of rat :\n{[r for r in contra_hemi.keys()]}")
fig.savefig(make_output_path(cfg.paths.figures / "inter_rat", f"violin_{metric}.png"))

if SHOW : 
        plt.show()
plt.close()