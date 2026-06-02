#!/usr/bin/env python

import joblib, time, sys
import numpy as np
from tqdm import tqdm
import pandas as pd

from rats_kinematics_utils.core.config import load_config, match_rule
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path, load_trial_data, dataframe_report, filter_contra_trials
from rats_kinematics_utils.preprocessing.preprocess import check_times, filter_outliers, filter_likelihood, interpolate_data, open_DLC_results
import rats_kinematics_utils.preprocessing.plot_preprocess as pp

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Preprocessing")

filenames = filter_contra_trials(cfg, single_rat=True)
yaml_filenames = list((cfg.paths.raw_clips).rglob("*.yaml"))

# print("\nPlotting behavior rate\n")
# pp.clip_behavior_rate(cfg, filenames)

# sys.exit()

# print("\nPlotting likelihood distribution of all bodyparts\n")
# pp.plot_likelihood_distribution(cfg, yaml_filenames)

# print("\nPlotting metadata report\n")
# pp.metadata_report(cfg, yaml_filenames)

# print("\nPlotting likelihood across frame of each trials\n")
# pp.plot_likelihood_across_frames(cfg, filenames)

print("\nPlotting the distribution of the failure reason\n")
pp.plot_trial_failure_reason(cfg, ["CHR"] ,filenames, inter_rat=False)

sys.exit()

# print("\nPlotting the distribution of the failure reason DETAIL VERSION\n")
# pp.plot_trial_failure_reason_detail(cfg, filenames)

# print("\nPlotting the distribution of clip success\n")
# pp.plot_manual_clip_success_distri(cfg, filenames)


print("\nAnnotating video to verify Behavior Boxes\n")
pp.verify_behavior_box(cfg, filenames)

sys.exit()

###############################

### verify if all the condition are right

################################


for file in filenames:
    
    exemple = load_trial_data(file)
    cue = {
        "CueL1": 0,
        "CueL2": 0,
        "NoCue": 0
    }

    hemi = {
        "LeftHemi": 0,
        "RightHemi": 0,
    }

    view = {
        "left": 0,
        "right": 0
    }

    for t in exemple: 
        observed_cue = t['cue_type']
        observed_hemi = t["stim_location"]
        observed_view = t["camera_view"]
        cue[observed_cue] += 1
        hemi[observed_hemi] += 1
        view[observed_view] += 1

    print(file.stem)
    print("  nb trial: ", len(exemple))

    print("  cue:")
    for k, v in cue.items(): 
        print("    -", k, v)

    print("  hemi:")
    for k, v in hemi.items(): 
        print("    -", k, v)

    print("  view: ")
    for k, v in view.items(): 
        print("    -", k, v)



###############################

### visualise all data for each rat

################################

all_rats_database = pd.read_csv(cfg.paths.database / "all_rats.csv")
all_rats_database = all_rats_database.drop(columns=["Unnamed: 0"], errors="ignore")

print(all_rats_database.columns)

report = dataframe_report(all_rats_database, sort=True)

for col, info in report.items():
    print(f"\nColumn: {col}")
    print(info['summary'].T)


rats = all_rats_database["rat_name"].unique()
import plotly.express as px

for r in rats: 

    data = all_rats_database.loc[all_rats_database["rat_name"] == r]

    counts = (
        data
        .groupby(["rat_name", 'rat_type', 'condition', 'stim_location',
                  'handedness', 'view', 'laser_intensity', 'task'])
        .size()
        .reset_index(name="count")
    )

    fig = px.sunburst(
        counts,
        path=["rat_type", "condition", "view", "stim_location", 
              'handedness', 'laser_intensity', 'task'],
        values="count"
    )

    fig.update_layout(
        title=f"Trial composition of rat {r}",
    )

    output_path = cfg.paths.data_root / f"{r}_trial_compo"
    fig.write_html(str(output_path.with_suffix(".html")))
    fig.show()

