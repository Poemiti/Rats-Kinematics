#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
import time
import seaborn as sns
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info
import rats_kinematics_utils.analysis.behavior_plot as bp



# ---------------------------- setup ----------------------------------

TRAJ = True
VELO = False
BEHA = False


cfg = load_config()
print_analysis_info(cfg, "Ethology Analysis")

filenames = list((cfg.paths.metrics).glob("*.joblib"))
filenames = sorted(filenames)

# --------------------------------- trajectory analysis ----------------------------------

if TRAJ : 
    data_traj = bp.preprocess_trajectory_behavior(cfg, filenames)

    # bp.plot_time_in_behavior_space(cfg, data_traj)
    bp.plot_trajectory_behavior(cfg, data_traj, crop=True)


# --------------------------------- velocity and y pos analysis ----------------------------------

if VELO : 
    data_velocity = bp.preprocess_velocity_behavior(cfg, filenames)

    bp.plot_metric_behavior(cfg, data_velocity, metric="velocity")
    bp.plot_metric_behavior(cfg, data_velocity, metric="y_pos")


# --------------------------------- behavior analysis ----------------------------------

if BEHA : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, filenames)
    biggest_condition, _ = bp.get_biggest_condition(cfg, filenames)

    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="pad_off")
    bp.ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_reach")
    bp.ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_open")
    bp.ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_grasp")
    bp.ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_press")


    # bp.behavior_proba_all(cfg, data_behavior )
    bp.behavior_proba_per_condition(cfg, data_behavior, align_by="pad_off")




print("Done !")