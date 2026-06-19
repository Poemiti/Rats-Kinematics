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
from rats_kinematics_utils.core.file_utils import print_analysis_info, check_analysis_choice, make_output_path
import rats_kinematics_utils.analysis.behavior_plot as bp
from rats_kinematics_utils.gui.figures_maker import load_figure_maker



# ---------------------------- setup ----------------------------------


cfg = load_config()
print_analysis_info(cfg, "Ethology Analysis")

input_filenames = sorted(cfg.paths.metrics.glob("*.joblib"))
filenames, plot_choice = load_figure_maker(input_filenames, kind="behavior")

check_analysis_choice(filenames, plot_choice)


# --------------------------------- trajectory analysis ----------------------------------

if plot_choice["plot_trajectory_behavior"] : 
    data_traj = bp.preprocess_trajectory_behavior(cfg, filenames)

    for laser_intensity, subset in data_traj.groupby("laser_intensity"): 
        
        g, crop_name = bp.plot_trajectory_behavior(cfg, data_traj, crop=True)

        g.add_legend(title="Behavior")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels("x (cm)", "y (cm)")
        g.figure.subplots_adjust(top=0.88)
        g.figure.suptitle(f"Behavior analysis{cfg.rat_name}\nLaser intensity: {laser_intensity} - Number of trials: {len(subset.groupby('id'))}", ha='center')
        g.savefig(make_output_path(cfg.paths.analysis / f"{cfg.rat_type}_behavior" / "trajectories", f"trajectories_{laser_intensity}{crop_name}.png"))


if plot_choice["plot_time_in_behavior_space"] : 
    data_traj = bp.preprocess_trajectory_behavior(cfg, filenames)

    for laser_intensity, subset in data_traj.groupby("laser_intensity"): 
    
        g = bp.plot_time_in_behavior_space(cfg, subset)

        g.add_legend(title="Behavior")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.figure.subplots_adjust(top=0.88)
        g.figure.suptitle(f"Behavior proportion of rat {cfg.rat_name}\nLaser intensity: {laser_intensity} - Number of trials: {len(subset.groupby('id'))}", ha='center')
        g.savefig(make_output_path(cfg.paths.analysis / f"{cfg.rat_type}_behavior", f"time_spend_per_behavior_{laser_intensity}.png"))



# --------------------------------- velocity and y pos analysis ----------------------------------

if plot_choice["plot_metric_behavior"] : 
    data_velocity = bp.preprocess_metric_behavior(cfg, filenames)

    bp.plot_metric_behavior(cfg, data_velocity, metric="velocity")
    bp.plot_metric_behavior(cfg, data_velocity, metric="y_pos")


# --------------------------------- behavior analysis ----------------------------------


if plot_choice["plot_ethogram_by_condition"] : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, filenames)
    biggest_condition, _ = bp.get_biggest_condition(cfg, filenames)

    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.plot_ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="pad_off")
    # bp.plot_ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_reach")
    # bp.plot_ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_open")
    # bp.plot_ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_grasp")
    # bp.plot_ethogram_by_condition(cfg, data_behavior, biggest_condition, align_by="first_press")


if plot_choice["plot_behavior_proba_per_condition"] : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, filenames)
    biggest_condition, _ = bp.get_biggest_condition(cfg, filenames)

    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.plot_behavior_proba_per_condition(cfg, data_behavior, align_by="pad_off")




if plot_choice["plot_behavior_proba_all"] : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, filenames)
    biggest_condition, _ = bp.get_biggest_condition(cfg, filenames)

    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.plot_behavior_proba_all(cfg, data_behavior )



if plot_choice["plot_behavior_proba_per_behavior"] : 

    data_proba = bp.preprocess_proba(cfg, filenames)
    bp.plot_behavior_proba_per_behavior(cfg, data_proba, output_dir=cfg.paths.analysis)


print("Done !")