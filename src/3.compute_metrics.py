#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import yaml
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_single_bodypart_trajectories, open_clean_csv, plot_3D_traj
from utils.video_annotation import annotate_single_bodypart
from utils.trajectory_metrics import Trajectory, plot_metric_time, define_End_of_trajectory, animate_plot
from utils.led_detection import get_time_led_on, get_time_led_off
from utils.split_video import split_clip_range

from config import load_config
from pipeline_maker import load_database, init_overall_metrics, init_trial_metrics

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
DATABASE = load_database()

RAT_NAME = DATABASE['rat_name'][0]

# ------------------------------------ get luminosity + classify ---------------------------------------

METRICS = init_overall_metrics()

old_dir = cfg.paths.metrics / make_directory_name(Path(DATABASE["filename"][0]).stem)
old_dir.mkdir(parents=True, exist_ok=True)

for i, coords_path in enumerate(DATABASE["filename"].iloc[:]) : 
    coords_path = Path(coords_path)

    print(f"\n[{i}/{len(DATABASE)}]")
    print(f"Getting coords of {coords_path}\n")

    # get time when pad is ON or OFF
    luminosity_path = cfg.paths.luminosity / RAT_NAME / coords_path.parent.stem / f"luminosity_{coords_path.stem.replace('pred_results_', '')}.csv"
    clip_path = cfg.paths.clips / RAT_NAME / coords_path.parent.stem / f"{coords_path.stem.replace('pred_results_', '')}.mp4"
    verify_exist(clip_path)
    verify_exist(luminosity_path)


    # if new condition, initialise metrics dictionary + make new folder
    new_dir = cfg.paths.metrics / make_directory_name(coords_path.stem)
    if new_dir != old_dir : 
        METRICS = init_overall_metrics()
        TRIAL_METRICS = init_trial_metrics(coords_path,
                                            luminosity_path,
                                            clip_path)

        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"File will be stored in {new_dir}")


    ##################################


    time_pad_off = get_time_led_off(luminosity_path, "LED_3", in_sec=True) # in sec
    time_laser_on = get_time_led_on(luminosity_path, "LED_4", in_sec=True) # in sec

    # get coords + filtration
    coords = open_clean_csv(coords_path)
    xy = coords[cfg.bodypart].copy()
    xy = xy.assign(t=np.arange(len(xy)) / 125)
    xy_filtered = xy.loc[xy["likelihood"] >= cfg.threshold, ["x", "y", "t"]]

    # count the number of lost coord after threshold
    n_lost_coords = len(coords) - len(xy_filtered)
    print("nb lost coords = ", n_lost_coords)

    # verification
    if n_lost_coords > 10 : # in sec
        print(f"  ! Too much lost coords : {n_lost_coords}")
        METRICS["metrics_per_trial"].append(TRIAL_METRICS)
        continue

    # verification
    if time_pad_off is None or time_laser_on+cfg.laser_on_duration > len(xy)-1 : # in sec
        print(f"  ! Failed trial on, Pad off at {time_pad_off}")
        METRICS["metrics_per_trial"].append(TRIAL_METRICS)
        continue
    
    #  pad off -> laser off coords 
    xy_pad_off = xy_filtered.loc[
        (xy_filtered["t"] >= time_pad_off) &
        (xy_filtered["t"] <= time_laser_on + cfg.laser_on_duration)
    ].reset_index(drop=True)

    print(f"pad off={time_pad_off}, laser on={time_laser_on}")

    # laser on -> laser off coords
    if time_laser_on: 
        xy_laserOn = xy_filtered.loc[
            (xy_filtered["t"] >= time_laser_on) &
            (xy_filtered["t"] <= time_laser_on + 0.3)
        ].reset_index(drop=True)
    else : 
        xy_laserOn = xy_pad_off

    # verification
    if len(xy_filtered) == 0 : 
        print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
        METRICS["metrics_per_trial"].append(TRIAL_METRICS)
        continue


    # compute metrics
    Traj = Trajectory(coords=xy,
                        reaching_coords=xy_pad_off, 
                        laserOn_coords=xy_laserOn,
                        fps=125, 
                        cm_per_pixel=cfg.cm_per_pixel)

    
    # final saving
    TRIAL_METRICS["trial success"] = True
    TRIAL_METRICS["pad off"] = True
    TRIAL_METRICS["laser on"] = True
    TRIAL_METRICS["instant velocity"] = True
    TRIAL_METRICS["acceleration"] = True
    TRIAL_METRICS["xy raw"] = True
    TRIAL_METRICS["xy filtered"] = True
    TRIAL_METRICS["xy pad off"] = True
    TRIAL_METRICS["xy laser on"] = True


    METRICS["metrics_per_trial"].append(TRIAL_METRICS)

