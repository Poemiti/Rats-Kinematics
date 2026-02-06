#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import yaml
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import make_directory_name, verify_exist
from utils.trajectory_ploting import open_clean_csv
from utils.led_detection import get_time_led_on, get_time_led_off

from config import load_config
from pipeline_maker import load_database, init_metrics, to_yaml, check_lost_coords, check_non_empty, check_times

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
DATABASE = load_database("csv")

RAT_NAME = DATABASE['rat_name'][0]

# ------------------------------------ get luminosity + classify ---------------------------------------

METRICS = {}

old_dir = cfg.paths.metrics / RAT_NAME / make_directory_name(Path(DATABASE["filename"][0]).stem)
old_dir.mkdir(parents=True, exist_ok=True)

db = sorted(DATABASE["filename"])

for i, coords_path in enumerate(db) : 
    coords_path = Path(coords_path)

    print(f"\n[{i}/{len(db)}]")
    print(f"Getting coords of {coords_path}\n")

    # get time when pad is ON or OFF
    luminosity_path = cfg.paths.luminosity / RAT_NAME / coords_path.parent.stem / f"luminosity_{coords_path.stem.replace('pred_results_', '')}.csv"
    clip_path = cfg.paths.clips / RAT_NAME / coords_path.parent.stem / f"{coords_path.stem.replace('pred_results_', '')}.mp4"
    verify_exist(clip_path)
    verify_exist(luminosity_path)
    
    # if new condition, save metrics.yaml + initialise metrics dictionary + make new folder
    new_dir = cfg.paths.metrics / RAT_NAME / make_directory_name(coords_path.stem)
    if new_dir != old_dir : 
        with open(cfg.paths.metrics / old_dir / f"metrics.yaml", "w") as file : 
            yaml.dump(METRICS, file , default_flow_style=False, indent=4, sort_keys=False)

        old_dir = new_dir
        METRICS = {}
        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"File will be stored in {new_dir}")


    ##################################

    TRIAL_METRICS : dict = init_metrics(coords_path,
                                        luminosity_path,
                                        clip_path)
    

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
    if not check_lost_coords(xy_filtered, coords):
        METRICS[clip_path.stem] = TRIAL_METRICS
        continue

    if not check_times(time_pad_off, time_laser_on, len(xy), cfg.laser_on_duration):
        METRICS[clip_path.stem] = TRIAL_METRICS
        continue

    if not check_non_empty(xy_filtered, time_pad_off):
        METRICS[clip_path.stem] = TRIAL_METRICS
        continue
    
    #  pad off -> laser off coords 
    xy_pad_off = xy_filtered.loc[
        (xy_filtered["t"] >= time_pad_off) &
        (xy_filtered["t"] <= time_pad_off + cfg.laser_on_duration + 0.025)
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
    



    # compute metrics
    Traj = Trajectory(coords=xy,
                    reaching_coords=xy_pad_off, 
                    laserOn_coords=xy_laserOn,
                    fps=cfg.fps, 
                    cm_per_pixel=cfg.cm_per_pixel)

    
    # final saving
    TRIAL_METRICS["trial_success"] = True
    TRIAL_METRICS["pad_off"] = time_pad_off
    TRIAL_METRICS["laser_on"] = time_laser_on

    TRIAL_METRICS["average_distance"] = to_yaml(Traj.distance())
    TRIAL_METRICS["average_velocity"] = to_yaml(Traj.mean_velocity())
    TRIAL_METRICS["peak_velocity"] = to_yaml(Traj.peak())
    TRIAL_METRICS["tortuosity"] = to_yaml(Traj.tortuosity())

    TRIAL_METRICS["instant_velocity"] = to_yaml(Traj.instant_velocity())
    TRIAL_METRICS["acceleration"] = to_yaml(Traj.acceleration())

    TRIAL_METRICS["xy_raw"] = to_yaml(xy)
    TRIAL_METRICS["xy_filtered"] = to_yaml(xy_filtered)
    TRIAL_METRICS["xy_pad_off"] = to_yaml(xy_pad_off)
    TRIAL_METRICS["xy_laser_on"] = to_yaml(xy_laserOn)
    
    METRICS[clip_path.stem] = TRIAL_METRICS

with open(cfg.paths.metrics / old_dir / f"metrics.yaml", "w") as file : 
            yaml.dump(METRICS, file , default_flow_style=False, indent=4, sort_keys=False)

print("Done !")

