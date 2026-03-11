#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from rats_kinematics_utils.file_management import make_name_by_condition, verify_exist, open_DLC_results
from rats_kinematics_utils.trajectory_metrics import Trajectory, crop_xy
from rats_kinematics_utils.led_detection import get_time_led_state

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, init_metrics, check_lost_coords, check_non_empty, check_times, check_reward, print_analysis_info

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Computing metrics")


RAT_NAME = cfg.rat_name
DATABASE = load_database(cfg.paths.coords / RAT_NAME, cfg.paths.database, "csv")

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)

METRICS = []

old_filename =  make_name_by_condition(Path(DATABASE["filename"][0]).stem)

filenames = (
    DATABASE.sort_values(
        by=["rat_name", "rat_type", "condition", "task", "laser_intensity", "laser_on"],
        ascending=[True, True, True, True, True, True], 
    )
    ["filename"]
    .tolist()
)


# ------------------------------------ loop ---------------------------------------

ALL_METRICS = {}

for i, coords_path in enumerate(filenames) : 
    
    # get all the path (coordinates, luminosity and video clips)
    coords_path = Path(coords_path)
    luminosity_path = cfg.paths.luminosity / RAT_NAME / coords_path.parent.stem / f"luminosity_{coords_path.stem.replace('pred_results_', '')}.csv"
    clip_path = cfg.paths.clips / RAT_NAME / coords_path.parent.stem / f"{coords_path.stem.replace('pred_results_', '')}.mp4"
    verify_exist(clip_path)
    verify_exist(luminosity_path)
    
    # if new condition, save metrics.yaml + initialise metrics dictionary + make new folder
    new_filename = make_name_by_condition(coords_path.stem)
    if new_filename != old_filename: 
        if old_filename in ALL_METRICS.keys() : 
            for trial_metric in METRICS :
                ALL_METRICS[old_filename].append(trial_metric)
        else : 
            ALL_METRICS[old_filename] = METRICS

        old_filename = new_filename
        METRICS = []
        print(f"File will be stored in {new_filename}")


    ##################################

    TRIAL_METRICS : dict = init_metrics(coords_path,
                                        luminosity_path,
                                        clip_path)
    

    time_pad_off = get_time_led_state(luminosity_path, cfg.task_pad, "OFF", min_duration=10,  in_sec=True)
    time_laser_on = get_time_led_state(luminosity_path, "LED_4", "ON", in_sec=True)
    time_reward = get_time_led_state(luminosity_path, "LED_5", "ON", in_sec=True)

    # get coords + filtration
    coords = open_DLC_results(coords_path)
    xy = coords[cfg.bodypart].copy()
    xy = xy.assign(t=np.arange(len(xy)) / 125)
    xy_filtered = xy.loc[xy["likelihood"] >= cfg.threshold, ["x", "y", "t"]]

    # verification
    if not check_times(time_pad_off, time_laser_on, len(xy), cfg.laser_on_duration) or \
    not check_non_empty(xy_filtered, time_pad_off):
        TRIAL_METRICS["trial_success"] = False
        TRIAL_METRICS["pad_off"] = time_pad_off
        TRIAL_METRICS["laser_on"] = time_laser_on
        TRIAL_METRICS["reward"] = time_reward
        TRIAL_METRICS["xy_raw"] = xy
        METRICS.append(TRIAL_METRICS)
        continue
    
    #  crop coordinates 
    xy_raw_pad_off = crop_xy(xy, time_pad_off, time_pad_off + cfg.laser_on_duration + 0.025) 
    xy_pad_off = crop_xy(xy_filtered, time_pad_off, time_pad_off + cfg.laser_on_duration + 0.025) 

    xy_reward = crop_xy(xy_filtered, time_pad_off, time_reward) if check_reward(time_reward) else None
    xy_laserOn = crop_xy(xy_filtered, time_laser_on, time_laser_on+0.3) if time_laser_on else None

    n_lost_coords = len(xy_raw_pad_off) - len(xy_pad_off)

    # verification
    if not check_lost_coords(xy_pad_off, xy_raw_pad_off):
        TRIAL_METRICS["trial_success"] = False
        TRIAL_METRICS["lost_coords"] = n_lost_coords
        TRIAL_METRICS["pad_off"] = time_pad_off
        TRIAL_METRICS["laser_on"] = time_laser_on
        TRIAL_METRICS["reward"] = time_reward
        TRIAL_METRICS["xy_raw"] = xy
        METRICS.append(TRIAL_METRICS)
        continue

    # print("  success")
    # compute metrics
    Traj_pad_off = Trajectory(xy_pad_off, cm_per_pixel=cfg.cm_per_pixel)
    Traj_filtered = Trajectory(xy_filtered, cm_per_pixel=cfg.cm_per_pixel)
    
    # final saving
    TRIAL_METRICS["trial_success"] = True
    TRIAL_METRICS["lost_coords"] = n_lost_coords
    TRIAL_METRICS["pad_off"] = time_pad_off
    TRIAL_METRICS["laser_on"] = time_laser_on
    TRIAL_METRICS["reward"] = time_reward

    TRIAL_METRICS["average_velocity"] = Traj_pad_off.mean_speed()
    TRIAL_METRICS["peak_velocity"] = Traj_pad_off.peak_speed()
    TRIAL_METRICS["tortuosity"] = Traj_pad_off.tortuosity()

    TRIAL_METRICS["instant_velocity"] = Traj_filtered.instant_velocity()
    TRIAL_METRICS["acceleration"] = Traj_filtered.acceleration()

    TRIAL_METRICS["xy_raw"] = xy
    TRIAL_METRICS["xy_filtered"] = xy_filtered
    TRIAL_METRICS["xy_pad_off"] = xy_pad_off
    TRIAL_METRICS["xy_laser_on"] = xy_laserOn
    TRIAL_METRICS["xy_reward"] = xy_reward
    
    METRICS.append(TRIAL_METRICS)

ALL_METRICS[old_filename] = METRICS
n_trial = 0 

print(f"\n.joblib files Outputs :")
for filename, metrics in ALL_METRICS.items() : 
    print(f"  {filename} : {len(metrics)}")
    joblib.dump(metrics, output_dir / f"{filename}.joblib")
    n_trial += len(metrics)

print(f"Number of joblib files generated: {len(ALL_METRICS)}")
print(f"Number of trials processed: {n_trial}")
print(f"Number of true pred.csv processed: {len(filenames)}")
print("Done !")

