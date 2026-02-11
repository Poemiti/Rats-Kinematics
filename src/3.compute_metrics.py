#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from rats_kinematics_utils.file_management import make_name_by_condition, verify_exist, open_clean_csv
from rats_kinematics_utils.trajectory_metrics import Trajectory
from rats_kinematics_utils.led_detection import get_time_led_on, get_time_led_off

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, init_metrics, to_yaml, check_lost_coords, check_non_empty, check_times

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
RAT_NAME = "#525"
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


for i, coords_path in enumerate(filenames) : 
    coords_path = Path(coords_path)

    print(f"\n[{i+1}/{len(filenames)}]")
    print(f"Getting coords of {coords_path}\n")

    # get time when pad is ON or OFF
    luminosity_path = cfg.paths.luminosity / RAT_NAME / coords_path.parent.stem / f"luminosity_{coords_path.stem.replace('pred_results_', '')}.csv"
    clip_path = cfg.paths.clips / RAT_NAME / coords_path.parent.stem / f"{coords_path.stem.replace('pred_results_', '')}.mp4"
    verify_exist(clip_path)
    verify_exist(luminosity_path)
    
    # if new condition, save metrics.yaml + initialise metrics dictionary + make new folder
    new_filename = make_name_by_condition(coords_path.stem)
    if new_filename != old_filename : 
        joblib.dump(METRICS, output_dir / f"{old_filename}.joblib")

        old_filename = new_filename
        METRICS = []
        print(f"File will be stored in {new_filename}")


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
    if not check_lost_coords(xy_filtered, coords) or \
    not check_times(time_pad_off, time_laser_on, len(xy), cfg.laser_on_duration) or \
    not check_non_empty(xy_filtered, time_pad_off):
        TRIAL_METRICS["trial_success"] = False
        TRIAL_METRICS["lost_coords"] = n_lost_coords
        TRIAL_METRICS["pad_off"] = time_pad_off
        TRIAL_METRICS["laser_on"] = time_laser_on
        METRICS.append(TRIAL_METRICS)
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
    TRIAL_METRICS["lost_coords"] = n_lost_coords
    TRIAL_METRICS["pad_off"] = time_pad_off
    TRIAL_METRICS["laser_on"] = time_laser_on

    TRIAL_METRICS["average_distance"] = Traj.distance()
    TRIAL_METRICS["average_velocity"] = Traj.mean_velocity()
    TRIAL_METRICS["peak_velocity"] = Traj.peak()
    TRIAL_METRICS["tortuosity"] = Traj.tortuosity()

    TRIAL_METRICS["instant_velocity"] = Traj.instant_velocity()
    TRIAL_METRICS["acceleration"] = Traj.acceleration()

    TRIAL_METRICS["xy_raw"] = xy
    TRIAL_METRICS["xy_filtered"] = xy_filtered
    TRIAL_METRICS["xy_pad_off"] = xy_pad_off
    TRIAL_METRICS["xy_laser_on"] = xy_laserOn
    
    METRICS.append(TRIAL_METRICS)

joblib.dump(METRICS, output_dir / f"{old_filename}.joblib")
print("Done !")

