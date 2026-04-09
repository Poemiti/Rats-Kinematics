#!/usr/bin/env python

import joblib
import sys

from rats_kinematics_utils.preprocessing.Trajectory import Trajectory
from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info
from rats_kinematics_utils.preprocessing.preprocess import check_reward, crop_xy

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Compute metrics")

filenames = list((cfg.paths.metrics).glob("*.joblib"))


# ----------------------- does validation has already been done ? -----------------------------

file_to_compute = []
nb = 0

print("Does metrics computation already exist?\n")

for file in filenames:

    metadata = joblib.load(file)

    is_validated = all(
        trial[cfg.bodypart]["xy_state"] is not None
        for trial in metadata
    )

    if not is_validated : 
        print(f"\nThis file has not been validated : {file.stem}")
        print(f"Please run 3.1.preprocessing_validation before !")
        continue

    already_computed = True if "instant_velocity" in metadata[0][cfg.bodypart] else False

    if already_computed:
        print(f"{file.stem}: yes")
        res = input("Overwrite? (y/n): ")

        if res == "y":
            file_to_compute.append(file)

    else:
        print(f"{file.stem}: no")
        file_to_compute.append(file)


# ------------------------------------ lauching of the computation ---------------------------------------

if len(file_to_compute) == 0 :
    print("\nNo file no compute, stop !")
    sys.exit()

print("\nLaunching of the metric computation for the following files :\n")
for f in file_to_compute : 
    print("  -",f.stem)


for i, file in enumerate(file_to_compute): 

    # open file and data
    filename  = file.stem
    data = joblib.load(file)

    for trial in data : 

        nb+=1

        if not trial[cfg.bodypart]["trial_success"] : 
            for metric in ["average_velocity", "peak_velocity", "tortuosity", "instant_velocity",
                           "acceleration", "xy_pad_off", "xy_laser_on", "xy_reward"] :
                trial[cfg.bodypart][metric] = None
            continue

        xy = trial[cfg.bodypart]["xy_raw"]
        time_pad_off = trial["pad_off"]
        time_laser_on = trial["laser_on"]
        time_reward = trial["reward"]

        #  pad off -> laser off coords 
        xy_pad_off = crop_xy(xy, time_pad_off, time_pad_off + cfg.laser_on_duration + 0.025) 

        if check_reward(time_reward) :
            xy_reward = crop_xy(xy, time_pad_off, time_reward) 
        else :
            xy_reward = None

        # laser on -> laser off coords
        if time_laser_on: 
            xy_laserOn = crop_xy(xy, time_laser_on, time_laser_on+0.3) 
        else : 
            xy_laserOn = None

        # compute metrics
        Traj_full = Trajectory(xy, cm_per_pixel=cfg.cm_per_pixel)
        Traj_pad_off = Trajectory(xy_pad_off, cm_per_pixel=cfg.cm_per_pixel)
        
        trial[cfg.bodypart]["average_velocity"] = Traj_pad_off.mean_speed()
        trial[cfg.bodypart]["peak_velocity"] = Traj_pad_off.peak_speed()
        trial[cfg.bodypart]["tortuosity"] = Traj_pad_off.tortuosity()

        trial[cfg.bodypart]["instant_velocity"] = Traj_full.instant_velocity()
        trial[cfg.bodypart]["acceleration"] = Traj_full.acceleration()

        trial[cfg.bodypart]["xy_pad_off"] = xy_pad_off
        trial[cfg.bodypart]["xy_laser_on"] = xy_laserOn
        trial[cfg.bodypart]["xy_reward"] = xy_reward
        
    # save updated metadata + metrics and trajectories
    joblib.dump(data, cfg.paths.metrics / f"{filename}.joblib")

print(nb)
print("Done !")
