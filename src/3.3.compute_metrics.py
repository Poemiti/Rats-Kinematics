#!/usr/bin/env python

import joblib, yaml
import sys, time
from tqdm import tqdm
from datetime import datetime

from rats_kinematics_utils.preprocessing.Trajectory import Trajectory
from rats_kinematics_utils.preprocessing.BehaviorBox import BehaviorBox

from rats_kinematics_utils.core.config import load_config, match_rule
from rats_kinematics_utils.core.file_utils import print_analysis_info
from rats_kinematics_utils.preprocessing.preprocess import check_reward, crop_xy
from rats_kinematics_utils.preprocessing.plot_preprocess import verify_behavior_box

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Compute metrics")

filenames = list((cfg.paths.metrics).glob("*.joblib"))

# ----------------------- does validation has already been done ? -----------------------------

file_to_compute = []
nb = 0

print("Does metrics computation already exist?\n")

for file in filenames:

    is_left = "H001" in (file.stem).split("_")
    if cfg.view == "right" and is_left or \
        cfg.view == "left" and not is_left: 
        print(f"\nNOT THE RIGHT VIEW (!={cfg.view}):", file.stem, "\n")
        continue

    if not cfg.rat_type in  (file.stem).split("_") : 
        print(f"\nNOT THE RIGHT RAT TYPE (!={cfg.rat_type}):", file.stem, "\n")
        continue

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

res = input("\nENTER to launch preprocessing or (q) to quit : ")
if res == "q" or res=="Q": 
    print("quit!")
    sys.exit()


start = time.perf_counter()

for file in tqdm(file_to_compute): 

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


        # compute the behavioral metrics
        with open("ethology_rules.yaml", "r") as f: 
            etho_rules = yaml.safe_load(f)

        date = datetime.fromisoformat(trial["date"])
        etho_meta = {
                "rat": int(trial["rat_name"][1:]),
                "day": date.day,
                "condition": trial["condition"],
                "view": trial["camera_view"],
                "month": date.month,
            }
        
        anchors_position = match_rule(etho_meta, etho_rules)

        # print(trial["name"])

        # compute trajectory
        Traj_full = Trajectory(xy, cm_per_pixel=cfg.cm_per_pixel, 
                               lever_position=anchors_position["lever"], 
                               frame_width=cfg.frame_width_px)
        Traj_pad_off = Trajectory(xy_pad_off, cm_per_pixel=cfg.cm_per_pixel, 
                                  lever_position=anchors_position["lever"],
                                  frame_width=cfg.frame_width_px)
        
        
        trial[cfg.bodypart]["average_velocity"] = Traj_pad_off.mean_speed()
        trial[cfg.bodypart]["peak_velocity"] = Traj_pad_off.peak_speed()
        trial[cfg.bodypart]["relative_mean_velocity"] = Traj_pad_off.relative_mean_speed(view=trial["camera_view"])
        trial[cfg.bodypart]["pre_post_velocity"] = Traj_pad_off.pre_post_velocity(time_pad_off=time_pad_off)

        trial[cfg.bodypart]["instant_velocity"] = Traj_full.instant_velocity()
        trial[cfg.bodypart]["acceleration"] = Traj_full.acceleration()
        trial[cfg.bodypart]["relative_velocity"] = Traj_full.relative_speed(view=trial["camera_view"])
        trial[cfg.bodypart]["lever_distance"] = Traj_full.lever_bodypart_distance()
        trial[cfg.bodypart]["tortuosity"] = Traj_full.tortuosity(time_pad_off=time_pad_off)
        trial[cfg.bodypart]["tortuosity_laser_period"] = Traj_full.tortuosity(time_pad_off=time_pad_off, fixed_period=cfg.laser_on_duration + 0.025)


        trial[cfg.bodypart]["xy_pad_off"] = xy_pad_off
        trial[cfg.bodypart]["xy_laser_on"] = xy_laserOn
        trial[cfg.bodypart]["xy_reward"] = xy_reward




        Boxes = BehaviorBox(xy_lever=anchors_position["lever"],
                            xy_pad=anchors_position["pad"],
                            view=trial["camera_view"],
                            frame_width=cfg.frame_width_px)

        trial[cfg.bodypart]["xy_etho"] = Boxes.classify_trajectory(xy)
        trial["behavior_anchors"] = anchors_position
        
    # save updated metadata + metrics and trajectories
    joblib.dump(data, cfg.paths.metrics / f"{filename}.joblib")


end = time.perf_counter()
process_time = (end - start) # sec

print(f"\nNumber of trial processed: {nb}")
print(f"Computation time: {process_time:.2f} s")

res = input("Do you want to annotate videos with the Behavior Boxes ? (y/n) : ")

if res == "y" : 
    print("\nAnnotating video to verify Behavior Boxes\n")
    verify_behavior_box(cfg, file_to_compute)

print("Done !")
