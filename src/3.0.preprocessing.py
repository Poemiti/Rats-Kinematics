#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm

from rats_kinematics_utils.file_management import make_name_by_condition, verify_exist, open_DLC_results
from rats_kinematics_utils.led_detection import get_time_led_state

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, init_metadata, print_analysis_info, make_output_path, check_times
from rats_kinematics_utils.trajectory_metrics import filter_outliers, filter_likelihood, interpolate_data
from rats_kinematics_utils.plot_preprocess import make_interpolation_figures

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Preprocessing")


RAT_NAME = cfg.rat_name
DATABASE = load_database(cfg.paths.coords / RAT_NAME, cfg.paths.database, "csv")

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)

SESSION_METADATA = []

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

all_metadata = {}

for coords_path in tqdm(filenames, desc="Preprocessing trials"): 

    # get all the path (coordinates, luminosity and video clips)
    coords_path = Path(coords_path)
    luminosity_path = cfg.paths.luminosity / RAT_NAME / coords_path.parent.stem / f"luminosity_{coords_path.stem.replace('pred_results_', '')}.csv"
    clip_path = cfg.paths.clips / RAT_NAME / coords_path.parent.stem / f"{coords_path.stem.replace('pred_results_', '')}.mp4"
    verify_exist(clip_path)
    verify_exist(luminosity_path)

    trial_name = clip_path.stem
    
    # if new condition, save metrics.jpblib + initialise metadata dictionary + make new folder
    new_filename = make_name_by_condition(coords_path.stem)
    if new_filename != old_filename: 
        if old_filename in all_metadata.keys() : 
            for trial_meta in SESSION_METADATA :
                all_metadata[old_filename].append(trial_meta)
        else : 
            all_metadata[old_filename] = SESSION_METADATA

        old_filename = new_filename
        SESSION_METADATA = []
        # print(f"File will be stored in {new_filename}")


    ##################################

    TRIAL_METADATA: dict = init_metadata(coords_path,
                                        luminosity_path,
                                        clip_path)
    
    # get led info
    time_pad_off = get_time_led_state(luminosity_path, cfg.task_pad, "OFF", min_duration=10,  in_sec=True)
    time_laser_on = get_time_led_state(luminosity_path, "LED_4", "ON", in_sec=True)
    time_reward = get_time_led_state(luminosity_path, "LED_5", "ON", in_sec=True)

    # do preprocessing on coordinates
    raw_coords = open_DLC_results(coords_path)
    raw_coords = raw_coords[cfg.bodypart].copy()
    raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / cfg.fps)
    
    outlier_filtered_coords, params = filter_outliers(raw_coords, stat_method='eucli')
    likelihood_filtered_coords = filter_likelihood(outlier_filtered_coords, cfg.threshold)
    interpolated_coords = interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)

    # make the figure to do the validation after
    if not check_times(time_pad_off, time_laser_on, cfg.laser_on_duration):
        TRIAL_METADATA["pad_off"] = time_pad_off
        TRIAL_METADATA["laser_on"] = time_laser_on
        TRIAL_METADATA["reward"] = time_reward
        TRIAL_METADATA[cfg.bodypart] = {
                "trial_success" : False,
                "xy_state" : "rejected",
                "xy_before" : raw_coords,
                "xy_after" : interpolated_coords
        }
        SESSION_METADATA.append(TRIAL_METADATA)
        continue
    
    make_interpolation_figures(interpolated_coords, 
                                likelihood_filtered_coords,
                                outlier_filtered_coords,
                                raw_coords,
                                time_pad_off,
                                title=old_filename, 
                                save_as=make_output_path(cfg.paths.figures / RAT_NAME / old_filename / "preprocessing", f"{trial_name}_interpolation.png"))

    # save trial metadata
    TRIAL_METADATA["pad_off"] = time_pad_off
    TRIAL_METADATA["laser_on"] = time_laser_on
    TRIAL_METADATA["reward"] = time_reward

    # save trial bodyparts metadata
    TRIAL_METADATA[cfg.bodypart] = {
            "trial_success" : None,
            "xy_state" : None,
            "xy_before" : raw_coords,
            "xy_after" : interpolated_coords
    }

    SESSION_METADATA.append(TRIAL_METADATA)


all_metadata[old_filename] = SESSION_METADATA
n_trial = 0 

print(f"\n.joblib files Outputs :")
for filename, meta in all_metadata.items() : 
    joblib.dump(meta, output_dir / f"{filename}.joblib")
    print(f"  {filename} : {len(meta)}")
    n_trial += len(meta)

print(f"Number of joblib files generated: {len(all_metadata)}")
print(f"Number of trials processed: {n_trial}")
print(f"Number of true pred.csv processed: {len(filenames)}")
print("Done !")

