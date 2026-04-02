#!/usr/bin/env python

import joblib, time, sys
import numpy as np
from tqdm import tqdm

from rats_kinematics_utils.file_management import open_DLC_results

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import print_analysis_info, make_output_path, check_times
from rats_kinematics_utils.trajectory_metrics import filter_outliers, filter_likelihood, interpolate_data
from rats_kinematics_utils.plot_preprocess import make_interpolation_figures

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Preprocessing")

RAT_NAME = cfg.rat_name

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)


filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))


# ----------------------- does preprocessing has already been done ? -----------------------------


file_to_preprocess = []

print("Does preprocessing already exist?\n")
nb = 0
for file in filenames:

    session = joblib.load(file)

    already_preprocess = all(
        trial.get(cfg.bodypart) is not None
        for trial in session
    )

    if already_preprocess:
        print(f"{file.stem}: yes")
        res = input("Overwrite? (y/n): ")

        if res == "y":
            file_to_preprocess.append(file)
            nb += len(session)

    else:
        print(f"{file.stem}: no")
        file_to_preprocess.append(file)
        nb += len(session)

# ----------------------- launch preprocess -----------------------------

if len(file_to_preprocess) == 0 :
    print("\nNo file no validate, stop !")
    sys.exit()


print("\nLaunching of the preprocessing for the following files :\n")
for f in file_to_preprocess : 
    print("  -",f.stem)
print(f"Total number of trial to preprocess : {nb}")

# ------------------------------------ loop ---------------------------------------

start = time.time()

for session_path in file_to_preprocess: 
    session = joblib.load(session_path)
    session_name = session_path.stem

    print(f"\nProcessing trials of {session_name}")
    for trial in tqdm(session) : 

        # do preprocessing on coordinates
        raw_coords = open_DLC_results(trial["filename_coords"])
        raw_coords = raw_coords[cfg.bodypart].copy()
        raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / cfg.fps)
        
        outlier_filtered_coords, params = filter_outliers(raw_coords, stat_method='eucli')
        likelihood_filtered_coords, likelihood_threshold = filter_likelihood(outlier_filtered_coords, cfg.threshold)
        interpolated_coords = interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)

        # make the figure to do the validation after
        if not check_times(trial["pad_off"], trial["laser_on"], cfg.laser_on_duration):
            trial[cfg.bodypart] = {
                    "trial_success" : False,
                    "xy_state" : "rejected",
                    "xy_before" : raw_coords,
                    "xy_after" : interpolated_coords
            }
            continue
        
        make_interpolation_figures(interpolated_coords, 
                                    likelihood_filtered_coords,
                                    outlier_filtered_coords,
                                    raw_coords,
                                    trial["pad_off"],
                                    title=session_name, 
                                    save_as=make_output_path(cfg.paths.figures / RAT_NAME / session_name / "preprocessing", f"{trial['name']}_interpolation.png"))


        # save trial bodyparts session
        trial[cfg.bodypart] = {
                "trial_success" : None,
                "xy_state" : None,
                "xy_before" : raw_coords,
                "xy_after" : interpolated_coords
        }

    joblib.dump(session, session_path)


end = time.time()
process_time = (end - start) / 60 # min

print(f"Processing time: {process_time:.1f} min")
print("Done !")

