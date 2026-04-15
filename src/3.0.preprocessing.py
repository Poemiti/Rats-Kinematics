#!/usr/bin/env python

import joblib, time, sys
import numpy as np
from tqdm import tqdm

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path
from rats_kinematics_utils.preprocessing.preprocess import check_times, filter_outliers, filter_likelihood, interpolate_data, open_DLC_results
from rats_kinematics_utils.preprocessing.plot_preprocess import make_interpolation_figures, plot_likelihood_across_frames, plot_preprocess_lost_points, plot_likelihood_distribution

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Preprocessing")

filenames = list((cfg.paths.metrics).glob("*.joblib"))


# ----------------------- does preprocessing has already been done ? -----------------------------


file_to_preprocess = []

print("Does preprocessing already exist?\n")
nb = 0
for file in filenames:

    is_left = "H001" in (file.stem).split("_")
    if cfg.view == "right" and is_left or \
        cfg.view == "left" and not is_left: 
        print(f"\nNOT THE RIGHT VIEW (!={cfg.view}):", file.stem, "\n")
        continue

    session = joblib.load(file)

    already_preprocess = all(
        trial.get(cfg.bodypart) is not None
        for trial in session
    )

    if already_preprocess:
        print(f"\n{file.stem}: yes")
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

res = input("\nENTER to launch preprocessing or (q) to quit : ")
if res == "q" or res=="Q": 
    print("quit!")
    sys.exit()

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
        if not check_times(trial["pad_off"], trial["laser_on"], cfg.laser_on_duration) or \
            trial["cue_type"] == "NoCue" :
                trial[cfg.bodypart] = {
                        "trial_success" : False,
                        "xy_state" : "rejected",
                }
                continue
        
        interpolation_path = make_output_path(cfg.paths.preprocessing / session_name, f"{trial['name']}_interpolation.png")
        make_interpolation_figures(interpolated_coords, 
                                    likelihood_filtered_coords,
                                    outlier_filtered_coords,
                                    raw_coords,
                                    trial["pad_off"],
                                    title=f"{session_name} - {cfg.view} view", 
                                    save_as=interpolation_path)


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



# ---------- show report before doing validation -----------


print("\nPlotting the number of points loosed at each step\n")
plot_preprocess_lost_points(cfg, filenames)


res = input("\nDo you want to plot likelihood across each trials ? "
" And the number of points loosed at each step ? (y/n) : ")

if res == "y" : 

    print("\nPlotting likelihood across frame of each trials\n")
    plot_likelihood_across_frames(cfg, filenames)

print("Done !")

