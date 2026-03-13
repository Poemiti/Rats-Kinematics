#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from rats_kinematics_utils.file_management import make_name_by_condition, verify_exist, open_DLC_results
from rats_kinematics_utils.led_detection import get_time_led_state

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, init_metadata, print_analysis_info, make_output_path, load_preprocess_validator
from rats_kinematics_utils.trajectory_metrics import filter_outliers, filter_likelihood, interpolate_data
from rats_kinematics_utils.plot_preprocess import make_interpolation_figures

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Validation of the preprocessing")


RAT_NAME = cfg.rat_name

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)


filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))


# ----------------------- does validation has already been done ? -----------------------------

file_to_validate = []

print("Does validation already exist?\n")

for file in filenames:

    metadata = joblib.load(file)

    already_validated = all(
        trial[cfg.bodypart]["xy_state"] is not None
        for trial in metadata
    )

    if already_validated:
        print(f"{file.stem}: yes")
        res = input("Overwrite? (y/n): ")

        if res == "y":
            file_to_validate.append(file)

    else:
        print(f"{file.stem}: no")
        file_to_validate.append(file)

# ----------------------- launch validation -----------------------------

print("\nLaunching of the validation for the following files :")
for f in file_to_validate : 
    print(f.stem)

validation_data = {}

for file in file_to_validate:
    val = load_preprocess_validator(cfg.paths.figures / RAT_NAME / file.stem)
    if not val : 
        break

    validation_data[file.stem] = val

print(f"\n.joblib files Outputs :")

for i, file in enumerate(file_to_validate): 

    filename = file.stem
    metadata = joblib.load(file)

    if filename in validation_data :
        for trial in metadata : 
            state = validation_data[filename][trial["name"]]
            trial[cfg.bodypart]["xy_state"] = state

            if state == "rejected" : 
                trial[cfg.bodypart]["trial_success"] = False 

            elif state == "raw" : 
                trial[cfg.bodypart]["trial_success"] = True 
                trial[cfg.bodypart]["xy_raw"] = trial[cfg.bodypart]["xy_before"]
                
            else : 
                trial[cfg.bodypart]["trial_success"] = True 
                trial[cfg.bodypart]["xy_raw"] = trial[cfg.bodypart]["xy_after"]

    # save updated metadata
    joblib.dump(metadata, output_dir / f"{filename}.joblib")

print("Done !")