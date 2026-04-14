#!/usr/bin/env python

import joblib
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info
from rats_kinematics_utils.preprocessing.preprocess import check_times
from rats_kinematics_utils.preprocessing.plot_preprocess import plot_trial_success_distri, plot_trial_failure_reason, plot_trial_failure_reason_detail
from rats_kinematics_utils.gui.preprocess_validator import load_preprocess_validator

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Validation of the preprocessing")

filenames = list((cfg.paths.metrics).glob("*.joblib"))

# ----------------------- does validation has already been done ? -----------------------------

file_to_validate = []

print("Does validation already exist?\n")
nb = 0
for file in filenames:

    is_left = "H001" in (file.stem).split("_")
    if cfg.view == "right" and is_left or \
        cfg.view == "left" and not is_left: 
        print(f"\nNOT THE RIGHT VIEW (!={cfg.view}):", file.stem, "\n")
        continue

    metadata = joblib.load(file)

    already_validated = all(
        trial[cfg.bodypart]["xy_state"] is not None
        for trial in metadata
    )

    if already_validated:
        print(f"\n{file.stem}: yes")
        res = input("Overwrite? (y/n): ")

        if res == "y":
            file_to_validate.append(file)
            nb += len(metadata)

    else:
        print(f"{file.stem}: no")
        file_to_validate.append(file)
        nb += len(metadata)

# ----------------------- launch validation -----------------------------

if len(file_to_validate) == 0 :
    print("\nNo file no validate, stop !")
    sys.exit()


print("\nLaunching of the validation for the following files :\n")
for f in file_to_validate : 
    print("  -",f.stem)
print(f"Total number of trial to validate : {nb}")

validation_data = {}

for file in file_to_validate:
    val = load_preprocess_validator(cfg.paths.preprocessing / file.stem)
    if not val : 
        break

    validation_data[file.stem] = val

print(f"\n.joblib files Outputs :")

for i, file in enumerate(file_to_validate): 

    filename = file.stem
    print(filename)
    metadata = joblib.load(file)

    if filename in validation_data :
        for trial in metadata : 
            if not check_times(trial["pad_off"], trial["laser_on"], cfg.laser_on_duration) or \
                trial["cue_type"] == "NoCue" : # pass the "rejected" by the preprocessing
                continue

            state = validation_data[filename].get(trial["name"])
            trial[cfg.bodypart]["xy_state"] = state

            if state is None :              # pass when the validation has been stop in the middle
                continue

            elif state == "rejected" : 
                trial[cfg.bodypart]["trial_success"] = False 

            elif state == "raw" : 
                trial[cfg.bodypart]["trial_success"] = True 
                trial[cfg.bodypart]["xy_raw"] = trial[cfg.bodypart]["xy_before"]
                
            else : 
                trial[cfg.bodypart]["trial_success"] = True 
                trial[cfg.bodypart]["xy_raw"] = trial[cfg.bodypart]["xy_after"]

    # save updated metadata
    joblib.dump(metadata, cfg.paths.metrics / f"{filename}.joblib")


# -------------- show report of the successful trials ------------

print("\nPlotting the distribution of trial success\n")
plot_trial_success_distri(cfg, filenames)

print("\nPlotting the distribution of the failure reason\n")
plot_trial_failure_reason(cfg, filenames)

print("\nPlotting the distribution of the failure reason DETAIL VERSION\n")
plot_trial_failure_reason_detail(cfg, filenames)
