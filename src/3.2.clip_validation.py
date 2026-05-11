#!/usr/bin/env python

import joblib
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info
from rats_kinematics_utils.preprocessing.preprocess import check_times
from rats_kinematics_utils.preprocessing.plot_preprocess import plot_manual_clip_success_distri, make_clip_annotation_for_validation
from rats_kinematics_utils.gui.clip_validator import load_clip_validator

"""
This script is used to verify manually each clip. Only the clips that were previously
set as 'successful' will be reviewed. 
In theory, this script is only used for rats that have abnormal behavior 
(Ex : rat 531 that cross paw)
"""


# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Validation of the preprocessing")

filenames = list((cfg.paths.metrics).glob("*.joblib"))


# ----------------------- does validation has already been done ? -----------------------------

file_to_validate = []

print("Does CLIP validation already exist?\n")
nb = 0
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

    already_validated = all(
         trial[cfg.bodypart].get("behavior_state") is not None
        for trial in metadata
    )

    nb_successful = sum(1 for m in metadata if m[cfg.bodypart].get('trial_success'))

    if already_validated:
        print(f"\n{file.stem}: yes")
        res = input("Overwrite? (y/n): ")

        if res == "y":
            file_to_validate.append(file)
            nb += nb_successful

    else:
        print(f"\n{file.stem}: no")
        file_to_validate.append(file)
        nb += nb_successful

# ----------------------- launch validation -----------------------------

if len(file_to_validate) == 0 :
    print("\nNo file no validate, stop !")
    sys.exit()


print("\nLaunching of the validation for the following files :\n")
for f in file_to_validate : 
    print("  -",f.stem)
print(f"Total number of trial to validate : {nb}")

res = input("\nENTER to launch preprocessing or (q) to quit : ")
if res == "q" or res=="Q": 
    print("quit!")
    sys.exit()

# -------------------------------- annotate videos ------------------------------------

if (cfg.paths.processed / "annotated_clips").exists() :
    res = input("\nClips are already annotated, do you want to overwrite ? (y/n): ")
    if res == "y" : 
        make_clip_annotation_for_validation(cfg, file_to_validate)

# ------------------------------- validator ---------------------------------------

validation_data = {}

for file in file_to_validate:
    val = load_clip_validator(cfg.paths.processed / "annotated_clips" / file.stem)
    if not val : 
        break

    validation_data[file.stem] = val


for k, v in validation_data.items(): 
    print(k, v)
print(f"\n.joblib files Outputs :")

for i, file in enumerate(file_to_validate): 

    filename = file.stem
    print(filename)
    metadata = joblib.load(file)

    if filename in validation_data :
        for trial in metadata : 

            state = validation_data[filename].get(trial["name"])
            trial[cfg.bodypart]["behavior_state"] = state


    # save updated metadata
    joblib.dump(metadata, cfg.paths.metrics / f"{filename}.joblib")


print("\nPlotting the distribution of clip success\n")
plot_manual_clip_success_distri(cfg, filenames)