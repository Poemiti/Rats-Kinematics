#!/usr/bin/env python

import joblib
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info
from rats_kinematics_utils.preprocessing.preprocess import check_times
from rats_kinematics_utils.preprocessing.plot_preprocess import plot_trial_success_distri, plot_trial_failure_reason, plot_trial_failure_reason_detail
from rats_kinematics_utils.gui.clip_validator import load_clip_validator

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Validation of the preprocessing")

filenames = list((cfg.paths.metrics).glob("*.joblib"))



validation_data = {}

for file in filenames:
    val = load_clip_validator(cfg.paths.processed / "annotated_clips" / file.stem)
    if not val : 
        break

    validation_data[file.stem] = val


for k, v in validation_data.items(): 
    print(k, v)
print(f"\n.joblib files Outputs :")

sys.exit()

for i, file in enumerate(filenames): 

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

