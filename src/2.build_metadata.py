#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib, yaml
import sys
import time

from rats_kinematics_utils.core.file_utils import make_name_by_condition, verify_exist, is_left_view, get_date, get_condition
from rats_kinematics_utils.preprocessing.led_detection import get_time_led_state, get_luminosity, define_cue_type
from rats_kinematics_utils.preprocessing.preprocess import init_metadata

from rats_kinematics_utils.core.config import load_config, match_rule
from rats_kinematics_utils.gui.database_filter import load_database
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path
from rats_kinematics_utils.preprocessing.plot_preprocess import metadata_report, plot_likelihood_distribution

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Build trial metadata")

DATABASE = load_database(cfg.paths.dlc, cfg.paths.database, "csv")

output_dir = cfg.paths.metrics
output_dir.mkdir(parents=True, exist_ok=True)

old_filename = ""


# ------------------------------------ warning message if metadata exist already ---------------------------------------

for file in output_dir.iterdir() : 
    if file.suffix == ".yaml" or file.suffix == ".joblib" : 
        print(f"Some metadata already exist in {output_dir} !")
        res = input("Do you want to overwrite ? (y/n) : ")

        if res == "n" : 
            print("Metadata building cancelled, stop !")
            sys.exit()

        elif res == "y" : 
            break

        else : 
            raise ValueError(f"ERROR : '{res}' is not a valid answer. Must be 'y' or 'n'")


# ------------------------------------ loop ---------------------------------------

ALL_METADATA = {}
dirs = set()
n_wrong_trial = 0

start = time.time()

for i, coords_path in enumerate(DATABASE["filename"]): 
    coords_path = Path(coords_path)
    print(f"\n[{i}/{len(DATABASE)}]")

    trial_name = coords_path.stem.replace('pred_results_', '')  
    session_name = coords_path.parent.stem 
    print(trial_name)

    # compute luminosity
    luminosity_dir = cfg.paths.luminosity / session_name 
    luminosity_dir.mkdir(parents=True, exist_ok=True)

    html_path = luminosity_dir / f"luminosity_{trial_name}.html"
    luminosity_path = luminosity_dir / f"luminosity_{trial_name}.csv"
    clip_path = cfg.paths.raw_clips / session_name / f"{trial_name}.mp4"
    verify_exist(clip_path)


    # ------------------------------------ get luminosities info + save them as csv ---------------------------------------

    camera_view = "left" if is_left_view(trial_name) else "right"

    # get annotation  number
    with open("./annotation_rules.yaml") as f:
        annotation_rules = yaml.safe_load(f)

    annotation_meta = {
        "condition": get_condition(trial_name), 
        "view": camera_view,
        "month": get_date(trial_name).month,
    }
    label_studio_annotation = match_rule(annotation_meta, annotation_rules)


    # luminosity has already been compute, just open the csv
    if luminosity_path.exists() : 
        print("opening luminosity")
        luminosities = pd.read_csv(luminosity_path)

        if len(luminosities) < 10 : 
            continue

        luminosities.columns = luminosities.iloc[0]      # use first row as column names
        luminosities = luminosities.drop(0).reset_index(drop=True) # remove useless row
        luminosities = luminosities[luminosities.iloc[:, 0] != 't']
    
    else : 
        luminosities: pd.DataFrame = get_luminosity(annotation_num=label_studio_annotation,        
                                                    video_path= clip_path,
                                                    fig_output_path= html_path if luminosity_dir not in dirs else None,
                                                    csv_ouput_path = luminosity_path,
                                                    max_n_frames=None,
                                                    label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                                    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
                                                    )
        
        if len(luminosities) < 10 : 
            continue

        # clean luminosities dataframe
        luminosities.columns = luminosities.columns.droplevel(0)        # columns = LED_1 ...
        luminosities = luminosities.drop([1]).reset_index(drop=True)    # remove useless row

    if luminosity_dir not in dirs : 
        dirs.add(luminosity_dir)


    # ------------------------------------ prepare list per protocoles ---------------------------------------
    
    # get led info
    cue_type = define_cue_type(luminosities["LED_1"])
    time_pad_off = get_time_led_state(luminosity_path, luminosities, cfg.task_pad, "OFF", min_duration=10,  in_sec=True)
    time_laser_on = get_time_led_state(luminosity_path, luminosities, "LED_4", "ON", in_sec=True)
    time_reward = get_time_led_state(luminosity_path, luminosities, "LED_5", "ON", in_sec=True)
    laser_state = "LaserOn" if time_laser_on is not None else "LaserOff"

    # ------------------------------------build metadata + save them as yaml ---------------------------------------

    if (camera_view == "left" and cue_type == "CueL2") or \
       (camera_view == "right" and cue_type == "CueL1") : 
        print(f"Camera: {camera_view} | cue: {cue_type} ! Not compatible\n")
        n_wrong_trial += 1
        continue

    TRIAL_METADATA: dict = init_metadata(coords_path,
                                        luminosity_path,
                                        clip_path)
    
    TRIAL_METADATA["label_studio_annotation"] = label_studio_annotation
    TRIAL_METADATA["camera_view"] = camera_view
    TRIAL_METADATA["laser_state"] = laser_state
    TRIAL_METADATA["cue_type"] = cue_type
    TRIAL_METADATA["pad_off"] = time_pad_off
    TRIAL_METADATA["laser_on"] = time_laser_on
    TRIAL_METADATA["reward"] = time_reward

    filename = make_name_by_condition(f"{session_name}_{trial_name}", laser_state)

    if not filename in ALL_METADATA.keys() : 
        ALL_METADATA[filename] = [TRIAL_METADATA]
    else : 
        ALL_METADATA[filename].append(TRIAL_METADATA) 

    yaml_path: Path =  clip_path.parent / f"{trial_name}.yaml"

    with open((yaml_path) , "w") as f : 
        yaml.safe_dump(TRIAL_METADATA, f)

    
end = time.time()
process_time = (end - start) / 60 # min

n_trial = 0 

print()
print("="*50)
print(f".joblib files Outputs :")
for filename, meta in ALL_METADATA.items() : 
    joblib.dump(meta, output_dir / f"{filename}.joblib")
    print(f"  {filename} : {len(meta)}")
    n_trial += len(meta)

print(f"Number of joblib files generated: {len(ALL_METADATA)}")
print(f"Number of trials processed: {n_trial}")
print(f"Number of trial not processed because of the camera and cue compatibility: {n_wrong_trial}")
print(f"Number of all orginal prediction: {len(DATABASE)}")
print(f"Processing time: {process_time:.1f} min")
print("Done !")

# show metadata report
print("\nVisualisation of the proportion of each experimental condition\n")
fig = metadata_report(cfg.paths.raw_clips, output_dir, show_noCue=False)
fig.write_html(str(cfg.paths.rat_root / f"{cfg.rat_name}_experimental_condition_proportion.html"))

print("\nPlotting likelihood distribution of all bodyparts\n")
plot_likelihood_distribution(cfg, list((cfg.paths.raw_clips).rglob("*.yaml")))
