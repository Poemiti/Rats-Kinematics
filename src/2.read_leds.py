#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import operator
import sys

from rats_kinematics_utils.file_management import is_left_view, get_clip_number
from rats_kinematics_utils.led_detection import get_luminosity, rename_file, define_cue_type, led_state, remove_file
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, print_analysis_info


# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Reading leads")


RAT_NAME = cfg.rat_name
DATABASE = load_database(cfg.paths.clips / RAT_NAME, cfg.paths.database, "video")

RAT_NAME = DATABASE['rat_name'][0]


# ------------------------------------- does luminosity has already been compute -----------------------------

noCue_files = []
already_renamed = []
files_to_rename = []

for video_path in DATABASE["filename"]:

    video_path = Path(video_path)

    if "LaserOn" in video_path.stem \
    or "LaserOff" in video_path.stem : 
        already_renamed.append(video_path)
    else : 
        files_to_rename.append(video_path)

    if "NoCue" in video_path.stem : 
        noCue_files.append(video_path)

print(f"Number of trials already renamed (luminosity has been computed): {len(already_renamed)}")
print(f"Number of trials where 'NoCue' was detected: {len(noCue_files)}")
print(f"Number of trials that needs to be processed: {len(files_to_rename)}")

if len(already_renamed) > 0 : 
    res = input("\nDo you want to rename the trials that have already been computed ? (y/n): ")

    if res == "y" : 
        files_to_rename.extend(already_renamed)

    elif res == "n" :
        if len(noCue_files) > 0 : 
            res = input("Do you want to rename the trial with 'NoCue' detected ? (y/n): ")

            if res == "y" : 
                files_to_rename.extend(noCue_files)

    else : 
        raise ValueError(f"[ERROR] '{res}' is not a valid answer, it can only be 'y' or 'n'!")

if len(files_to_rename) == 0 :
    print("\nNo trials to process, stop !")
    sys.exit()

APPLY_RENAME = input("\nThis script rename files, do you want to apply the renaming ? (y/n) : ")

print()
print("="*50)

print(f"Launching luminosity analysis on {len(files_to_rename)} files!")
print("RENAMING will be apply !") if APPLY_RENAME == "y" else print("NO renaming will be apply !")

print("="*50)

# ------------------------------------ get luminosity + classify ---------------------------------------

dirs = set()

for i, video_path in enumerate(DATABASE["filename"].iloc[:]) : 
    video_path = Path(video_path)

    print(f"\n[{i+1}/{len(DATABASE)}]")
    print(f"Getting luminosity of {video_path}\n")

    trial_name = video_path.parent.stem
    clip_number = get_clip_number(video_path.stem)

    output_dir = cfg.paths.luminosity / RAT_NAME / trial_name # get the folder name
    output_dir.mkdir(parents=True, exist_ok=True)

    html_output_path = output_dir / f"luminosity_{trial_name}.html"
    csv_output_path = output_dir / f"luminosity_{trial_name}.csv"

    # choose annotation number (label_studio) based on the view (voir readme)

    if is_left_view(str(video_path.stem)) : 
        label_studio_annotation = 1814
    else : 
        label_studio_annotation = 1811
    print(f"label studio annotation : {label_studio_annotation}\n")

    luminosities: pd.DataFrame = get_luminosity(annotation_num=label_studio_annotation,        
                                                video_path= video_path,
                                                fig_output_path= html_output_path if output_dir not in dirs else None,
                                                csv_ouput_path = csv_output_path,
                                                max_n_frames=None,
                                                label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
                                                )
    
    if output_dir not in dirs : 
        dirs.add(output_dir)
        files_to_rename.append(html_output_path)

    # get the trajectory prediction csv path for renaming
    trajectory_csv_dir = cfg.paths.coords / RAT_NAME / trial_name
    trajectory_csv_path = trajectory_csv_dir / f"pred_results_{video_path.stem}.csv"

    if len(luminosities) < 10 : 
        print("Trial clip is too short ! this video is being deleted")
        for path in [video_path, trajectory_csv_path, csv_output_path] : 
            remove_file(path)
        continue

    if not trajectory_csv_path.exists() :
        print(f"   ! video path : {video_path}")
        print(f"   ! trajectory does not exist : {trajectory_csv_path}")
        continue

    if "LaserOff" in video_path.stem or "LaserOn" in video_path.stem:
        print("already done!")
        files_to_rename.append(csv_output_path)
    else : 
        files_to_rename.append(video_path, csv_output_path, trajectory_csv_path)

    # clean dataframe
    luminosities.columns = luminosities.columns.droplevel(0) # columns = LED_1 ...
    luminosities = luminosities.drop([1]).reset_index(drop=True)  # remove useless row

    cue_type = define_cue_type(luminosities["LED_1"])
    led_on, _ = led_state(luminosities["LED_4"],
                          min_duration=5,           # beta ~ 5 consecutive frames 
                          comparator=operator.gt)   # gt = greater
    
    print("LED INFO :")
    print(f"  cue type: {cue_type}")
    print(f"  opto laser ON: {led_on}")

    # rename : original clip, trajectory csv, luminosity outputs (csv + html)
    for path in files_to_rename : 
        rename_file(path,
                    laser_on=led_on,
                    new_cue=cue_type, 
                    apply_rename=APPLY_RENAME)   
        
print("Done !")