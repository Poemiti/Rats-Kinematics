#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from rats_kinematics_utils.file_management import is_left_view
from rats_kinematics_utils.led_detection import get_luminosity, rename_file, define_cue_type, is_led_on, remove_file
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, print_analysis_info


# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Reading leads")


RAT_NAME = cfg.rat_name
DATABASE = load_database(cfg.paths.clips / RAT_NAME, cfg.paths.database, "video")

RAT_NAME = DATABASE['rat_name'][0]

# ------------------------------------ get luminosity + classify ---------------------------------------


for i, video_path in enumerate(DATABASE["filename"].iloc[:]) : 
    video_path = Path(video_path)

    print(f"\n[{i+1}/{len(DATABASE)}]")
    print(f"Getting luminosity of {video_path}\n")

    if "LaserOff" in video_path.stem or "LaserOn" in video_path.stem:
        print("already done!")
        continue

    output_dir = cfg.paths.luminosity / RAT_NAME / video_path.parent.stem  # get the folder name
    output_dir.mkdir(parents=True, exist_ok=True)

    html_output_path = output_dir / f"luminosity_{video_path.stem}.html"
    csv_output_path = output_dir / f"luminosity_{video_path.stem}.csv"


    # choose annotation number (label_studio) based on the view (voir readme)

    if is_left_view(str(video_path.stem)) : 
        label_studio_annotation = 1814
    else : 
        label_studio_annotation = 1811
    print(f"label studio annotation : {label_studio_annotation}\n")

    luminosities: pd.DataFrame = get_luminosity(annotation_num=label_studio_annotation,        
                                                video_path= video_path,
                                                fig_output_path= html_output_path,
                                                csv_ouput_path = csv_output_path,
                                                max_n_frames=None,
                                                label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
                                                )
    # get the trajectory prediction csv path for renaming
    clip_number = str(video_path.stem)[-8:]
    trajectory_csv_dir = cfg.paths.coords / RAT_NAME / str(video_path.stem)[:-8]
    trajectory_csv_path = trajectory_csv_dir / f"pred_results_{video_path.stem}.csv"


    if len(luminosities) < 10 : 
        print("Video is too short ! this video is being deleted")
        for path in [video_path, trajectory_csv_path, csv_output_path, html_output_path] : 
            remove_file(path)
        continue

    # if not verify_exist(trajectory_csv_path) : 
    if not trajectory_csv_path.exists() :
        print(f"   ! video path : {video_path}")
        print(f"   ! trajectory does not exist : {trajectory_csv_path}")
        continue

    # clean dataframe
    luminosities.columns = luminosities.columns.droplevel(0) # columns = LED_1 ...
    luminosities = luminosities.drop([1]).reset_index(drop=True)  # remove useless row

    cue_type = define_cue_type(luminosities["LED_1"])
    led_on, _ = is_led_on(luminosities["LED_4"])

    # rename : original clip, trajectory csv, luminosity outputs (csv + html)
    for path in [video_path, trajectory_csv_path, csv_output_path, html_output_path] : 
        rename_file(path,
                    laser_on=led_on,
                    new_cue=cue_type)
        
print("Done !")