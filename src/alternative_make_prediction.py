#!/usr/bin/env python

from pathlib import Path
import os
import shutil
from deeplabcut.pose_estimation_pytorch import set_load_weights_only

from rats_kinematics_utils.core.video_utils import split_video
from rats_kinematics_utils.prediction.dlc_prediction import dlc_predict_Julien
from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.gui.database_filter import load_database

# lauching line : 
#            nohup python3 -u 1.make_prediction.py > main.out &
#            tail -f main.out 

# Disable "weights only" before analyzing
set_load_weights_only(False)


# ------------------------------------ setup  ---------------------------------------

cfg = load_config()
DATABASE = load_database(cfg.paths.raw_videos, cfg.paths.database, "video")
RAT_NAME = DATABASE['rat_name'][0]


right_video = [
    "Rat_#521RightHanded_20240521_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0001",
    "Rat_#521RightHanded_20240521_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0002",
    "Rat_#521RightHanded_20240521_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0003",
    "Rat_#521RightHanded_20240528_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0001",
    "Rat_#521RightHanded_20240528_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0002",
    "Rat_#521RightHanded_20240528_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0003",
    "Rat_#521RightHanded_20240528_Session1_ContiMT300_0,5mW_Laser5050_LeftHemiCHR_onlyL2RightHand_C001H002S0004"]

# remove wrong vid : 
# for vid in os.listdir(cfg.paths.clips / RAT_NAME) : 
#     item_path = os.path.join(cfg.paths.clips / RAT_NAME, vid)

#     # Check if it's a directory
#     if os.path.isdir(item_path):
#         if vid not in right_video:
#             print(f"Deleting folder: {item_path}")
#             shutil.rmtree(item_path)
#         else:
#             print(f"Keeping folder: {item_path}")


for i, video_path in enumerate(DATABASE["filename"].iloc[12:]): 

    video_path = Path(video_path) 
    video_name = video_path.parent.stem
    output_clips_dir = cfg.paths.clips / RAT_NAME / video_name 

    if "FIBER_BROKEN" in video_name : 
        print("Fiber broken, skipped")
        continue

    if video_name in right_video :
        DURATION = 12.5 # sec
    else : 
        DURATION = 8.33

    print(f"\n[{i+1}/{len(DATABASE)}]")
    print(f"Prediction of {video_name}")
    print(f"Duration: {DURATION}")

    
    # ----------------------------------------------- video splitting --------------------------------------------------

    split_video(input_path= video_path, 
                output_path= output_clips_dir, 
                CLIP_DURATION= DURATION,
                CRF=13)
    
    
    OUTPUT_H5_PATH = cfg.paths.h5 / RAT_NAME / video_name
    OUTPUT_CSV_PATH = cfg.paths.coords / RAT_NAME /  video_name

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------- prediction --------------------------------------------------

    for t, clip_path in enumerate(output_clips_dir.iterdir()) :

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"

        print(f"\nPrediction of clip : {clip_path}\n")

        dlc_predict_Julien(
            model_path=cfg.paths.model,
            video_path=clip_path,
            output_csv_path=csv_path,
        )



print("Done !")