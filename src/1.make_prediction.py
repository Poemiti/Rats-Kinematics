#!/usr/bin/env python

from pathlib import Path
from deeplabcut.pose_estimation_pytorch import set_load_weights_only

from rats_kinematics_utils.split_video import split_video
from rats_kinematics_utils.dlc_prediction import dlc_predict_Julien
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database

# Disable "weights only" before analyzing
set_load_weights_only(False)


# ------------------------------------ setup  ---------------------------------------

cfg = load_config()
DATABASE = load_database(cfg.paths.raw_videos, cfg.paths.database, "video")

RAT_NAME = DATABASE['rat_name'][0]

for i, video_path in enumerate(DATABASE["filename"].iloc[:]): 
    video_path = Path(video_path) 

    print(f"\n[{i+1}/{len(DATABASE)}]")
    print(f"Prediction of {video_path}\n")


    output_clips_dir = cfg.paths.clips / RAT_NAME / video_path.stem 
    
    # ----------------------------------------------- video splitting --------------------------------------------------

    print(f"\nSplitting video : {video_path.stem}\n")

    split_video(input_path= video_path, 
                output_path= output_clips_dir, 
                CLIP_DURATION= cfg.clip_length)
    
    
    OUTPUT_H5_PATH = cfg.paths.h5 / RAT_NAME / video_path.stem
    OUTPUT_CSV_PATH = cfg.paths.coords / RAT_NAME /  video_path.stem

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------- prediction --------------------------------------------------

    for clip_path in output_clips_dir.iterdir() :

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"

        print(f"\nPrediction of clip : {clip_path}\n")

        dlc_predict_Julien(
            model_path=cfg.paths.model,
            video_path=clip_path,
            output_csv_path=csv_path,
        )















