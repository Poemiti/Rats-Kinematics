#!/usr/bin/env python

import time
import yaml
from pathlib import Path
from deeplabcut.pose_estimation_pytorch import set_load_weights_only

from rats_kinematics_utils.split_video import split_video, verify_video
from rats_kinematics_utils.dlc_prediction import dlc_predict_Julien
from rats_kinematics_utils.config import load_config, match_rule
from rats_kinematics_utils.pipeline_maker import load_database
from rats_kinematics_utils.file_management import get_date

# lauching line : 
#            nohup python3 -u src/1.1.make_prediction.py > main.out &
#            tail -f main.out 

# Disable "weights only" before analyzing
set_load_weights_only(False)


# ------------------------------------ setup  ---------------------------------------

cfg = load_config()
DATABASE = load_database(cfg.paths.raw_videos, cfg.paths.database, "video")

RAT_NAME = DATABASE['rat_name'][0]

start = time.perf_counter()

for i, video_path in enumerate(DATABASE["filename"].iloc[:]): 
    video_path = Path(video_path) 

    if i < 21 : 
        continue

    print(f"\n[{i+1}/{len(DATABASE)}]")
    print(f"Prediction of video: {video_path}\n")


    output_clips_dir = cfg.paths.clips / RAT_NAME / video_path.stem 

    # ----------------------------------------------- get clips lenght by looking at the month --------------------------------------------------

    with open("./clip_duration_rules.yaml") as f:
        clip_duration_rules = yaml.safe_load(f)

    trial_name = video_path.stem

    meta = {"month": get_date(trial_name).month}
    clip_duration = match_rule(meta, clip_duration_rules)
    
    # ----------------------------------------------- video splitting --------------------------------------------------

    print(f"\nSplitting video : {video_path.stem}\n")

    split_video(input_path= video_path, 
                output_path= output_clips_dir, 
                CLIP_DURATION= clip_duration)
    
    
    OUTPUT_H5_PATH = cfg.paths.h5 / RAT_NAME / video_path.stem
    OUTPUT_CSV_PATH = cfg.paths.coords / RAT_NAME /  video_path.stem

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------- prediction --------------------------------------------------

    for j, clip_path in enumerate(output_clips_dir.iterdir() ):

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"

        print(f"\n[{j+1}/{i+1}/{len(DATABASE)}]")
        print(f"\nPrediction of clip : {clip_path.stem}\n")

        if not verify_video(clip_path): 
            continue

        dlc_predict_Julien(
            model_path=cfg.paths.model,
            video_path=clip_path,
            output_csv_path=csv_path,
        )

end = time.perf_counter()
process_time = end - start

print(f"Processing time : {(process_time / 60 / 60):.1f} h")
print("Done !")














