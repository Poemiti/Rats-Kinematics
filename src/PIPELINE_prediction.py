#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import os
from colorama import Back, Style, init
from deeplabcut.pose_estimation_pytorch import set_load_weights_only


from utils.file_management import is_video, classify_video
from utils.split_video import split_video
from utils.dlc_prediction import dlc_predict_Julien, annotate_video_from_csv

# Disable "weights only" before analyzing
set_load_weights_only(False)

init()

# ------------------------------------ setup path ---------------------------------------

INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
MODEL_PATH = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/DLC-project-2025-06-18")

GENERATED_DATA_DIR = Path("../data")
GENERATED_VIDEOS_DIR = GENERATED_DATA_DIR / "clips"

TEMPORARY_PATH = Path("../data/temporary")
DATABASE_DIR = GENERATED_DATA_DIR / "database"

GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
TEMPORARY_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------ setup parameters ---------------------------------------

DURATION = 12.5  # sec
RATS_NAME = "#517"
PRED_LIKELIHOOD = 0.5

# ------------------------------------ classify videos ---------------------------------------

print(Back.BLUE + "Creating (or updating) database ..." + Style.RESET_ALL)

sorted_videos = []
ct_video = 0
for root, dirs, files in os.walk(INPUT_VIDEO_DIR): 
    for name in files: 
        if is_video(name): 
            classify_video(os.path.join(root, name), sorted_videos) 
            ct_video += 1

DATABASE = pd.DataFrame(sorted_videos)
DATABASE = DATABASE[DATABASE["rat_type"] != "Unknown"]
DATABASE.to_csv(DATABASE_DIR / "database.csv")

DATABASE = DATABASE[DATABASE["rat_name"] == RATS_NAME]

print(Back.BLUE + f"Nombre total de vidéo : {ct_video}" + Style.RESET_ALL)
print(Back.BLUE + f"Nombre de video collecté : {len(DATABASE)}" + Style.RESET_ALL)

# ------------------------------------ loop ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 999

for video_path in DATABASE["filename"]: 

    if COUNTER > COUNTER_LIMIT : 
        break

    video_path = Path(video_path) 
    output_clips_dir = GENERATED_VIDEOS_DIR / video_path.stem 
    
    print(Back.BLUE + f"\nSplitting video : {video_path.stem}\n" + Style.RESET_ALL)

    split_video(input_path= video_path, 
                output_path= output_clips_dir, 
                CLIP_DURATION= DURATION)
    
    
    OUTPUT_H5_PATH = GENERATED_DATA_DIR / "dlc_results" / video_path.stem
    OUTPUT_CSV_PATH = GENERATED_DATA_DIR / "csv_results" / video_path.stem
    OUTPUT_ANNOTATED_CLIP_PATH = GENERATED_DATA_DIR / "video_annotation" / video_path.stem

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANNOTATED_CLIP_PATH.mkdir(parents=True, exist_ok=True)


    for clip_path in output_clips_dir.iterdir() :

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"
        annotated_clip_path = OUTPUT_ANNOTATED_CLIP_PATH / f"annotated_csv_{clip_path.stem}.mp4"

        print(Back.BLUE + f"\nPrediction of clip : {clip_path}\n" + Style.RESET_ALL)

        dlc_predict_Julien(
            model_path=MODEL_PATH,
            video_path=clip_path,
            output_csv_path=csv_path,
        )

        print(Back.BLUE + f"\Annotation of clip : {clip_path}\n" + Style.RESET_ALL)


    COUNTER += 1















