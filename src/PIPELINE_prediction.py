#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import os
from deeplabcut.pose_estimation_pytorch import set_load_weights_only


from utils.file_management import make_database, is_video
from utils.split_video import split_video
from utils.dlc_prediction import dlc_predict_Julien
from utils.database_filter import Model, View, Controller

# Disable "weights only" before analyzing
set_load_weights_only(False)


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
PRED_LIKELIHOOD = 0.5

# ------------------------------------ make database ---------------------------------------

raw_database = make_database(INPUT_VIDEO_DIR, is_video)

model = Model(raw_database, DATABASE_DIR)
view = View()
controller = Controller(model, view)
view.mainloop()

DATABASE = controller.filtered_dataset.reset_index(drop=True)
print(DATABASE)

# save database
if controller.dataset_name.get() :
    dataset_name = f"{controller.dataset_name.get().strip()}.csv"
    DATABASE.to_csv(DATABASE_DIR / dataset_name)
    print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")

print(f"\nNumber of files in database : {len(DATABASE)}")

RAT_NAME = DATABASE['rat_name'][0]

# ------------------------------------ loop ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 999

for video_path in DATABASE["filename"]: 

    if COUNTER > COUNTER_LIMIT : 
        break

    video_path = Path(video_path) 
    output_clips_dir = GENERATED_VIDEOS_DIR / video_path.stem 
    
    # ----------------------------------------------- video splitting --------------------------------------------------

    print(f"\nSplitting video : {video_path.stem}\n")

    split_video(input_path= video_path, 
                output_path= output_clips_dir, 
                CLIP_DURATION= DURATION)
    
    
    OUTPUT_H5_PATH = GENERATED_DATA_DIR / "dlc_results" / video_path.stem
    OUTPUT_CSV_PATH = GENERATED_DATA_DIR / "csv_results" / video_path.stem

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------- prediction --------------------------------------------------

    for clip_path in output_clips_dir.iterdir() :

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"

        print(f"\nPrediction of clip : {clip_path}\n")

        dlc_predict_Julien(
            model_path=MODEL_PATH,
            video_path=clip_path,
            output_csv_path=csv_path,
        )

    COUNTER += 1















