#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import os

from video_database import is_video, classify_video
from trajectory_analysis import plot_bodyparts_trajectories, plot_stacked_trajectories, plot_average_trajectories


# ------------------------------------ setup path ---------------------------------------

INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
MODEL_PATH = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/DLC-project-2025-06-18")

GENERATED_DATA_DIR = Path("../data")
GENERATED_VIDEOS_DIR = GENERATED_DATA_DIR / "clips"

TEMPORARY_PATH = Path("../data/temporary")  # Temporary directory for dlc prediction
DATABASE_DIR = GENERATED_DATA_DIR / "database" 
    

GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
TEMPORARY_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------ setup parameters ---------------------------------------

DURATION = 12.5  # sec
RATS_NAME = "#517"


# ----------------------------------------------  classify Video by condition -------------------------------------------------

print("Creating (or updating) database ...")
ct_video = 0
sorted_videos = []
for root, dirs, files in os.walk(INPUT_VIDEO_DIR):
    for name in files:
        if is_video(name):
            classify_video(os.path.join(root, name), sorted_videos)
            ct_video += 1


DATABASE = pd.DataFrame(sorted_videos)
DATABASE = DATABASE[DATABASE["rat_type"] != "Unknown"]   # remove KO rats
DATABASE.to_csv(DATABASE_DIR / "database.csv") # saving

DATABASE = DATABASE[DATABASE["rat_name"] == RATS_NAME]

print(f"Nombre total de vidéo {ct_video}")
print(f"Nombre de video collecté : {len(DATABASE)}")
print(f"Final dataframe : \n{DATABASE}")