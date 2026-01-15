#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import os
import sys
from tqdm import tqdm

from video_database import is_video, classify_video
from split_video_by_trial import split_video
from dlc_prediction import dlc_predict_Julien, annotate_video_from_csv

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

# ------------------------------------ classify videos ---------------------------------------

print("Creating (or updating) database ...")

sorted_videos = []
ct_video = 0

all_files = []
for root, _, files in os.walk(INPUT_VIDEO_DIR):
    for name in files:
        all_files.append(os.path.join(root, name))

for file_path in tqdm(all_files, desc="Scanning videos", unit="file", file=sys.stderr):
    if is_video(file_path):
        classify_video(file_path, sorted_videos)
        ct_video += 1

DATABASE = pd.DataFrame(sorted_videos)
DATABASE = DATABASE[DATABASE["rat_type"] != "Unknown"]
DATABASE.to_csv(DATABASE_DIR / "database.csv")

DATABASE = DATABASE[DATABASE["rat_name"] == RATS_NAME]

print(f"Nombre total de vidéo : {ct_video}")
print(f"Nombre de video collecté : {len(DATABASE)}")

# ------------------------------------ split videos ---------------------------------------

for video_path in tqdm(
    DATABASE["filename"],
    desc="Splitting videos",
    unit="video",
    file=sys.stderr,
):
    video_path = Path(video_path)

    output_clips_path = GENERATED_VIDEOS_DIR / video_path.stem
    split_video(
        input_path=video_path,
        output_path=output_clips_path,
        CLIP_DURATION=DURATION,
    )

# ------------------------------------ DLC prediction ---------------------------------------

video_dirs = [p for p in GENERATED_VIDEOS_DIR.iterdir() if p.is_dir()]

for video_dir in tqdm(
    video_dirs,
    desc="DLC per video",
    unit="video",
    file=sys.stderr,
):

    OUTPUT_H5_PATH = GENERATED_DATA_DIR / "dlc_results" / video_dir.stem
    OUTPUT_CSV_PATH = GENERATED_DATA_DIR / "csv_results" / video_dir.stem
    OUTPUT_ANNOTATED_CLIP_PATH = GENERATED_DATA_DIR / "video_annotation" / video_dir.stem

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANNOTATED_CLIP_PATH.mkdir(parents=True, exist_ok=True)

    clips = [p for p in video_dir.iterdir() if p.is_file() and p.suffix == ".mp4"]

    for clip_path in tqdm(
        clips,
        desc=f"{video_dir.name}",
        unit="clip",
        leave=False,
        file=sys.stderr,
    ):

        csv_path = OUTPUT_CSV_PATH / f"pred_results_{clip_path.stem}.csv"
        annotated_clip_path = OUTPUT_ANNOTATED_CLIP_PATH / f"annotated_csv_{clip_path.stem}.mp4"

        dlc_predict_Julien(
            model_path=MODEL_PATH,
            video_path=clip_path,
            output_csv_path=csv_path,
        )

        annotate_video_from_csv(
            video_path=clip_path,
            csv_path=csv_path,
            output_path=annotated_clip_path,
            radius=5,
            likelihood_threshold=0.5,
        )
