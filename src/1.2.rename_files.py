#!/usr/bin/env python

from pathlib import Path

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database


# ------------------------------------ setup  ---------------------------------------

cfg = load_config()
DATABASE = load_database(cfg.paths.raw_videos, cfg.paths.database, "video")

RAT_NAME = DATABASE['rat_name'][0]


#################### for clips 

# for i, video_path in enumerate(DATABASE["filename"].iloc[:]): 
#     video_path = Path(video_path) 

#     print(f"\n\n[{i+1}/{len(DATABASE)}]")
#     print(f"Renaming of {video_path}\n")


#     old_clips_dir = cfg.paths.clips / RAT_NAME / video_path.stem 
#     new_clips_dir = cfg.paths.clips / RAT_NAME / video_path.parent.stem

#     print(f"clip dir : {old_clips_dir} -> \n{new_clips_dir}")

#     if not old_clips_dir.exists():
#             print("Old folder does not exist. Skipping.")
#             continue

#     new_clips_dir.mkdir(parents=True, exist_ok=True)

#     for old_clip_path in old_clips_dir.iterdir() : 

#         if not old_clip_path.is_file():
#             continue

#         clip_number = old_clip_path.stem[-7:]
#         new_clip_name = f"{video_path.parent.stem}_{clip_number}.mp4"
#         new_clip_path = new_clips_dir / new_clip_name

#         print(f"\nFile : {old_clip_path} ->\n{new_clip_path}")

#         old_clip_path.rename(new_clip_path)

#     try:
#         old_clips_dir.rmdir()
#         print(f"\nRemoved empty folder: {old_clips_dir}")
#     except OSError:
#         print(f"\nFolder not empty or cannot remove: {old_clips_dir}")


############## for csv

for i, video_path in enumerate(DATABASE["filename"].iloc[:]): 
    video_path = Path(video_path) 

    print(f"\n\n[{i+1}/{len(DATABASE)}]")
    print(f"Renaming of {video_path}\n")


    old_csv_dir = cfg.paths.coords / RAT_NAME / video_path.stem 
    new_csv_dir = cfg.paths.coords / RAT_NAME / video_path.parent.stem


    print(f"clip dir : {old_csv_dir} -> \n{new_csv_dir}")

    if not old_csv_dir.exists():
            print("Old folder does not exist. Skipping.")
            continue

    new_csv_dir.mkdir(parents=True, exist_ok=True)

    for old_csv_path in old_csv_dir.iterdir() : 

        if not old_csv_path.is_file():
            continue

        clip_number = old_csv_path.stem[-7:]
        new_clip_name = f"pred_results_{video_path.parent.stem}_{clip_number}.csv"
        new_clip_path = new_csv_dir / new_clip_name

        print(f"\nFile : {old_csv_path} ->\n{new_clip_path}")

        old_csv_path.rename(new_clip_path)

    try:
        old_csv_dir.rmdir()
        print(f"\nRemoved empty folder: {old_csv_dir}")
    except OSError:
        print(f"\nFolder not empty or cannot remove: {old_csv_dir}")













