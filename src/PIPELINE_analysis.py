#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sort_files import is_video, classify_file
from trajectory_analysis import plot_bodyparts_trajectories, plot_stacked_trajectories, plot_average_trajectories
from video_annotation import annotate_single_bodypart

# ------------------------------------ setup parameters ---------------------------------------

RAT_NAME = "#517"
BODYPART = ["left_hand"]
SINGLE_TRAJ = True
THRESHOLD = 0.5
VIEW = "H001"

print(f"\nTrajectory analysis parameters : ")
print(f"  Rat analysed : {RAT_NAME}")
print(f"  Bodypart analysed : {BODYPART}")
print(f"  Threshold set for point selection : {THRESHOLD}")
print(f"  View selected : {VIEW}")
print(f"  Save every clip trajectory as single files ? : {SINGLE_TRAJ}\n")


# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
PREDICTION_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database"
# LUMINOSITY_DIR = GENERATED_DATA_DIR / "luminosity" 

OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "trajectory_figures" / RAT_NAME

OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)


# ------------------------------------ classify csv results ---------------------------------------


sorted_videos = []
ct_video = 0
for root, dirs, files in os.walk(PREDICTION_DIR): 
    for name in files: 
        classify_file(os.path.join(root, name), sorted_videos) 
        ct_video += 1

DATABASE = pd.DataFrame(sorted_videos)
DATABASE.to_csv(DATABASE_DIR / f"pred_database_rat{RAT_NAME}.csv")

DATABASE = DATABASE[DATABASE["view"] == VIEW]

# ------------------------------------ plot trajectory ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 1

for csv_path in DATABASE["filename"] : 
    
    if COUNTER >= COUNTER_LIMIT : 
        break

    csv_path = Path(csv_path)
    csv_dir = csv_path.parent

    print(f"\nAnalysing trajectory of video : {csv_dir}\n ")
    stacked_start = time.perf_counter()

    output_fig_dir = OUTPUT_TRAJECTORY_PATH / csv_dir.stem 
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    output_stacked_traj = output_fig_dir / f"trajectory_stacked.png"
    output_mean_traj = output_fig_dir / f"trajectory_average.png"

    plot_stacked_trajectories(csv_dir= csv_dir , 
                            output_fig_path= output_stacked_traj,
                            bodyparts= BODYPART, 
                            invert_y=True,
                            show=False,
                            threshold=THRESHOLD)
    stacked_end = time.perf_counter()


    if SINGLE_TRAJ : 
        

        for csv_path in csv_dir.glob("*.csv") : 

            # -------------------------------------------- figure trajectory ----------------------------------------------
            
            clip_name = csv_path.stem[-7:]  # clip_XX

            print(f"\nMaking figures for : {clip_name}")
            single_start = time.perf_counter()

            output_traj_dir = output_fig_dir / "trajectory_per_clip" 
            output_traj_path = output_fig_dir / "trajectory_per_clip" / f"trajectory_{clip_name}.png"
            output_traj_dir.mkdir(parents=True, exist_ok=True)

            bodyparts_point = pd.read_csv(csv_path)

            ax = plot_bodyparts_trajectories(csv_path= csv_path, 
                                            bodyparts= BODYPART, 
                                            invert_y=True,
                                            threshold=THRESHOLD)
            
            ax.set_title(f"Trajectories of {csv_path.stem[-7:]}")
            ax.legend()

            fig = ax.figure
            fig.savefig(output_traj_path)

            plt.close(fig)
            single_end = time.perf_counter()

            # -------------------------------------------- video trajectory ----------------------------------------------

            output_annotated_clip_dir = output_fig_dir / "annotated_clips"
            output_annotated_clip_dir.mkdir(parents=True, exist_ok=True)

            anot_start = time.perf_counter()
            annotate_single_bodypart(video_path=CLIP_DIR / csv_dir.stem / f"{clip_name}.mp4",
                                    csv_path=PREDICTION_DIR / csv_dir.stem / f"pred_results_{clip_name}.csv",
                                    output_path=output_annotated_clip_dir / f"annotated_{clip_name}.mp4",
                                    bodypart_name=BODYPART[0],
                                    radius=5,
                                    likelihood_threshold= THRESHOLD)
            anot_end = time.perf_counter()


        

    n_csv = len(list(csv_dir.glob("*.csv")))
    print(f"\nPerfomance : ")
    print(f"  Time for stacked trajectory : {(stacked_end - stacked_start):.02f} s")
    print(f"  Time for every clip trajectory ({n_csv} files): {(single_end - single_start):.02f} s")
    print(f"  Time for clip annotation ({n_csv} files): {(anot_end - anot_start):.02f} s\n")

    # Perfomance : 
    #   Time for stacked trajectory : 0.75 s
    #   Time for every clip trajectory (44 files): 0.24 s
    #   Time for clip annotation (44 files): 0.77 s

            
    COUNTER += 1




