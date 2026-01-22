#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from utils.file_management import is_video, classify_file
from utils.trajectory_analysis import plot_bodyparts_trajectories, plot_stacked_trajectories, plot_average_trajectories
from utils.video_annotation import annotate_single_bodypart

THRESHOLD = 0.5
BODYPART = 'left_hand'
SINGLE_TRAJ = False

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
PREDICTION_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database" 

# ------------------------------------ make database ---------------------------------------

# raw_database = make_database(CLIP_DIR, is_video)

# model = Model(raw_database) # or DATABASE_PRED
# view = View()
# controller = Controller(model, view)
# view.mainloop()

# DATABASE = controller.filtered_dataset.reset_index(drop=True)
# print(DATABASE)

# # save database
# dataset_name = f"{controller.dataset_name.get().strip()}.csv"
# DATABASE.to_csv(DATABASE_DIR / dataset_name)
# print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")
# print(f"Number of files in {dataset_name} : {len(DATABASE)}")

# ------------------------------------ OR import database ---------------------------------------

DATABASE = pd.read_csv(GENERATED_DATA_DIR / "database/#517_CHR_beta_H001.csv")

RAT_NAME = DATABASE['rat_name'][0]

OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "trajectory_figures" / RAT_NAME
OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------ plot trajectory ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 10

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
                                            bodyparts= [BODYPART], 
                                            invert_y=True,
                                            threshold=THRESHOLD)
            
            ax.set_title(f"Trajectories of {csv_path.stem[-7:]}")
            ax.legend()

            fig = ax.figure
            fig.savefig(output_traj_path)

            plt.close(fig)
            single_end = time.perf_counter()

            # -------------------------------------------- single bodypart trajectory ----------------------------------------------

            output_annotated_clip_dir = output_fig_dir / "annotated_clips"
            output_annotated_clip_dir.mkdir(parents=True, exist_ok=True)

            anot_start = time.perf_counter()
            annotate_single_bodypart(video_path=CLIP_DIR / csv_dir.stem / f"{clip_name}.mp4",
                                    csv_path=PREDICTION_DIR / csv_dir.stem / f"pred_results_{clip_name}.csv",
                                    output_path=output_annotated_clip_dir / f"annotated_{clip_name}.mp4",
                                    bodypart_name=BODYPART,
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




