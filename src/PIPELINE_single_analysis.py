#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_bodyparts_trajectories, plot_stacked_trajectories, plot_average_trajectories
from utils.video_annotation import annotate_single_bodypart
from utils.trajectory_metrics import Trajectory, create_trajectory_object, plot_metric_time
from utils.led_detection import is_led_on

THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left' # or 'right'

# choose which function to use
SINGLE_TRAJ = True
ANNOT_CLIP = False
METRICS = True

# define m per pixel
if view == 'left' : 
    frame_width_m = 0.87 # meters
else : 
    frame_width_m = 0.83 # meters
frame_width_px = 512 # pixel
M_PER_PIXEL = frame_width_m / frame_width_px

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
PREDICTION_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database" 

# ------------------------------------ make database ---------------------------------------

raw_database = make_database(PREDICTION_DIR, is_csv)

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

OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "analysis_results" / RAT_NAME
OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------ output file directory preparation ---------------------------------------


output_fig_dir = OUTPUT_TRAJECTORY_PATH / Path(make_directory_name(Path(DATABASE["filename"][0]).stem))
output_fig_dir.mkdir(parents=True, exist_ok=True)

print(f"File will be stored in {output_fig_dir}")

csv_list = [Path(path) for path in DATABASE["filename"]]
csv_list = csv_list

for path in csv_list : 
    verify_exist(path)

print(f"\nAnalysing trajectory of video with these setting: ")
print([val for val in str(output_fig_dir.name).split("_")])

#############################################################################################################


metrics : list[dict] = []

for i, csv_path in enumerate(csv_list) : 
    
    verify_exist(csv_path)

    print(f"\n[{i+1}/{len(csv_list)}]")
    print(f"Working on : {csv_path}")
    

    if SINGLE_TRAJ : 
        # ------------------------------------ plot single trajectory + video annotation  ---------------------------------------
        
        print("Making single figures ...")

        output_traj_dir = output_fig_dir / "trajectory_per_clip" 
        output_traj_dir.mkdir(parents=True, exist_ok=True)
        output_traj_path = output_traj_dir / f"trajectory_{csv_path.stem.replace('_pred_results', '')}.png"

        ax = plot_bodyparts_trajectories(csv_path= csv_path, 
                                        bodyparts= [BODYPART], 
                                        invert_y=True,
                                        threshold=THRESHOLD)
        
        ax.set_title(f"Trajectories of \n{csv_path.name}")
        ax.legend()

        fig = ax.figure
        fig.savefig(output_traj_path)

        plt.close(fig)

    if ANNOT_CLIP :
        # --------------------------------------- single bodypart trajectory annotation ----------------------------------------------

        print(f"Making video annotation ...")
        output_annotated_clip_dir = output_fig_dir / "annotated_clips"
        output_annotated_clip_dir.mkdir(parents=True, exist_ok=True)

        input_clip_path = CLIP_DIR / csv_path.parent.stem / f"{csv_path.stem.replace('_pred_results', '')}.mp4"
        verify_exist(input_clip_path)
        
        annotate_single_bodypart(video_path=input_clip_path,
                                csv_path=csv_path,
                                output_path=output_annotated_clip_dir / f"annotated_{csv_path.stem.replace('_pred_results', '')}.mp4",
                                bodypart_name=BODYPART,
                                radius=5,
                                likelihood_threshold= THRESHOLD)

    if METRICS : 
        # --------------------------------------- compute metrics ----------------------------------------------

        print("Metrics measurement ...")

        Traj: Trajectory = create_trajectory_object(coords_path=csv_path,
                                                    bodypart=BODYPART,
                                                    threshold=THRESHOLD,
                                                    m_per_pixel=M_PER_PIXEL)
        
        metrics.append({"file" : csv_path,
                        "distance" : Traj.distance(),
                        "velocity" : Traj.mean_velocity()
                        })

        output_velo_dir = output_fig_dir / "velocity_per_clip" 
        output_acc_dir = output_fig_dir / "acceleration_per_clip" 
        output_velo_dir.mkdir(parents=True, exist_ok=True)
        output_acc_dir.mkdir(parents=True, exist_ok=True)

        output_velo_path = output_velo_dir / f"velocity_{csv_path.stem.replace('_pred_results', '')}.csv"
        output_acc_path = output_acc_dir / f"acceleration_{csv_path.stem.replace('_pred_results', '')}.csv"

        # compute metrics
        instant_velo = Traj.instant_velocity()
        acceleration = Traj.acceleration()
        y_position = Traj.reaching_coords[["y"]]    # _scale(Traj.reaching_coords[["y"]])

        # save computed metrics
        instant_velo.to_csv(output_velo_path, index=False)
        acceleration.to_csv(output_acc_path, index=False)

        # ---------------------------------------- plot metrics ---------------------------------------

        print(f"Plotting velocity, acceleration and position over time...")

        output_fig_metric_dir = output_fig_dir / "metric_over_time" 
        output_fig_metric_dir.mkdir(parents=True, exist_ok=True)

        output_fig_metric_path = output_fig_metric_dir / f"metricOverTime_{csv_path.stem.replace('_pred_results', '')}.png"


        # get the time when the laser is on
        luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
        verify_exist(luminosity_path)

        luminosities = pd.read_csv(luminosity_path)
        luminosities.columns = luminosities.iloc[0]      # use first row as column names
        luminosities = luminosities.drop(0).reset_index(drop=True) # remove useless row
        luminosities = luminosities[luminosities.iloc[:, 0] != 't']

        _, frame_laser_on = is_led_on(luminosities["LED_4"])
        if frame_laser_on : 
            time_laser_on = frame_laser_on / 125
            # time_laser_on = float(luminosities["led_name"].iloc[frame_laser_on])
            print(f"Laser one at {time_laser_on} sec, {frame_laser_on} frame")
        else : 
            time_laser_on = frame_laser_on
        
        # plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plot_metric_time(instant_velo,
                        ax = axs[0], 
                        laser_on=time_laser_on,
                        color="red")
        axs[0].set_title("Velocity over time of a trial", color="red")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Velocity (m.s$^{-1}$)")
        
        plot_metric_time(acceleration, 
                        ax = axs[1],
                        laser_on=time_laser_on,
                        color="green")
        axs[1].set_title("Acceleration over time of a trial", color="green")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Acceleration (m.s$^{-2}$)")

        plot_metric_time(y_position, 
                        ax = axs[2],
                        laser_on=time_laser_on,
                        color="blue", 
                        y_invert=True)
        axs[2].set_title("Height position over time of a trial", color="blue")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Position (pixel)")

        plt.savefig(output_fig_metric_path)
        plt.close(fig)


# save computed metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(output_fig_dir / "metrics_per_clips.csv", index=False)

print("\nDone !")


