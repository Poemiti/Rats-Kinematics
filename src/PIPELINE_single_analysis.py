#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_single_bodypart_trajectories, open_clean_csv, plot_3D_traj
from utils.video_annotation import annotate_single_bodypart
from utils.trajectory_metrics import Trajectory, plot_metric_time, define_End_of_trajectory
from utils.led_detection import get_time_led_on, get_time_led_off

# set parameters
THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left'           # or 'right'f
LEVER_POSITION = 215    # pixels
LASER_ON_TIME = 0.325   # sec

# choose which function to use
SINGLE_TRAJ = False
ANNOT_CLIP = False
METRICS = True
METRICS_PLOT = False
PLOT3D = False

# define m per pixel
if view == 'left' : 
    frame_width_m = 8.7 # cm
else : 
    frame_width_m = 8.3 # cm
frame_width_px = 512 # pixel
CM_PER_PIXEL = frame_width_m / frame_width_px

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
failed_trials : list[dict] = []

for i, csv_path in enumerate(csv_list) : 
    
    verify_exist(csv_path)

    print(f"\n[{i+1}/{len(csv_list)}]")
    print(f"Working on : {csv_path}")

# ------------------------------------ OPENING AND GET SOME VARIABLE  ---------------------------------------

    # get time when pad is ON or OFF
    luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
    verify_exist(luminosity_path)

    time_pad_off = get_time_led_off(luminosity_path, "LED_3", in_sec=True) # in sec
    time_laser_on = get_time_led_on(luminosity_path, "LED_4", in_sec=True) # in sec

    # get coords + filtration
    coords = open_clean_csv(csv_path)
    xy = coords[BODYPART].copy()
    xy = xy.assign(t=np.arange(len(xy)) / 125)
    xy_filtered = xy.loc[xy["likelihood"] >= THRESHOLD, ["x", "y", "t"]]

    # verification
    if time_pad_off is None or time_pad_off+LASER_ON_TIME > len(xy_filtered)-1 : # in sec
        print(f"  ! Failed trial on, Pad off at {time_pad_off}")
        failed_trials.append({
            "path": csv_path.as_posix(),
            "reason": f"Failed trial",
            "pad_off": time_pad_off,
        })
        continue
    
    #  pad off -> laser off coords 
    xy_filtered = xy.loc[
        (xy["t"] >= time_pad_off) &
        (xy["t"] <= time_pad_off + LASER_ON_TIME)
    ].reset_index(drop=True)

    print(f"pad off={time_pad_off}, laser on={time_laser_on}")

    # laser on -> laser off coords
    if time_laser_on: 
        xy_laserOn = xy.loc[
            (xy["t"] >= time_laser_on) &
            (xy["t"] <= time_laser_on + 0.3)
        ].reset_index(drop=True)
    else : 
        xy_laserOn = xy_filtered

    # verification
    if len(xy_filtered) == 0 : 
        print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
        failed_trials.append({
            "path": csv_path.as_posix(),
            "reason": f"Empty reaching coords",
            "pad_off": time_pad_off,
        })
        continue

    
    if SINGLE_TRAJ : 
        # ------------------------------------ plot single trajectory + video annotation  ---------------------------------------
        
        print("Making single figures ...")

        output_traj_dir = output_fig_dir / "trajectory_per_clip" 
        output_traj_dir.mkdir(parents=True, exist_ok=True)
        output_traj_path = output_traj_dir / f"trajectory_{csv_path.stem.replace('_pred_results', '')}.png"

        

        ax = plot_single_bodypart_trajectories(
                    coords=xy_filtered,
                    ax=None,
                    invert_y=True,
                    color="red",
                    transparancy=0.7,
                    marker="o"
                )
        
        ax.set_title(f"Trajectories of \n{csv_path.stem.replace('_pred_results', '')}")

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

        # creat trajectory object for metric calculation
        Traj = Trajectory(coords=xy,
                        reaching_coords=xy_filtered, 
                        laserOn_coords=xy_laserOn,
                        fps=125, 
                        cm_per_pixel=CM_PER_PIXEL)

        instant_velo : pd.DataFrame = Traj.instant_velocity()
        
        metrics.append({"file" : csv_path,
                        "distance" : Traj.distance(),
                        "velocity_avg" : Traj.mean_velocity(),
                        "velocity_peak" : Traj.peak(),
                        "toruosity" : Traj.tortuosity()
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

        # save computed metrics
        instant_velo.to_csv(output_velo_path, index=False)
        acceleration.to_csv(output_acc_path, index=False)


    if METRICS_PLOT:
        # ---------------------------------------- plot metrics ---------------------------------------

        print(f"Plotting velocity, acceleration and position over time...")

        output_fig_metric_dir = output_fig_dir / "metric_over_time" 
        output_fig_metric_dir.mkdir(parents=True, exist_ok=True)
        output_fig_metric_path = output_fig_metric_dir / f"metricOverTime_{csv_path.stem.replace('_pred_results', '')}.png"
        
        # plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plot_metric_time(metric=instant_velo["velocity"],
                         time=instant_velo['t'],
                        ax = axs[0], 
                        laser_on=time_laser_on,
                        color="red")
        axs[0].set_title("Velocity over time of a trial", color="red")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Velocity (cm.s$^{-1}$)")
        
        plot_metric_time(metric=acceleration['acceleration'], 
                         time=acceleration['t'],
                        ax = axs[1],
                        laser_on=time_laser_on,
                        color="green")
        axs[1].set_title("Acceleration over time of a trial", color="green")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Acceleration (cm.s$^{-2}$)")

        plot_metric_time(metric=xy_filtered["y"].to_numpy(), 
                         time=xy_filtered['t'].to_numpy(),
                        ax = axs[2],
                        laser_on=time_laser_on,
                        color="blue", 
                        y_invert=True)
        axs[2].set_title("Height position over time of a trial", color="blue")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Position (pixel)")

        plt.savefig(output_fig_metric_path)
        plt.close(fig)


    if PLOT3D : 
        # ----------------------------------- plot trajectory over time 3D plot ---------------------------------------

        print(f"Plotting 3D trajectory over time")

        output_fig_plo3D_dir = output_fig_dir / "plot_3D" 
        output_fig_plo3D_dir.mkdir(parents=True, exist_ok=True)
        output_fig_plo3D_path = output_fig_plo3D_dir / f"plot3D_{csv_path.stem.replace('_pred_results', '')}.png"

        ax = plot_3D_traj(coords=xy_filtered,
                          time=xy_filtered["t"],
                          laser_on=time_pad_off + 0.025,
                          ax= None,
                          color="blue",
                          transparancy=0.7,
                          y_invert=True)        

        ax.set_title(f"Trajectories over time of \n{csv_path.stem.replace('_pred_results', '')}")

        fig = ax.figure
        fig.savefig(output_fig_plo3D_path)

        plt.close(fig)


# save overall computed metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(output_fig_dir / "metrics_per_clips.csv", index=False)

print(f"\nNumber of failed trial : {len(failed_trials)}")

# save failed trial
pd.DataFrame(failed_trials).to_csv(output_fig_dir / "failed_trial.csv", index=False)

print("\nDone !")


