#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_single_bodypart_trajectories, open_clean_csv
from utils.trajectory_metrics import Trajectory, create_trajectory_object, plot_metric_time, define_End_of_trajectory
from utils.led_detection import get_time_led_on, get_time_led_off

# set parameters
THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left'           # or 'right'
LEVER_POSITION = 215    # pixels
LASER_ON_TIME = 0.325   # sec

# choose which function to use
SHOW = True
STACKED_TRAJ = True
STACKED_METRICS = True

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


if STACKED_TRAJ : 
    # ------------------------------------ plot stacked trajectory ---------------------------------------
    print("\n TRAJECTORY PLOTTING \n")

    output_stacked_traj = output_fig_dir / f"trajectory_stacked.png"
    output_mean_traj = output_fig_dir / f"trajectory_average.png"
    
    fig, ax = plt.subplots(figsize=(9, 7))
    all_coords = []
    failed_trial = []

    for csv_path in csv_list:

        # get time when the PAD OFF led is ON
        luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
        verify_exist(luminosity_path)

        time_pad_off = get_time_led_off(luminosity_path, "LED_3", in_sec=True)

        if time_pad_off is None or time_pad_off+LASER_ON_TIME > 2 : # in sec
            print(f"  ! Failed trial on, Pad off at {time_pad_off}")
            failed_trial.append(csv_path)
            continue

        coords = open_clean_csv(csv_path)
        xy = coords[BODYPART]
        mask = xy["likelihood"] >= THRESHOLD
        xy_filtered = xy[mask]

        xy_filtered = xy_filtered.iloc[int(time_pad_off*125)-1 : int((time_pad_off + LASER_ON_TIME)*125)]  # in frame

        if len(xy_filtered) == 0 : 
            print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
            failed_trial.append(csv_path)
            continue

        all_coords.append(xy_filtered)

        plot_single_bodypart_trajectories(
            coords=xy_filtered,
            ax=ax,
            invert_y=True,
            color="red",
            transparancy=0.2
        )

    avg_coords = (pd.concat(all_coords, axis=1)
                    .T
                    .groupby(level=0)
                    .mean()
                    .T)

    plot_single_bodypart_trajectories(
            coords=avg_coords,
            ax=ax,
            invert_y=True,
            color="blue",
            transparancy=1
        )

    title = (
        "Trajectories across trials with settings:\n"
        f"{output_fig_dir.stem}\n"
        f"Number of trials: {len(csv_list)}"
    )

    print("\nFailed trial : ")
    print(str(path) for path in failed_trial)
    print(f"Number of failed trial : {len(failed_trial)}")

    ax.set_title(title, fontsize=12)
    fig.savefig(output_stacked_traj)

    if SHOW:
        plt.show()
    plt.close(fig)

#############################################################################################


if STACKED_METRICS : 
    # ------------------------------- plot stacked VELOCITY + average in bold -------------------------

    print("\n METRIC PLOTTING \n")

    output_stacked_velo = output_fig_dir / f"velocity_stacked.png"
    fig, axs = plt.subplots(figsize=(9, 7))
    failed_trial = []

    for csv_path in csv_list : 

        all_velocity = []
        all_laser_on = []

        # get time when the PAD OFF led is ON
        luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
        verify_exist(luminosity_path)

        time_pad_off = get_time_led_off(luminosity_path, "LED_3", in_sec=True)

        if time_pad_off is None or time_pad_off+LASER_ON_TIME > 2 : # in sec
            print(f"  ! Failed trial on, Pad off at {time_pad_off}")
            failed_trial.append(csv_path)
            continue

        coords = open_clean_csv(csv_path)
        xy = coords[BODYPART]
        mask = xy["likelihood"] >= THRESHOLD
        xy_filtered = xy[mask]
        xy_filtered = xy_filtered[["x", "y"]]

        # add a time columns in seconds
        xy_filtered["t"] = (np.arange(len(xy_filtered)) / 125)
        xy_filtered = xy_filtered.iloc[int(time_pad_off*125)-1 : int((time_pad_off + LASER_ON_TIME)*125)]  # in frame

        if len(xy_filtered) == 0 : 
            print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
            failed_trial.append(csv_path)
            continue

        # creat trajectory object for metric calculation
        Traj = Trajectory(coords=xy,
                        reaching_coords=xy_filtered, 
                        fps=125, 
                        m_per_pixel=M_PER_PIXEL)

        instant_velo : pd.DataFrame = Traj.instant_velocity()

        # get the time when the laser is on
        luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
        verify_exist(luminosity_path)

        time_laser_on = get_time_led_on(luminosity_path, "LED_4", in_sec=True)
        all_laser_on.append(time_laser_on)

        ## ploting single velocities over time
        plot_metric_time(instant_velo['velocity'],
                         time=instant_velo['t'],
                        ax = axs, 
                        laser_on=None,
                        color="green",
                        transparancy=0.2)
        
        all_velocity.append(instant_velo.set_index("t")["velocity"])

    # plotting average velocity over time
    avg_velocity = pd.concat(all_velocity, axis=1).mean(axis=1)

    if all_laser_on[0] is not None : 
        avg_laser_on = np.array(all_laser_on).mean()
    else : 
        avg_laser_on = None

    plot_metric_time(avg_velocity,
                     time = avg_velocity.index,
                    ax = axs, 
                    laser_on=avg_laser_on,
                    color="blue",
                    transparancy=1)
    
    title = (
        "Velocities across trials with settings:\n"
        f"{output_fig_dir.name}\n"
        f"Number of trials: {len(csv_list)}"
        )

    axs.set_title(title, color="blue")
    axs.set_xlabel("Time (seconds)")
    axs.set_ylabel("Velocity (m.s$^{-1}$)")

    print("\nFailed trial : ")
    print(str(path) for path in failed_trial)
    print(f"Number of failed trial : {len(failed_trial)}")

    fig.savefig(output_stacked_velo)

    if SHOW:
        plt.show()
    plt.close(fig)
