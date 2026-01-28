#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_single_bodypart_trajectories, open_clean_csv
from utils.trajectory_metrics import Trajectory, create_trajectory_object, plot_metric_time, define_StartEnd_of_trajectory
from utils.led_detection import is_led_on

THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left' # or 'right'

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

    output_stacked_traj = output_fig_dir / f"trajectory_stacked.png"
    output_mean_traj = output_fig_dir / f"trajectory_average.png"
    
    fig, ax = plt.subplots(figsize=(9, 7))
    all_coords = []

    for csv_path in csv_list:

        coords = open_clean_csv(csv_path)
        xy = coords[BODYPART]
        mask = xy["likelihood"] >= THRESHOLD
        xy_filtered = xy[mask]

        start, end = define_StartEnd_of_trajectory(xy_filtered, lever_position=210)
        xy_filtered = xy_filtered.iloc[start : end]

        print(len(xy_filtered))
        all_coords.append(xy_filtered)

        plot_single_bodypart_trajectories(
            coords=xy_filtered,
            ax=ax,
            invert_y=True,
            color="red",
            transparancy=0.2
        )

    def normalize_traj(df, n=100):
        t = df.index / df.index.max()
        t_new = np.linspace(0, 1, n)
        return (
            df.set_index(t)
            .reindex(t_new)
            .interpolate()
        )

    all_norm = [normalize_traj(df) for df in all_coords]

    avg_coords = (
        pd.concat(all_norm, axis=1)
        .T.groupby(level=0).mean().T
    )


    # avg_coords = (pd.concat(all_coords, axis=1)
    #                 .T
    #                 .groupby(level=0)
    #                 .mean()
    #                 .T)

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

    ax.set_title(title, fontsize=12)

    fig.savefig(output_stacked_traj)

    if SHOW:
        plt.show()

    plt.close(fig)

#############################################################################################


if STACKED_METRICS : 
    # ------------------------------- plot stacked VELOCITY + average in bold -------------------------

    output_stacked_velo = output_fig_dir / f"velocity_stacked.png"
    fig, axs = plt.subplots(figsize=(9, 7))

    for csv_path in csv_list : 

        all_velocity = []
        all_laser_on = []

        # get the metric : 
        Traj: Trajectory = create_trajectory_object(coords_path=csv_path,
                                                    bodypart=BODYPART,
                                                    threshold=THRESHOLD,
                                                    m_per_pixel=M_PER_PIXEL)
        instant_velo : pd.Series = Traj.instant_velocity()

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
            # print(f"Laser on at {time_laser_on} sec, {frame_laser_on} frame")
            all_laser_on.append(time_laser_on)
        else : 
            time_laser_on = frame_laser_on

        ## ploting single velocities over time
        plot_metric_time(instant_velo,
                        ax = axs, 
                        laser_on=None,
                        color="green",
                        transparancy=0.2)
        
        all_velocity.append(instant_velo)


    # plotting average velocity over time
    all_velocity = pd.concat(all_velocity, axis=1).mean(axis=1)
    avg_laser_on = np.array(all_laser_on).mean()

    plot_metric_time(all_velocity,
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
    axs.set_xlabel("Time (frame)")
    axs.set_ylabel("Velocity (m.s$^{-1}$)")

    fig.savefig(output_stacked_velo)

    if SHOW:
        plt.show()

    plt.close(fig)
