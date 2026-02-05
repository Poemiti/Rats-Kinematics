#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.trajectory_metrics import Trajectory
import csv_list as dataset
from utils.database_filter import Model, View, Controller
from utils.file_management import make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_single_bodypart_trajectories, open_clean_csv, plot_3D_traj
from utils.trajectory_metrics import Trajectory, plot_metric_time, define_End_of_trajectory, animate_plot, plot_stacked_metric
from utils.led_detection import get_time_led_on, get_time_led_off
from utils.split_video import split_clip_range

# set parameters
THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left'           # or 'right'f
LEVER_POSITION = 215    # pixels
LASER_ON_TIME = 0.325   # sec
SHOW = True

# define m per pixel
if view == 'left' : 
    frame_width_m = 8.7 # cm
else : 
    frame_width_m = 8.3 # cm
frame_width_px = 512 # pixel
CM_PER_PIXEL = frame_width_m / frame_width_px

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
ANALYSIS_DIR = GENERATED_DATA_DIR / "analysis_results" / "#517"
OUTPUT_DIR = GENERATED_DATA_DIR / "analysis_results" / "#517" / "selected_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAT_NAME = "#517"
OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "analysis_results" / RAT_NAME

# ------------------------------------ output file directory preparation ---------------------------------------

on = [dataset.beta_1, dataset.beta_25,dataset.conti05, dataset.conti075]
off = [dataset.beta_1_nostim, dataset.beta_25_nostim, dataset.conti05_nostim, dataset.conti075_nostim]

for csv_list in on : 

    output_fig_dir = OUTPUT_TRAJECTORY_PATH / Path(make_directory_name(Path(csv_list[0]).stem))
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"File will be stored in {output_fig_dir}")

            
    #############################################################################################

    output_stacked_traj = output_fig_dir / f"selected_trajectory.png"
    output_stacked_velo = output_fig_dir / f"selected_velocity.png"
    fig, axs = plt.subplots(figsize=(9, 7))

    all_coords = []
    all_velocity = []
    all_laser_on = [] 
    all_pad_off = []

    for i, csv_path in enumerate(csv_list) : 

        csv_path = Path(csv_path)
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
        xy = xy.loc[xy["likelihood"] >= THRESHOLD, ["x", "y", "t"]]
        
        #  pad off -> laser off coords 
        xy_filtered = xy.loc[
            (xy["t"] >= 0) &
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


        ################################################################################################################################################################

        # # ------------------------------------------------------------------------------------------------
        # # --------------------------------------- compute metrics ----------------------------------------------
        # # ------------------------------------------------------------------------------------------------


        # print("Metrics measurement ...")

        # # creat trajectory object for metric calculation
        # Traj = Trajectory(coords=xy,
        #                 reaching_coords=xy_filtered, 
        #                 laserOn_coords=xy_laserOn,
        #                 fps=125, 
        #                 cm_per_pixel=CM_PER_PIXEL)

        # # compute metrics
        # instant_velo = Traj.instant_velocity()
        # acceleration = Traj.acceleration()


        # # ---------------------------------------- plot metrics ---------------------------------------


        # print(f"Plotting velocity, acceleration and position over time...")

        # output_fig_metric_dir = output_fig_dir / "selected_metric_over_time" 
        # output_fig_metric_dir.mkdir(parents=True, exist_ok=True)
        # output_fig_metric_path = output_fig_metric_dir / f"metricOverTime_{csv_path.stem.replace('_pred_results', '')}.png"
        
        # # plotting
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # plot_metric_time(metric=instant_velo["velocity"],
        #                  time=instant_velo['t'],
        #                 ax = axs[0], 
        #                 laser_on=time_laser_on,
        #                 color="red")
        # axs[0].set_title("Velocity over time of a trial", color="red")
        # axs[0].set_xlabel("Time (s)")
        # axs[0].set_ylabel("Velocity (cm.s$^{-1}$)")
        
        # plot_metric_time(metric=acceleration['acceleration'], 
        #                  time=acceleration['t'],
        #                 ax = axs[1],
        #                 laser_on=time_laser_on,
        #                 color="green")
        # axs[1].set_title("Acceleration over time of a trial", color="green")
        # axs[1].set_xlabel("Time (s)")
        # axs[1].set_ylabel("Acceleration (cm.s$^{-2}$)")

        # plot_metric_time(metric=xy_filtered["y"], 
        #                  time=xy_filtered['t'],
        #                 ax = axs[2],
        #                 laser_on=time_laser_on,
        #                 color="blue", 
        #                 y_invert=True)
        # axs[2].set_title("Height position over time of a trial", color="blue")
        # axs[2].set_xlabel("Time (s)")
        # axs[2].set_ylabel("Position (pixel)")

        # plt.savefig(output_fig_metric_path)
        # plt.close(fig)

    # # ------------------------------------------------------------------------------------------------
    # # -------------------------------------- animation -----------------------------------------------
    # # ------------------------------------------------------------------------------------------------

    #     print("Plotting animation...")

    #     output_anim_dir = output_fig_dir / "animated_plot" 
    #     output_anim_dir.mkdir(parents=True, exist_ok=True)
    #     output_anim_path = output_anim_dir / f"anim_plot_{csv_path.stem.replace('_pred_results', '')}.mp4"

    #     output_annotated_clip_dir = output_fig_dir / "annotated_clips"
    #     input_clip_path = output_annotated_clip_dir / f"annotated_{csv_path.stem.replace('_pred_results', '')}.mp4"
    #     output_clip_path = output_anim_dir / f"splited_clip_{csv_path.stem.replace('_pred_results', '')}.mp4"
        
    #     verify_exist(input_clip_path)

    #     # ----------------------------- extract data --------------------------

    #     Traj = Trajectory(coords=xy,
    #                     reaching_coords=xy.loc[
    #                                     (xy["t"] >= 0) &
    #                                     (xy["t"] <= time_pad_off + 0.5)
    #                                 ].reset_index(drop=True), 
    #                     laserOn_coords=xy_filtered,
    #                     fps=125, 
    #                     cm_per_pixel=CM_PER_PIXEL)
    #     instant_velo : pd.DataFrame = Traj.instant_velocity()
    #     print(f"lenght instant velo : {(len(instant_velo))}")

    #     # ----------------------------- cut clip --------------------------

    #     split_clip_range(input_clip_path,
    #                      output_path=output_clip_path,
    #                      start=0,
    #                      duration= len(instant_velo)/30 ) # converte frames to sec (30 fps)
        
    #     # ----------------------------- animate plot --------------------------
        
        
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     anim = animate_plot(data=instant_velo["velocity"],
    #                         time=instant_velo['t'],
    #                         laser_on=time_laser_on,
    #                         ax=ax)
        
    #     title = (
    #         "Velocity of the left paw, with settings:\n" 
    #         f"{output_fig_dir.name}\n" 
    #         f"clip = {csv_path.stem[-15:-8]}"
    #         )
        
    #     ax.set_title(title, pad=15)
    #     ax.set_xlabel("Time (sec)")
    #     ax.set_ylabel("Velocity")

    #     anim.save(output_anim_path, writer="ffmpeg", fps=30)
    #     plt.close(fig)


    # ------------------------------------------------------------------------------------------------
    # -------------------------------------- plot trajectories ----------------------------------------
    # ------------------------------------------------------------------------------------------------


        plot_single_bodypart_trajectories(
            coords=xy_filtered,
            cm_per_pixel=CM_PER_PIXEL,
            frame_laser_on=int(time_laser_on * 125) if time_laser_on is not None else None,
            ax=axs,
            invert_y=False,
            color="blue",
            transparancy=0.5
        )
        all_coords.append(xy_filtered)
        print(len(xy_filtered))


    title = (
        "Trajectories across trials with settings:\n"
        f"{output_fig_dir.stem}\n"
        f"Number of trials: {len(csv_list)}"
    )

    axs.invert_xaxis()

    axs.yaxis.tick_right()
    axs.yaxis.set_label_position("right")

    axs.spines["top"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.spines["right"].set_visible(True)

    axs.tick_params(direction="out")

    axs.set_xlabel("x (cm)")
    axs.set_ylabel("y (cm)")

    axs.set_title(title, fontsize=12)
    fig.savefig(output_stacked_traj)

    if SHOW:
        plt.show()
    plt.close(fig)
