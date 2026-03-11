#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rats_kinematics_utils.file_management import verify_exist
from rats_kinematics_utils.plot import plot_single_bodypart_trajectories, plot_3D_traj, plot_animation, plot_metric_time, plot_annotated_video
from rats_kinematics_utils.split_video import split_clip_range

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice, check_trial_success, print_analysis_info

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Making single figures")


RAT_NAME = cfg.rat_name
filenames, plot_choice = load_figure_maker(cfg.paths.metrics / RAT_NAME, single_plot=True)

check_analysis_choice(filenames, plot_choice)

for i, metrics_path in enumerate(filenames) :

    metrics_path = Path(metrics_path) 
    output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

    print(f"\n[{i+1}/{len(filenames)}]")
    print(f"Making figures of {metrics_path.stem}\n")

    metrics = load_metrics(metrics_path)

    for t, trial in enumerate(metrics) : 

        print(f"\n[{t+1}/{len(metrics)}]")

        # if not check_trial_success(trial) : 
        #     continue

        trial_name = trial['filename_clips'].stem
        print(f"Making figures of {trial_name}")


        # ------------------------------------ plot single trajectory + video annotation  ---------------------------------------

        if plot_choice["plot_single_bodypart_trajectories"] : 
            
            print("Making single figures ...")

            xy = trial["xy_raw"]

            if trial["laser_on"] is not None:
                frame_laser_on = xy.index[xy["t"] >= trial["laser_on"]][0]
                print(f"frame laser on: {frame_laser_on}, {trial['laser_on']}")
            else:
                frame_laser_on = None

            ax = plot_single_bodypart_trajectories(
                coords=xy,
                cm_per_pixel=cfg.cm_per_pixel,
                frame_laser_on=frame_laser_on,
                color="blue",
                marker=".",
                transparancy=0.5,
                rat_background=False
            )

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(True)


            ax.tick_params(direction="out")

            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            ax.set_title(f"Trajectories of \n{trial_name[0 : len(trial_name)//2]}\n{trial_name[len(trial_name)//2 : ]}")

            ax.invert_xaxis()
            # ax.invert_yaxis()
        
            # ax.set_xlim(0, 512)
            # ax.set_ylim(0, 512)

            ax.legend()

            fig = ax.figure
            fig.savefig(make_output_path(output_fig_dir / "trajectory_per_clip", f"traj_{trial_name}.png"))

            plt.close(fig)



        # --------------------------------------- single bodypart trajectory annotation ----------------------------------------------

        if plot_choice["plot_annotated_video"] :

            print(f"Making video annotation ...")
            
            plot_annotated_video(video_path=trial['filename_clips'],
                                    csv_path=trial["filename_coords"],
                                    output_path=make_output_path(output_fig_dir / "annotated_clips", f"anot_{trial_name}.mp4"),
                                    bodypart_name=cfg.bodypart,
                                    radius=5,
                                    likelihood_threshold=0.1)
            



        # ---------------------------------------- plot metrics ---------------------------------------

        if plot_choice["plot_metric_time"]:

            print(f"Plotting velocity, acceleration and position over time...")
            print(trial["pad_off"])

            # plotting
            fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
            plot_metric_time(metric=trial["instant_velocity"]["velocity"],
                            time=trial["instant_velocity"]['t'],
                            ax = axs[0], 
                            laser_on=trial["laser_on"],
                            color="red")
            axs[0].set_title("Velocity over time of a trial", color="red")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Velocity (cm.s$^{-1}$)")
            axs[0].set_xlim(trial["pad_off"]-0.015, trial["pad_off"] +  0.4)
            
            plot_metric_time(metric=trial["acceleration"]['acceleration'], 
                            time=trial["acceleration"]['t'],
                            ax = axs[1],
                            laser_on=trial["laser_on"],
                            color="green")
            axs[1].set_title("Acceleration over time of a trial", color="green")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Acceleration (cm.s$^{-2}$)")
            axs[1].set_xlim(trial["pad_off"]-0.015, trial["pad_off"] +  0.4)

            plot_metric_time(metric=trial["xy_filtered"]["y"], 
                            time=trial["xy_filtered"]['t'],
                            ax = axs[2],
                            laser_on=trial["laser_on"],
                            color="blue")
            axs[2].set_title("Height position over time of a trial", color="blue")
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Position (pixel)")
            axs[2].invert_yaxis()
            axs[2].set_xlim(trial["pad_off"]-0.015, trial["pad_off"] +  0.4)

            plt.savefig(make_output_path(output_fig_dir / "metric_over_time", f"metricOverTime_{trial_name}.png"))
            plt.close(fig)



        # ----------------------------------- plot trajectory over time 3D plot ---------------------------------------

        if plot_choice["plot_3D_traj"] : 

            print(f"Plotting 3D trajectory over time")

            ax = plot_3D_traj(coords=trial["xy_filtered"],
                            time=trial["xy_filtered"]["t"],
                            laser_on= trial["laser_on"] + 0.025,
                            ax= None,
                            color="blue",
                            transparancy=0.7,
                            y_invert=True)        

            ax.set_title(f"Trajectories over time of \n{trial_name}")

            fig = ax.figure
            fig.savefig(make_output_path(output_fig_dir / "plot_3D", f"plot3D_{trial_name}.png"))

            plt.close(fig)




        # ----------------------------------- plot animation ---------------------------------------

        if plot_choice["plot_animation"] : 

            print("Plotting animation...")

            input_clip_path = output_fig_dir / "annotated_clips" / f"annotated_{trial_name}.mp4"
            verify_exist(input_clip_path)

            # ----------------------------- cut clip --------------------------

            split_clip_range(input_clip_path,
                            output_path=make_output_path(output_fig_dir / "animated_plot",f"splited_clip_{trial_name}.mp4" ),
                            start=0,
                            duration= len(trial["instant_velocity"])/30 ) # converte frames to sec (30 fps)
            
            # ----------------------------- animate plot --------------------------

            fig, ax = plt.subplots(figsize=(12, 6))
            anim = plot_animation(data=trial["instant_velocity"]["velocity"],
                                time=trial["instant_velocity"]['t'],
                                laser_on=trial["laser_on"],
                                ax=ax)
            
            title = (
                "Velocity of the left paw, with settings:\n" 
                f"{output_fig_dir.name}\n" 
                f"clip = {trial_name[-15:-8]}"
                )
            
            ax.set_title(title, pad=15)
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Velocity")

            anim.save(make_output_path(output_fig_dir / "animated_plot", f"anim_plot_{trial_name}.mp4"), 
                        writer="ffmpeg", fps=30)
            plt.close(fig)


print("Done ! ")