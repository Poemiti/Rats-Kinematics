#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rats_kinematics_utils.file_management import verify_exist
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, plot_violin_distribution_velocity, plot_stacked_trajectories, plot_velocity_over_sessiontime
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice

# ------------------------------------ setup ---------------------------------------

RAT_NAME = "#525"
SHOW = True
cfg = load_config()
filenames, plot_choice = load_figure_maker(cfg.paths.metrics / RAT_NAME, single_plot=False)

check_analysis_choice(filenames, plot_choice)
 
if plot_choice["plot_stacked_velocity"] : 

    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

        print(f"\n[{i+1}/{len(filenames)}]")
        print(f"Making figures of {metrics_path.parent.stem}\n")

        metrics = load_metrics(metrics_path)
        ax = plot_stacked_velocity(cfg, metrics)

        title = (
            "Average velocity of the left paw, across trials with settings:\n"
            f"{output_fig_dir.name}\n"
            f"Number of trials: {sum(1 for m in metrics if m.get('trial_success'))}"
            )

        ax.set_title(title, color="blue")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Velocity (cm.s$^{-1}$)") 

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_velocity.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)





if plot_choice["plot_stacked_Yposition"] : 
        
    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

        metrics = load_metrics(metrics_path)
        ax = plot_stacked_Yposition(cfg, metrics)

        title = (
            "Average y position of the left paw, across trials with settings:\n"
            f"{output_fig_dir.name}\n"
            f"Number of trials: {sum(1 for m in metrics if m.get('trial_success'))}"
            )

        ax.invert_yaxis()
        ax.set_title(title, color="blue")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Position (cm)")  

        # ax.set_xlim(0, 1.35)

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_Y_position_3sec.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)





if plot_choice["plot_stacked_trajectories"] : 

    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

        metrics = load_metrics(metrics_path)
        ax = plot_stacked_trajectories(cfg, metrics)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)


        ax.tick_params(direction="out")

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"Stacked Trajectories of \n{metrics_path.parent.stem}")

        ax.invert_xaxis()

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_trajectories.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)







if plot_choice["plot_violin_distribution_velocity"] : 
        
    violin_data = []
    
    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))
        violin_data.append(pd.Series(v["average_velocity"] for v in metrics if v.get('trial_success')))

    ax = plot_violin_distribution_velocity(cfg, violin_data)
    
    ax.set_ylabel("Velocity (cm.s$^{-1}$)")
    ax.set_title("Velocity distribution, for CHR rat population\n" \
                    "bodypart observed : left hand, L1 task")
    ax.grid(axis="y", alpha=0.3)

    fig = ax.figure
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "violin_distribution", f"violin_velocity_DETAIL_CHR_L1.png"))

    if SHOW : 
        plt.show()
    plt.close(fig)







if plot_choice['plot_velocity_over_sessiontime'] : 

    velocity= []
    date = []

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))
        velocity.append(pd.Series(v["average_velocity"] for v in metrics if v.get('trial_success')))
        date.append(pd.Series(d["date"] for d in metrics if d.get('trial_success')))

    ax = plot_velocity_over_sessiontime(cfg, velocity, date)

    plt.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "metrics_by_sessions", f"velocity_DETAIL_CHR_L1.png"))

    if SHOW : 
        plt.show()
    plt.close()




print("Done ! ")