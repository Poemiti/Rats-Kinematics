#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.file_management import verify_exist
from utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, plot_violin_distribution_velocity
from config import load_config
from utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice

# ------------------------------------ setup ---------------------------------------

RAT_NAME = "#525"
cfg = load_config()
filenames, plot_choice = load_figure_maker(cfg, RAT_NAME, single_plot=False)

check_analysis_choice(filenames, plot_choice)

if plot_choice["plot_stacked_velocity"] : 

    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.parent.stem

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
        plt.close(fig)





if plot_choice["plot_stacked_Yposition"] : 
        
    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.parent.stem

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

        ax.set_xlim(0, 1.35)


        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_Y_position.png"))
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
    plt.close(fig)





print("Done ! ")