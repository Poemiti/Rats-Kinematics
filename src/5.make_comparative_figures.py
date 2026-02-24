#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rats_kinematics_utils.file_management import verify_exist, get_session
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, _plot_violin_distribution, plot_stacked_trajectories, plot_velocity_over_cliptime
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice, print_analysis_info

# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()
print_analysis_info(cfg, "Making comparative figures")


RAT_NAME = cfg.rat_name
filenames, plot_choice = load_figure_maker(cfg.paths.metrics / RAT_NAME, single_plot=False)

check_analysis_choice(filenames, plot_choice)
 
if plot_choice["plot_stacked_velocity"] : 

    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

        print(f"\n[{i+1}/{len(filenames)}]")
        print(f"Making figures of {metrics_path.stem}\n")

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

        ax.set_xlim(-0.1, 0.5)

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

        ax.set_xlim(-0.1, 0.5)

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
        ax.set_title(f"Stacked Trajectories of \n{metrics_path.stem}")

        # ax.invert_xaxis()

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_trajectories.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)





def _preprocess_violin(METRIC: str) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial["trial_success"] : 
                continue

            name = trial["filename_clips"].as_posix()
            condition, laser_state = trial["condition"].split('_', 1)

            df = pd.DataFrame({
                "value": [trial[METRIC]],
                "condition": [condition],
                "laser_state": [laser_state],
                "laser_intensity": [trial["laser_intensity"]]
            })
            data = pd.concat([data, df])

    return data.sort_values(by="condition")


def _make_violin(cfg, data, metric) : 
    ax = _plot_violin_distribution(cfg, data)
    
    ax.set_ylabels(metric)
    ax.set_titles(col_template="{col_name}", row_template="{row_name}")
    ax.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "violin_distribution", f"violin_{metric}_all_CHR_L2.png"))

    if SHOW : 
        plt.show()
    plt.close()


if plot_choice["plot_violin_distribution_tortuosity"] : 
    violin_data = _preprocess_violin(METRIC= "tortuosity")
    _make_violin(cfg, violin_data, "tortuosity")

if plot_choice["plot_violin_distribution_velocity"] : 
    violin_data = _preprocess_violin(METRIC= "average_velocity")
    print(violin_data)
    _make_violin(cfg, violin_data, "average_velocity")
    
if plot_choice["plot_violin_distribution_peak"] : 
    violin_data = _preprocess_violin(METRIC= "peak_velocity")
    print(violin_data)
    _make_violin(cfg, violin_data, "peak_velocity")






############


if plot_choice['plot_velocity_over_cliptime'] : 

    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial["trial_success"] : 
                continue

            name = trial["filename_clips"].as_posix()
            session = get_session(name)

            df = pd.DataFrame({
                "date": [trial["date"]],
                "velocity": [trial["average_velocity"]],
                "condition": [trial["condition"]],
                "clip" : [trial["nb_clip"]],
                "session" : [session]
            })


            data = pd.concat([data, df])

    final_data: pd.DataFrame = data.sort_values(
                by=["date", "condition",  "clip", "session"],
                ascending=[True, True, True, True], 
            )
    final_data["date"] = pd.to_datetime(final_data["date"]).dt.date
    final_data.to_csv(cfg.paths.figures / RAT_NAME / "metrics_by_sessions" / f"data.csv")

    fig = plot_velocity_over_cliptime(final_data)
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "metrics_by_sessions", f"velocity_overclip_LeftHemi_CHR_L2.png"))

    if SHOW : 
        plt.show()
    plt.close()




print("Done ! ")