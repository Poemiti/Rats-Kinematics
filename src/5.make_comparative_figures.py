#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rats_kinematics_utils.file_management import verify_exist, get_session
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, _plot_violin_distribution, plot_stacked_trajectories, plot_velocity_over_cliptime, _plot_displot, _plot_violin_statistic
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice, print_analysis_info
from rats_kinematics_utils.statistics import compute_statistics, save_stat_results


# ------------------------------------ setup ---------------------------------------

SHOW = False
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
        successful_trial = sum(1 for m in metrics if m[cfg.bodypart].get('trial_success'))
        print(f"Number of successful trials over total: {successful_trial}/{len(metrics)}")

        if successful_trial == 0 : 
            print("NO SUCCESSFUL TRIALS")
            continue

        ax = plot_stacked_velocity(cfg, metrics)

        title = (
            "Average velocity of the left paw, across trials with settings:\n"
            f"{output_fig_dir.name}\n"
            f"Number of trials: {successful_trial}"
            )

        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Velocity (cm.s$^{-1}$)") 

        ax.set_xlim(-0.1, 0.5)
        ax.set_ylim(-10, 150)

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_velocity.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)





if plot_choice["plot_stacked_Yposition"] : 
        
    for i, metrics_path in enumerate(filenames) :

        metrics_path = Path(metrics_path) 
        output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

        print(f"\n[{i+1}/{len(filenames)}]")
        print(f"Making figures of {metrics_path.stem}\n")

        metrics = load_metrics(metrics_path)

        successful_trial = sum(1 for m in metrics if m[cfg.bodypart].get('trial_success'))
        print(f"Number of successful trials over total: {successful_trial}/{len(metrics)}")

        if successful_trial == 0 : 
            print("NO SUCCESSFUL TRIALS")
            continue

        ax = plot_stacked_Yposition(cfg, metrics)

        title = (
            "Average y position of the left paw, across trials with settings:\n"
            f"{output_fig_dir.name}\n"
            f"Number of trials: {successful_trial}"
            )

        ax.invert_yaxis()
        ax.set_title(title)
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

        print(f"\n[{i+1}/{len(filenames)}]")
        print(f"Making figures of {metrics_path.stem}\n")

        metrics = load_metrics(metrics_path)

        successful_trial = sum(1 for m in metrics if m[cfg.bodypart].get('trial_success'))
        print(f"Number of successful trials over total: {successful_trial}/{len(metrics)}")

        if successful_trial == 0 : 
            print("NO SUCCESSFUL TRIALS")
            continue

        ax = plot_stacked_trajectories(cfg, metrics)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)


        ax.tick_params(direction="out")

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"Stacked Trajectories of \n{metrics_path.stem}\nNumber of trials: {successful_trial}")

        ax.invert_xaxis()

        fig = ax.figure
        fig.savefig(make_output_path(output_fig_dir, f"stacked_trajectories.png"))

        if SHOW : 
            plt.show()
        plt.close(fig)



############### violin ########################



def _preprocess_violin(METRIC: str, split_condition: bool = False) -> pd.DataFrame : 
    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial[cfg.bodypart]["trial_success"] : 
                continue

            name = trial["filename_clips"].stem
            rat = name[4:8]
            
            if split_condition : 
                condition = trial["condition"]
                laser_state = trial["laser_state"]
            else : 
                condition = trial["condition"] + "_" + trial["laser_state"]
            reward = "yes" if trial["reward"] else "no"

            
            if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
            elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
            else : laser_intensity = "high"

            df = pd.DataFrame({
                "value": [trial[cfg.bodypart][METRIC]],
                "rat": [rat],
                "condition": [condition],
                "laser_state": [laser_state if split_condition else None],  
                "laser_intensity": [laser_intensity],
                "reward" : [reward]
            })
            data = pd.concat([data, df])

    return data.sort_values(by="condition")



def _make_violin(cfg, data, metric) : 
    g = _plot_violin_distribution(cfg, data)
    
    g.set_ylabels(metric)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "violin_distribution", f"violin_{metric}_left_CHR_L1.png"))

    if SHOW : 
        plt.show()
    plt.close()



if plot_choice["plot_violin_distribution_tortuosity"] : 
    violin_data = _preprocess_violin(METRIC= "tortuosity", split_condition=True)
    _make_violin(cfg, violin_data, "tortuosity (true path over shortest path)")

if plot_choice["plot_violin_distribution_velocity"] : 
    violin_data = _preprocess_violin(METRIC= "average_velocity", split_condition=True)

    # ANOVA(violin_data, "value ~ condition * laser_state * laser_intensity")

    _make_violin(cfg, violin_data, "average velocity (cm.s$^{-1}$)")
    



######################## violin statistics #################################




def _make_violin_stat(data, metric, formula) : 
    stats_res = compute_statistics(data, formula)
    
    # save statistics results has joblib: dict of dataframe for each statistical test
    save_stat_results(stats_res, cfg.paths.metrics / "statistics" / RAT_NAME / f"{metric}.joblib")

    if "mann_whitney" in stats_res.keys() :
        pairwise_results = stats_res["mann_whitney"] 
        significant_pair = pairwise_results[pairwise_results["p_value"] < 0.05]

        fig = _plot_violin_statistic(cfg, data, significant_pair, strip=len(significant_pair) == 0)
        fig.suptitle(f"{metric} distribution for rat {RAT_NAME}")
        fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "violin_distribution",  f"stat_violin_{metric}_{RAT_NAME}.png"))

        if SHOW : 
                plt.show()
        plt.close()



if plot_choice["plot_violin_stat_velocity"] : 
    metric = "average_velocity"
    data = _preprocess_violin(METRIC= metric, split_condition=False)
    _make_violin_stat(data, metric, "value ~ condition * laser_intensity")



if plot_choice["plot_violin_stat_tortuosity"] : 
    metric = "tortuosity"
    data = _preprocess_violin(METRIC= metric, split_condition=False)
    _make_violin_stat(data, metric, "value ~ condition * laser_intensity")







################ displot #####################""







def _make_displot(cfg, data, metric) : 
    g = _plot_displot(data)
    
    g.figure.suptitle(f'Distribution of {metric} depending on condition', ha='center')
    g.figure.subplots_adjust(top=0.88)
    g.set_axis_labels(metric, "Density (KDE)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # g.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "violin_distribution", f"displot_{metric}_left_CHR_L2.png"))

    if SHOW : 
        plt.show()
    plt.close()


if plot_choice["plot_displot_tortuosity"] : 
    displot_data = _preprocess_violin(METRIC= "tortuosity", split_condition=True)
    _make_displot(cfg, displot_data, "tortuosity")

if plot_choice["plot_displot_velocity"] : 
    displot_data = _preprocess_violin(METRIC= "average_velocity", split_condition=True)
    _make_displot(cfg, displot_data, "average velocity")
    




############


if plot_choice['plot_velocity_over_cliptime'] : 

    data = pd.DataFrame()

    for i, metrics_path in enumerate(filenames) :
        metrics = load_metrics(Path(metrics_path))

        for trial in metrics : 

            if not trial[cfg.bodypart]["trial_success"] : 
                continue

            name = trial["filename_clips"].as_posix()
            session = get_session(name)
            condition = trial["condition"]
            laser_state = trial["laser_state"]

            df = pd.DataFrame({
                "date": [trial["date"]],
                "velocity": [trial[cfg.bodypart]["average_velocity"]],
                "condition": condition,
                "laser_state" : laser_state,
                "clip" : [int(trial["nb_clip"])],
                "session" : [session]
            })


            data = pd.concat([data, df])

    final_data: pd.DataFrame = data.sort_values(
                by=["date", "condition", "clip", "session"],
                ascending=[True, True, True, True], 
            )
    final_data["date"] = pd.to_datetime(final_data["date"]).dt.date
    final_data.to_csv(make_output_path(cfg.paths.figures / RAT_NAME / "metrics_by_sessions", f"data.csv"))

    fig = plot_velocity_over_cliptime(final_data)
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "metrics_by_sessions", f"velocity_overclip_LeftHemi_CHR_L1.png"))

    if SHOW : 
        plt.show()
    plt.close()




print("Done ! ")