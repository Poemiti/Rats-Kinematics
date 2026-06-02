#!/usr/bin/env python

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import rats_kinematics_utils.preprocessing.plot_preprocess as pp

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import dataframe_report, make_output_path, load_trial_data, check_analysis_choice, filter_contra_trials
from rats_kinematics_utils.gui.figures_maker import load_figure_maker

import rats_kinematics_utils.analysis.inter_rat as ir
import rats_kinematics_utils.analysis.plot_comparative as pc
import rats_kinematics_utils.analysis.behavior_plot as bp

CONTEXT= "talk"
pc.set_plot_style(CONTEXT)

# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()

file_to_process = filter_contra_trials(cfg, single_rat=False)
print()

file_to_process_list = [item for lst in file_to_process.values() for item in lst]
filenames, plot_choice = load_figure_maker(file_to_process_list, kind="inter_rat")

check_analysis_choice(filenames, plot_choice)


# --------------------------------------- some metadata visualisation -------------------------------------------

if plot_choice["plot_metadata"] : 

    # print(f"\nMaking inter rat metadata report\n")
    # ir.inter_rat_metadata_report(cfg, ["CHR"], file_to_process_list)

    print(f"\nPlotting overall trials failure reason\n")
    pp.plot_trial_failure_reason(cfg, ["CHR"], file_to_process_list, inter_rat=True)

    sys.exit()

    print(f"\nPlotting overall bodyparts likelihood distribution\n")
    ir.plot_likelihood_distri_interrat(cfg, file_to_process_list)


# --------------------------------------- main -------------------------------------------
        

# velocity tendency

if plot_choice["plot_velocity_per_rat"] : 
                
    data = ir._preprocess_tendency(cfg, file_to_process_list, "instant_velocity", "velocity")

    for error_function in [None, "pi","sem"] :
        error_name = error_function if error_function is not None else ""
            
        g = ir._plot_tendency_per_rat(cfg, data, error_function)

        g.figure.suptitle(f"Velocity tendency of rat {cfg.inter_rat.rats} - (error: {error_name})\nNumber of trials: {len(data.groupby('id'))}", ha='center')
        g.figure.subplots_adjust(top=0.80)
        g.set_axis_labels("Time (sec)", "Velocity (cm.s$^{-1}$)")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"velocity_tendency_per_rats_{error_name}_{CONTEXT}.png"))
        g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"velocity_tendency_per_rats_{error_name}_{CONTEXT}.svg"))




if plot_choice["plot_velocity_mean_tendency"] : 
                
    data = ir._preprocess_tendency(cfg, file_to_process_list, "instant_velocity", "velocity")

    for error_function in ["sem"] :
        error_name = error_function if error_function is not None else ""
            
        g = pc._plot_tendency(data, error_function)

        g.figure.suptitle(f"Velocity tendency of rat {cfg.inter_rat.rats} - (error: {error_name})\nNumber of trials: {len(data.groupby('id'))}", ha='center')
        g.figure.subplots_adjust(top=0.80)
        g.set_axis_labels("Time (sec)", "Velocity (cm.s$^{-1}$)")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"mean_velocity_tendency_{error_name}_{CONTEXT}.png"))
        g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"mean_velocity_tendency_{error_name}_{CONTEXT}.svg"))


# lever distance

if plot_choice["plot_lever_distance"] : 
                
    data = ir._preprocess_tendency(cfg, file_to_process_list, "lever_distance", "distance")

    for error_function in [None, "pi","sem"] :
        error_name = error_function if error_function is not None else ""
    
        g = ir._plot_tendency_per_rat(cfg, data, error_function)

        g.figure.suptitle(f"Distance between lever and {cfg.bodypart} of rat {cfg.inter_rat.rats} {error_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')
        g.figure.subplots_adjust(top=0.80)
        g.set_axis_labels("Time (sec)", "Distance (cm)")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.savefig(make_output_path(cfg.paths.inter_rat / f"lever_distance", f"lever_distance_{error_name}_{CONTEXT}.png"))
        g.savefig(make_output_path(cfg.paths.inter_rat / f"lever_distance", f"lever_distance_{error_name}_{CONTEXT}.svg"))


################## setup + print d'info

# metric = "average_velocity"
# data = ir._preprocess(cfg, file_to_process_list, metric, split_condition=False)
# print(data.head(5))
# info = dataframe_report(data)

# for col, info in info.items():
#     print(f"\nColumn: {col}")
#     print(info['summary'].T)


# condition = ["Beta", "Conti"]
# l_state = data["laser_state"].unique()
# l_intensity = ["low", "high"]

# for intensity in l_intensity :  
#     print(f"\n{intensity} :")
#     for state in l_state : 
#         for c in condition : 
#             size = len(data[(data["condition"] == c) &
#                             (data["laser_state"] == state) &
#                             (data["laser_intensity"] == intensity)])
#             print(f"  n {c} {state} : {size}")

################# normal statistics


if plot_choice["plot_statistics"] : 

    # merged laserOff
    comparisons = [
                # conti vs beta
                ("Conti.low",  "Beta.low"),
                ("Conti.high", "Beta.high"),

                # low vs high
                ("LaserOff.high", "Conti.high"),
                ("LaserOff.high",  "Beta.high"),
                ("LaserOff.low",    "Beta.low"),
                ("LaserOff.low",   "Conti.low"),

                # condition high vs condition low
                ("Conti.low", "Conti.high"),
                ("Beta.low",  "Beta.high"),

                # Beta high ON vs Conti low ON 
                # (because conti low is suppose to be as damagefull as the beta high)
                ("Beta.high", "Conti.low")
            ]

    print(f"\n ------ computing statistics on average velocity ------\n")
    ir.plot_statistics(cfg, file_to_process_list, "average_velocity", comparisons, 
                    merge_laserOff=True, per_rats=False)

    # print(f"\n ------ computing statistics on tortuosity ------\n")
    # ir.plot_statistics(cfg, file_to_process_list, "tortuosity", comparisons, merge_laserOff=True)




# comparisons = [
#             # Conti vs Beta
#             ("Conti_LaserOff.low",  "Beta_LaserOff.low"),
#             ("Conti_LaserOff.high", "Beta_LaserOff.high"),
#             ("Conti_LaserOn.low",   "Beta_LaserOn.low"),
#             ("Conti_LaserOn.high",  "Beta_LaserOn.high"),

#             # Off vs On
#             ("Conti_LaserOff.low",  "Conti_LaserOn.low"),
#             ("Conti_LaserOff.high", "Conti_LaserOn.high"),
#             ("Beta_LaserOff.low",   "Beta_LaserOn.low"),
#             ("Beta_LaserOff.high",  "Beta_LaserOn.high"),

#             # low vs high
#             ("Beta_LaserOff.low",   "Beta_LaserOff.high"),
#             ("Beta_LaserOn.low",    "Beta_LaserOn.high"),
#             ("Conti_LaserOff.low",  "Conti_LaserOff.high"),
#             ("Conti_LaserOn.low",   "Conti_LaserOn.high"),

#             # Beta high ON vs Conti low ON 
#             # (because conti low is suppose to be as damagefull as the beta high)
#             ("Beta_LaserON.high", "Conti_LaserON.low")
#         ]



# # ################### permutation


if plot_choice["plot_permutation"] : 

    n_perm = 100000

    for metric in ["average_velocity"] :

        print("="*60)
        print(f"\nSize effect of LOW laser intensity, metric={metric} :")
        ir.plot_permutation(cfg, file_to_process_list, metric, "low", n_perm)

        print("="*60)
        print(f"\nSize effect of HIGH laser intensity, metric={metric} :")
        ir.plot_permutation(cfg, file_to_process_list, metric, "high", n_perm)


    
# --------------------------------------- reward proportion ------------------------------------------

if plot_choice["plot_reward_proportion"] : 
    data = bp.preprocess_reward(cfg, file_to_process_list, remove_may=False)
    print(data)

    g = pc.plot_rewarded_bar(data, during_laser=True)
    g.figure.subplots_adjust(top=0.76)

    g.figure.suptitle(f"Rewarded trial proportion (using LED - during laser stim)\nNumber of trials: {len(data.groupby('id'))}", ha='center')
    g.savefig(make_output_path(cfg.paths.inter_rat / f"reward", f"rewarded_proportion_during_laser_{CONTEXT}.png"))
    g.savefig(make_output_path(cfg.paths.inter_rat / f"reward", f"rewarded_proportion_during_laser_{CONTEXT}.svg"))

    plt.close()
 
# ---------------------------------------------- big ethogram ----------------------------------
# remove the may trial !!!

if plot_choice["plot_ethogram_by_condition"] : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, file_to_process_list, remove_may=False)
    biggest_condition, n_trials_biggest = bp.get_biggest_condition(cfg, data_behavior)

    print(biggest_condition, n_trials_biggest)
    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.plot_ethogram_by_condition(cfg, data_behavior, n_trials_biggest, cfg.paths.inter_rat ,align_by="pad_off")




if  plot_choice["plot_behavior_proba_per_condition"] : 
    data_behavior, time_stamps = bp.preprocess_trial_behavior(cfg, file_to_process_list, remove_may=False)
    biggest_condition, n_trials_biggest = bp.get_biggest_condition(cfg, data_behavior)

    print(biggest_condition, n_trials_biggest)
    data_behavior = data_behavior.merge(time_stamps, on="id", how="left")
    bp.plot_behavior_proba_per_condition(cfg, data_behavior, cfg.paths.inter_rat, align_by="pad_off")



# # ------------------------------------------- probability per behavior --------------------------------------------------
# # remove the may trial !!!

if  plot_choice["plot_behavior_proba_per_behavior"] : 

    data = bp.preprocess_proba(cfg, file_to_process_list, remove_may=False)
    bp.plot_behavior_proba_per_behavior(cfg, data, cfg.paths.inter_rat)


if  plot_choice["plot_mean_proba_per_behavior"] : 

    data = bp.preprocess_proba(cfg, file_to_process_list, remove_may=False)
    data = data.loc[(data["t"] >= 0) & (data["t"] <= 0.325)]    # crop around laser stim

    bp.plot_mean_proba_per_behavior(cfg, data, cfg.paths.inter_rat)



# # ------------------------------------------- transition matrix --------------------------------------------------
# # uses the may trials since we only look at the period of laser

if plot_choice["plot_transition_matrix"] : 

    rewarded = None

    if rewarded is None: 
        r_filter = "both"
        r_name = ""
    elif rewarded == True: 
        r_filter = "reward_only"
        r_name = "reward"
    elif rewarded == False : 
        r_filter = "non_reward_only"
        r_name = "non_reward"

    print(f"\n{r_filter}")
    data = bp.preprocess_proba(cfg, file_to_process_list, remove_may=False, reward=r_filter)
    data = data.loc[(data["t"] >= 0) & (data["t"] <= 0.325)]    # crop around laser stim

    for (condition, l_state, l_intensity), subset in data.groupby(["condition", 'laser_state', 'laser_intensity']) :
        
        # compute transition matrix
        ax, transition_matrix = bp.plot_transition_matrix(cfg, subset)

        ax.set_xlabel("Next behavior")
        ax.set_ylabel("Current behavior")
        ax.set_title(f"{condition}_{l_state}_{l_intensity}\n{r_name} - n trial: {len(subset.groupby('id'))}")

        ax.figure.savefig(make_output_path(cfg.paths.inter_rat / "transition_matrix" / r_name, f"{condition}_{l_state}_{l_intensity}_{r_name}_{CONTEXT}.png"))
        ax.figure.savefig(make_output_path(cfg.paths.inter_rat /  "transition_matrix" / r_name, f"{condition}_{l_state}_{l_intensity}_{r_name}_{CONTEXT}.svg"))
        
        plt.close(ax.figure)

        # draw transition as a graph
        ax2 = bp._draw_transition_graph(transition_matrix, threshold=0.01)

        ax2.set_title(f"{condition}_{l_state}_{l_intensity}\n{r_name} - n trial: {len(subset.groupby('id'))}")
        ax2.axis("off")

        ax2.figure.savefig(make_output_path(cfg.paths.inter_rat / "transition_matrix" / "markov_chain" / r_name, f"{condition}_{l_state}_{l_intensity}_{r_name}_{CONTEXT}.png"))
        ax2.figure.savefig(make_output_path(cfg.paths.inter_rat / "transition_matrix" / "markov_chain" / r_name, f"{condition}_{l_state}_{l_intensity}_{r_name}_{CONTEXT}.svg"))

        plt.close(ax2.figure)

# # ------------------------------------------- combinaison of behavior --------------------------------------------------
# # uses the may trials since we only look at the period of laser


if plot_choice["plot_proportion_behavior_combinaison"] : 

    data = bp.preprocess_proba(cfg, file_to_process_list, remove_may=False, reward="both")
    data = data.loc[(data["t"] >= 0) & (data["t"] <= 0.325)]    # crop around laser stim

    print(data)
    for (condition, l_intensity), subset in data.groupby(["condition", 'laser_intensity']) :
        
        print()
        print(condition, l_intensity)
        ax = bp.plot_proportion_behavior_combinaison(cfg, subset)
        ax.set_title(f"{condition}_{l_intensity}\nn trial: {len(subset.groupby('id'))}")

        
        ax.figure.subplots_adjust(top=0.80)
        ax.figure.savefig(make_output_path(cfg.paths.inter_rat / f"behavior" / "behavior_combinaison", f"{condition}_{l_intensity}_{CONTEXT}.png"))
        ax.figure.savefig(make_output_path(cfg.paths.inter_rat / f"behavior" / "behavior_combinaison", f"{condition}_{l_intensity}_{CONTEXT}.svg"))

        plt.close()





if plot_choice["plot_time_in_behavior_space"] : 

    data = bp.preprocess_proba(cfg, file_to_process_list, remove_may=False)
    data = data.loc[(data["t"] >= 0) & (data["t"] <= 0.325)]    # crop around laser stim

    for laser_intensity, subset in data.groupby("laser_intensity"): 
        
        print(laser_intensity)
        g = bp.plot_time_in_behavior_space(cfg, subset)

        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.figure.subplots_adjust(top=0.76)
        g.figure.suptitle(f"Time spend in each zones (during laser stim)\nLaser intensity: {laser_intensity} - Number of trials: {len(subset.groupby('id'))}", ha='center')
        g.savefig(make_output_path(cfg.paths.inter_rat / f"behavior", f"time_spend_per_behavior_{laser_intensity}.png"))
        g.savefig(make_output_path(cfg.paths.inter_rat / f"behavior", f"time_spend_per_behavior_{laser_intensity}.svg"))

        plt.close()





def _make_displot(cfg, data, metric) : 
    g = pc._plot_displot(data)
    
    g.figure.suptitle(f'Distribution of {metric} depending on condition', ha='center')
    # g.figure.subplots_adjust(top=0.88)
    g.set_axis_labels(metric, "Density (KDE)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.savefig(make_output_path(cfg.paths.inter_rat / "analysis_distribution", f"effect_size_{metric}_{CONTEXT}.png"))

    if SHOW : 
        plt.show()
    plt.close()



if plot_choice["plot_velocity_effect_size"] : 

    displot_data = ir._preprocess(cfg, filenames=file_to_process_list, METRIC= "average_velocity", merge_laserOff=True)
    print(displot_data)
    _make_displot(cfg, displot_data, "average velocity")

print("Done !")