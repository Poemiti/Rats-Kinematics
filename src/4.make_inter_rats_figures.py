#!/usr/bin/env python

import sys
from pathlib import Path
import pandas as pd

import rats_kinematics_utils.preprocessing.plot_preprocess as pp

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import dataframe_report, make_output_path, load_trial_data
import rats_kinematics_utils.analysis.inter_rat as ir
import rats_kinematics_utils.analysis.plot_comparative as pc


# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()

file_to_process = ir.filter_contra_trials(cfg, single_rat=False)
rat_proportion = ir.get_rat_proportion(file_to_process)

print(rat_proportion)


print(f"\nThe following files will be used for the inter rat analysis:")
for r, f_list in file_to_process.items(): 
    print(r)
    for f in f_list: 
        print(f"  {f.stem}")
print()

res = input("ENTER to launch preprocessing or (q) to quit : ")
if res == "q" or res=="Q": 
    print("quit!")
    sys.exit()

# --------------------------------------- some metadata visualisation -------------------------------------------

file_to_process_list = [item for lst in file_to_process.values() for item in lst]

rat_types = ["CHR"]

res = input("Do you want to plot metadata of the whole dataset ? (y/n): ")
if res == "y": 

    print(f"\nMaking inter rat metadata report\n")
    ir.inter_rat_metadata_report(cfg, rat_types, file_to_process_list)

    sys.exit()

    print(f"\nPlotting overall trials failure reason\n")
    pp.plot_trial_failure_reason(cfg, file_to_process_list, inter_rat=True)

    print(f"\nPlotting overall bodyparts likelihood distribution\n")
    ir.plot_likelihood_distri_interrat(cfg, file_to_process_list)

elif res not in ["y", "n", "Y", "N"] : 
    raise ValueError(f"ERROR, '{res}' is not a valid answer, must be 'y' or 'n'")


# --------------------------------------- main -------------------------------------------

# for metric in  ["instant_velocity"] : 
    
#     metric_vector = "velocity" if metric == "instant_velocity" else "distance"
#     data = ir._preprocess_tendency(cfg, file_to_process_list, metric, metric_vector)

#     for error_function in [None, "pi","sem"] :
#         error_name = error_function if error_function is not None else ""

#         # velocity tendency
#         if metric == "instant_velocity" : 
#             g = ir.plot_tendency_per_rat(cfg, data, error_function)

#             g.figure.suptitle(f"Velocity tendency of rat {cfg.inter_rat.rats} - (error: {error_name})\nNumber of trials: {len(data.groupby('id'))}", ha='center')
#             g.figure.subplots_adjust(top=0.88)
#             g.set_axis_labels("Time (sec)", "Velocity (cm.s$^{-1}$)")
#             g.set_titles(col_template="{col_name}", row_template="{row_name}")

#             g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"velocity_tendency_per_rats_{error_name}.png"))
#             g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"velocity_tendency_per_rats_{error_name}.svg"))
#             g.savefig(make_output_path(cfg.paths.inter_rat / f"tendency", f"velocity_tendency_per_rats_{error_name}.pdf"))


        # # lever distance
        # if metric == "lever_distance" : 
        #     g = pc.plot_lever_distance(cfg, data, error_function)

        #     g.figure.suptitle(f"Distance between lever and {cfg.bodypart} of rat {cfg.inter_rat.rats} {error_name}\nNumber of trials: {len(data.groupby('id'))}", ha='center')
        #     g.figure.subplots_adjust(top=0.88)
        #     g.set_axis_labels("Time (sec)", "Distance (cm)")
        #     g.set_titles(col_template="{col_name}", row_template="{row_name}")

        #     g.savefig(make_output_path(cfg.paths.inter_rat / f"lever_distance", f"lever_distance_{error_name}.png"))
        #     g.savefig(make_output_path(cfg.paths.inter_rat / f"lever_distance", f"lever_distance_{error_name}.svg"))
        #     g.savefig(make_output_path(cfg.paths.inter_rat / f"lever_distance", f"lever_distance_{error_name}.pdf"))


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

# # merged laserOff
# comparisons = [
#             # conti vs beta
#             ("Conti.low",  "Beta.low"),
#             ("Conti.high", "Beta.high"),

#             # low vs high
#             ("LaserOff.high", "Conti.high"),
#             ("LaserOff.high",  "Beta.high"),
#             ("LaserOff.low",    "Beta.low"),
#             ("LaserOff.low",   "Conti.low"),

#             # condition high vs condition low
#             ("Conti.low", "Conti.high"),
#             ("Beta.low",  "Beta.high"),

#             # Beta high ON vs Conti low ON 
#             # (because conti low is suppose to be as damagefull as the beta high)
#             ("Beta.high", "Conti.low")
#         ]

# print(f"\n ------ computing statistics on average velocity ------\n")
# ir.plot_statistics(cfg, file_to_process_list, "average_velocity", comparisons, 
#                    merge_laserOff=True, per_rats=True)

# # print(f"\n ------ computing statistics on tortuosity ------\n")
# # ir.plot_statistics(cfg, file_to_process_list, "tortuosity", comparisons, merge_laserOff=True)


# sys.exit()


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

# print(f"\n ------ computing statistics on average velocity ------\n")
# ir.plot_statistics(cfg, file_to_process_list, "average_velocity", comparisons, rat_proportion)

# print(f"\n ------ computing statistics on tortuosity ------\n")
# ir.plot_statistics(cfg, file_to_process_list, "tortuosity", comparisons, rat_proportion)


# # ################### permutation

# n_perm = 100000

# for metric in ["average_velocity", "tortuosity"] :

#     print("="*60)
#     print(f"\nSize effect of LOW laser intensity, metric={metric} :")
#     ir.plot_permutation(cfg, file_to_process_list, metric, "low", n_perm)

#     print("="*60)
#     print(f"\nSize effect of HIGH laser intensity, metric={metric} :")
#     ir.plot_permutation(cfg, file_to_process_list, metric, "high", n_perm)




data = pd.DataFrame()
n=0

for i, metrics_path in enumerate(file_to_process_list) :
    metrics = load_trial_data(Path(metrics_path))

    for j, trial in enumerate(metrics) : 

        if not trial[cfg.bodypart]["trial_success"] or trial["laser_intensity"] == "NOstim": 
            continue

        condition = trial["condition"]
        laser_state = trial["laser_state"]
        pad_off = trial["pad_off"]

        if trial["laser_intensity"] == "0,5mW" or trial["laser_intensity"] == "1mW" : laser_intensity = "low" 
        elif trial["laser_intensity"] == "NOstim" : laser_intensity = "NOstim" 
        else : laser_intensity = "high"

        reward = trial["reward"] is not None

        df = pd.DataFrame({
            "rewarded": [reward],
            "cond_state": [condition + "_" + laser_state],
            "condition": [condition],
            "laser_state": [laser_state],
            "laser_intensity": [laser_intensity],
            "id": [n],
        })

        data = pd.concat([data, df], ignore_index=True)

        n+=1

g = pc.plot_rewarded_bar(data)
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.figure.subplots_adjust(top=0.88)
g.figure.suptitle(f"Rewarded trial proportion on of rat {cfg.inter_rat.rats}\nNumber of trials: {len(data.groupby('id'))}", ha='center')
g.savefig(make_output_path(cfg.paths.inter_rat / f"reward", f"rewarded_proportion.png"))
g.savefig(make_output_path(cfg.paths.inter_rat / f"reward", f"rewarded_proportion.svg"))



print("Done !")