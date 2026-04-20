#!/usr/bin/env python

import sys
import pandas as pd

import rats_kinematics_utils.preprocessing.plot_preprocess as pp

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import dataframe_report
import rats_kinematics_utils.analysis.inter_rat as ir
# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()

file_to_process = ir.filter_contra_trials(cfg, single_rat=False)


print(f"\nThe following files will be used for the inter rat analysis:")
for r, f_list in file_to_process.items(): 
    print(r)
    for f in f_list: 
        print(f"  {f.stem}")
print()

res = input("\nENTER to launch preprocessing or (q) to quit : ")
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

    print(f"\nPlotting overall trials failure reason\n")
    pp.plot_trial_failure_reason(cfg, file_to_process_list, inter_rat=True)

    print(f"\nPlotting overall bodyparts likelihood distribution\n")
    ir.plot_likelihood_distri_interrat(cfg, file_to_process_list)

elif res not in ["y", "n", "Y", "N"] : 
    raise ValueError(f"ERROR, '{res}' is not a valid answer, must be 'y' or 'n'")


# --------------------------------------- main -------------------------------------------


################## setup + print d'info

metric = "average_velocity"
data = ir._preprocess(cfg, file_to_process_list, metric, split_condition=False)
print(data.head(5))
info = dataframe_report(data)

for col, info in info.items():
    print(f"\nColumn: {col}")
    print(info['summary'].T)


condition = ["Beta", "Conti"]
l_state = data["laser_state"].unique()
l_intensity = ["low", "high"]

for intensity in l_intensity :  
    print(f"\n{intensity} :")
    for state in l_state : 
        for c in condition : 
            size = len(data[(data["condition"] == c) &
                            (data["laser_state"] == state) &
                            (data["laser_intensity"] == intensity)])
            print(f"  n {c} {state} : {size}")

################# normal statistics

comparisons = [
            # Conti vs Beta
            ("Conti_LaserOff.low",  "Beta_LaserOff.low"),
            ("Conti_LaserOff.high", "Beta_LaserOff.high"),
            ("Conti_LaserOn.low",   "Beta_LaserOn.low"),
            ("Conti_LaserOn.high",  "Beta_LaserOn.high"),

            # Off vs On
            ("Conti_LaserOff.low",  "Conti_LaserOn.low"),
            ("Conti_LaserOff.high", "Conti_LaserOn.high"),
            ("Beta_LaserOff.low",   "Beta_LaserOn.low"),
            ("Beta_LaserOff.high",  "Beta_LaserOn.high"),

            # low vs high
            ("Beta_LaserOff.low",   "Beta_LaserOff.high"),
            ("Beta_LaserOn.low",    "Beta_LaserOn.high"),
            ("Conti_LaserOff.low",  "Conti_LaserOff.high"),
            ("Conti_LaserOn.low",   "Conti_LaserOn.high"),
        ]

print(f"\n ------ computing statistics on average velocity ------\n")
ir.plot_statistics(cfg, file_to_process_list, "average_velocity", comparisons)

print(f"\n ------ computing statistics on tortuosity ------\n")
ir.plot_statistics(cfg, file_to_process_list, "tortuosity", comparisons)


# ################### permutation

n_perm = 100000

for metric in ["average_velocity", "tortuosity"] :

    print("="*60)
    print(f"\nSize effect of LOW laser intensity, metric={metric} :")
    ir.plot_permutation(cfg, file_to_process_list, metric, "low", n_perm)

    print("="*60)
    print(f"\nSize effect of HIGH laser intensity, metric={metric} :")
    ir.plot_permutation(cfg, file_to_process_list, metric, "high", n_perm)


