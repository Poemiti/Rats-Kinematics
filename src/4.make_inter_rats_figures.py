#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re, yaml

import rats_kinematics_utils.analysis.plot_comparative as plot_comparative
from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import dataframe_report
from rats_kinematics_utils.analysis.inter_rat import filter_contra_trials, _preprocess, plot_statistics, plot_permutation

# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()

file_to_process = filter_contra_trials(cfg)

print(f"\nThe following files will be used for the inter rat analysis:")
for r, f_list in file_to_process.items(): 
    print(r)
    for f in f_list: 
        print(f"  {f.stem}")
print()


# --------------------------------------- main -------------------------------------------

file_to_process_list = [item for lst in file_to_process.values() for item in lst]

################## setup + print d'info

metric = "average_velocity"
data = _preprocess(cfg, file_to_process_list, metric, split_condition=False)
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
plot_statistics(cfg, file_to_process_list, "average_velocity", comparisons)

print(f"\n ------ computing statistics on tortuosity ------\n")
plot_statistics(cfg, file_to_process_list, "tortuosity", comparisons)



# ################# linear model try


# result = LMM(data, "value ~ condition * laser_state * laser_intensity")
# print(result.summary())



# ################### permutation

n_perm = 100000

print("="*60)
print(f"\nSize effect of LOW laser intensity, metric={metric} :")
plot_permutation(cfg, file_to_process_list, metric, "low", n_perm)

print("="*60)
print(f"\nSize effect of HIGH laser intensity, metric={metric} :")
plot_permutation(cfg, file_to_process_list, metric, "high", n_perm)
