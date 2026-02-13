#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rats_kinematics_utils.file_management import verify_exist, get_session
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, plot_violin_distribution_velocity, plot_stacked_trajectories, plot_velocity_over_sessiontime, plot_velocity_over_cliptime
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice


RAT_NAME = "#525"
SHOW = True
cfg = load_config()

filenames = sorted((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))
n_trial = 0
total_trial = 0

fig, ax = plt.subplots(figsize=(9, 7))

for i, metrics_path in enumerate(filenames) :

    metrics_path = Path(metrics_path) 
    output_fig_dir = cfg.paths.figures / RAT_NAME / metrics_path.stem

    print(f"\n[{i+1}/{len(filenames)}]")
    print(f"Making figures of {metrics_path.parent.stem}\n")

    metrics = load_metrics(metrics_path)
    plot_stacked_trajectories(cfg, metrics, ax)
    n_trial += sum(1 for m in metrics if m.get('trial_success'))
    total_trial += len(metrics)
    
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)


ax.tick_params(direction="out")

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title(f"Stacked Trajectories of {RAT_NAME}\nnumber of trial = {n_trial} / {total_trial}")

ax.invert_xaxis()


fig = ax.figure
plt.show()
plt.close(fig)