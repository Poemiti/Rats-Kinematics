#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re

import rats_kinematics_utils.plot_inter_rats as plot_inter_rats
from rats_kinematics_utils.file_management import verify_exist, get_session
from rats_kinematics_utils.plot_comparative import plot_stacked_velocity, plot_stacked_Yposition, _plot_violin_distribution, plot_stacked_trajectories, plot_velocity_over_cliptime, _plot_displot
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, load_figure_maker, make_output_path, check_analysis_choice, print_analysis_info, print_interRat_analysis_info
from rats_kinematics_utils.statistics import ANOVA


# ------------------------------------ setup ---------------------------------------

SHOW = True
cfg = load_config()
print_analysis_info(cfg, "Making inter-rats figures")


root = cfg.paths.metrics
pattern = re.compile(r"#\d{3}")

filenames = sorted(
    p for p in root.glob("#*/")      # only first-level folders
    if pattern.fullmatch(p.name)
    for p in p.glob("*.joblib")      # only files directly inside
)

available_functions = {name: func
                for name, func in inspect.getmembers(plot_inter_rats, inspect.isfunction)
                if name.startswith("plot")}
RAT_NAME = cfg.rat_name

check_analysis_choice(filenames, available_functions)
print_interRat_analysis_info(filenames, available_functions)

