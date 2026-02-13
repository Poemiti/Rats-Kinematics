#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics

# ------------------------------------ setup ---------------------------------------

RAT_NAME = "#525"
cfg = load_config()
filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))


def trial_report(trials: list[dict]) -> dict:

    total_trials = len(trials)
    success_count = sum(t.get("trial_success", False) for t in trials)
    failed_trials = total_trials - success_count

    failure_counter = Counter()

    for t in trials:
        if t["trial_success"]:
            continue  # skip successful trials

        if not t["pad_off"]:
            failure_counter["no_pad_off"] += 1
            continue

        if not t["reward"]:
            failure_counter["no_reward"] += 1
            continue

        if t["lost_coords"] > 0:
            failure_counter["lost_coords"] += 1
            continue

        

    report = {
        "laser_intensity" : trials[0]["laser_intensity"],
        "total_trials": total_trials,
        "successful_trials": success_count,
        "failed_trials": failed_trials,
        "success_rate": success_count / total_trials if total_trials > 0 else 0,
        "failure_reason": dict(failure_counter)
    }

    return report


def annotate(data, **kws):
    ax = plt.gca()
    ax.text(.23, .9, f"{int(data['failed_trials'].values[0])}/{int(data['total_trials'].values[0])}",
            transform=ax.transAxes)
    ax.text(.63, .9, f"{int(data['failed_trials'].values[1])}/{int(data['total_trials'].values[1])}",
            transform=ax.transAxes)


################################################ prep data

all_reports = []
for i, metrics_path in enumerate(filenames) :

    metrics_path = Path(metrics_path) 
    metrics = load_metrics(metrics_path)

    report = trial_report(metrics)
    report["condition"] = metrics[0]["condition"]

    all_reports.append(report)

df_long = pd.DataFrame(all_reports)
df_long[["condition", "laser_state"]] = (df_long["condition"].str.split("_", expand=True))
print(df_long)

df_failures = (
    df_long
    .set_index(["condition", "laser_intensity", "laser_state"])
    ["failure_reason"]
    .apply(pd.Series)           # convert dict → columns
    .stack()                    # pivot to long
    .reset_index()
)

df_failures.columns = [
    "condition",
    "laser_intensity",
    "laser_state",
    "failure_type",
    "count"
]
print(df_failures)


laser_color = {
    "1mW":  '#a1c9f4',
    "0,75mW" :  '#a1c9f4',
    "2,5mW": "#ff9f9b",
    "0,5mW" : "#ff9f9b",
}


######################################################### ploting


sns.set_theme(style="darkgrid")

g1 = sns.FacetGrid(
    data=df_long,
    row="laser_state",
    col="condition",
    height=4,
    aspect=1,
    margin_titles=True,
    palette="pastel",
    legend_out=True,
)

laser_color = {
    "1mW":  '#a1c9f4',
    "0,75mW" :  '#a1c9f4',
    "2,5mW": "#ff9f9b",
    "0,5mW" : "#ff9f9b",
}

g1.map_dataframe(
    sns.barplot,
    y="failed_trials",
    hue="laser_intensity",
    palette=laser_color)


g1.map_dataframe(annotate)
# g1.add_legend(title="Laser intensity :\nBeta                   Conti",
#              ncol=2)
g1.set_titles(col_template="{col_name}", row_template="{row_name}")
g1.set_axis_labels("", "Overall Failed trial count")
g1.tight_layout()

################

g2 = sns.FacetGrid(
    data=df_failures,
    row="laser_state",
    col="condition",
    height=4,
    aspect=1,
    margin_titles=True,
    palette="pastel",
    legend_out=True,
)
g2.map_dataframe(
    sns.barplot,
    x="failure_type",
    y="count",
    hue="laser_intensity",
    palette=laser_color)

g2.add_legend(title="Laser intensity :\nBeta                   Conti",
             ncol=2)
g2.set_titles(col_template="{col_name}", row_template="{row_name}")
g2.set_axis_labels("", "Failed trials count")
g2.tight_layout()



plt.show()
plt.close("all")

