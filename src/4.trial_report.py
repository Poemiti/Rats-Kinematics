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

        if t.get("pad_off") is None:
            failure_counter["no_pad_off"] += 1

        if t.get("reward") is None:
            failure_counter["no_reward"] += 1

        if t.get("lost_coords", 0) > 0:
            failure_counter["lost_coords"] += 1

    report = {
        "total_trials": total_trials,
        "successful_trials": success_count,
        "failed_trials": failed_trials,
        "success_rate": success_count / total_trials if total_trials > 0 else 0,
        "failure_reason": dict(failure_counter)
    }

    return report




def make_visualisation(all_reports: dict):

    rows = []
    summary_rows = []

    for session_name, report in all_reports.items():

        # failure breakdown
        for reason, count in report["failure_reason"].items():
            rows.append({
                "Session": session_name,
                "Failure Type": reason,
                "Count": count
            })

        # session summary
        summary_rows.append({
            "Session": session_name,
            "Failed": report["failed_trials"],
            "Total": report["total_trials"]
        })

    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="darkgrid", rc=custom_params)

    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=df,
        x="Failure Type",
        y="Count",
        hue="Session"
    )

    # ---- Add annotation per session ----
    y_max = df["Count"].max()

    for i, row in summary_df.iterrows():
        ax.text(
            x=i,  # session index (position in hue order)
            y=y_max - 1,
            s=f"{row['Failed']}/{row['Total']}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.legend(
        title="Condition",
        # bbox_to_anchor=(1.02, 1),  # move right
        loc="upper right",
        borderaxespad=0,
        fontsize=6,
        ncol=2
    )

    plt.title("Failure Reasons per Session")
    plt.ylabel("Number of Failed Trials")
    plt.grid(True)
    plt.yticks(rotation=90)
    plt.show()




all_reports = {}
for i, metrics_path in enumerate(filenames) :

    metrics_path = Path(metrics_path) 
    metrics = load_metrics(metrics_path)

    report = trial_report(metrics)
    # all_reports[metrics_path.stem] = report
    all_reports[metrics[0]["condition"]] = report


make_visualisation(all_reports)

