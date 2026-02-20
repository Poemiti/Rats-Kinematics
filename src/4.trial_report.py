#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, print_analysis_info


# ------- plot styling -------

sns.set_theme(
    style="darkgrid",
    palette="pastel",
)


# ---------------- function ------------


def trial_report(trials: list[dict]) -> dict:
    from collections import defaultdict

    def new_block(intensities):
        return {
            "Total": 0,
            "Laser ON": {"Total": 0, **{i: 0 for i in intensities}},
            "Laser OFF": {"Total": 0, **{i: 0 for i in intensities}},
        }

    def init_group(intensities):
        return {
            "Total": 0,
            "Successful": new_block(intensities),
            "No reward": new_block(intensities),
            "Too much lost coords": new_block(intensities),
            "No pad off": new_block(intensities),
        }

    def update(group, outcome, laser_state, intensity):
        group["Total"] += 1
        group[outcome]["Total"] += 1
        group[outcome][laser_state]["Total"] += 1
        group[outcome][laser_state][intensity] += 1


    report = {
        "Total number of trials": len(trials),
        "Beta trials": init_group(["1mW", "2,5mW"]),
        "Conti trials": init_group(["0,5mW", "0,75mW", "2,5mW"]),
        "NOstim trials": init_group(["NOstim"]),
    }

    #--------------------- loop ---------------------------

    for t in trials:
        condition = t["condition"]
        intensity = t["laser_intensity"]

        if "Beta" in condition:
            group = report["Beta trials"]
        elif "Conti" in condition:
            group = report["Conti trials"]
        else : 
            group = report["NOstim trials"]

        laser_state = "Laser ON" if "On" in condition else "Laser OFF"

        # Successful
        if t["trial_success"]:
            try : 
                update(group, "Successful", laser_state, intensity)
            except :
                raise KeyError(f"{condition} {laser_state} {intensity}")
        # No pad off
        elif not t["pad_off"]:
            update(group, "No pad off", laser_state, intensity)

        # No reward
        elif not t["reward"]:
            update(group, "No reward", laser_state, intensity)

        # No coords
        elif not t["lost_coords"]:
            update(group, "Too much lost coords", laser_state, intensity)

    return report



def plot_trial_report(yaml_file: Path, output_path: Path) : 
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    rows = []
    total = 0

    for trial_type in ["Beta trials", "Conti trials", "NOstim trials"]:
        trial_block = data[trial_type]

        for outcome in trial_block:
            if outcome == "Total":
                continue

            outcome_block = trial_block[outcome]

            for laser_state in ["Laser ON", "Laser OFF"]:
                laser_block = outcome_block[laser_state]

                for power, value in laser_block.items():
                    if power == "Total":
                        total = power
                        continue

                    rows.append({
                        "Trial Type": trial_type.replace(" trials", ""),
                        "Outcome": outcome,
                        "Laser": laser_state.replace("Laser ", ""),
                        "Laser intensity": power,
                        "Count": value
                    })

    df = pd.DataFrame(rows)

    g = sns.catplot(
        data=df,
        kind="bar",
        x="Outcome",
        y="Count",
        hue="Laser intensity",
        col="Trial Type",
        row="Laser",
        palette="pastel",
        height=4,
        aspect=1,
    )

    for ax in g.axes.flat:
        total = 0
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=10)
            total += sum(bar.get_height() for bar in container)

        ax.text(
            0.95, 0.95,
            f"N={int(total)}",
            transform=ax.transAxes,   # important!
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

    g.set_xticklabels(rotation=45, fontsize=9)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", fontsize=10)
    g.set_axis_labels("", "Count")
    g.set_xticklabels(["Successful", "No reward", "Lost coords", "No pad off"])

    g.savefig(output_path)
    plt.show()
    plt.close()



# -------------------------------- main -----------------------------------------


cfg = load_config()
print_analysis_info(cfg, "Trials reporting")


RAT_NAME = cfg.rat_name
filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))
output_dir = cfg.paths.report / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "trial_report.yaml"

if report_path.exists() : 
    print("Making trial report ...")
    all_trials = []
    for i, metrics_path in enumerate(filenames) :
        metrics_path = Path(metrics_path) 
        metrics = load_metrics(metrics_path)
        for trial in metrics : 
            all_trials.append(trial)

    print("True number of trials :", len(all_trials))
    report = trial_report(all_trials)
    print(report.keys())
    # save report
    with open(output_dir / "trial_report.yaml", "w") as file :
        yaml.dump(report, file, default_flow_style=False, indent=4, sort_keys=False)


# plot report
print(f"Loading {report_path} and plotting")
plot_trial_report(output_dir / "trial_report.yaml",
                  output_dir / "trial_report.png")

