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

sns.set_theme("paper", style="whitegrid", palette="pastel")

# ---------------- function ------------


def trial_report(trials: list[dict]) -> dict:

    def new_block(intensities):
        return {
            "Total": 0,
            "LaserOn": {"Total": 0, **{i: 0 for i in intensities}},
            "LaserOff": {"Total": 0, **{i: 0 for i in intensities}},
        }

    def init_group(intensities):
        return {
            "Total": 0,
            "Successful": new_block(intensities),
            "Rejected": new_block(intensities),
            "No reward": new_block(intensities),
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

        laser_state = t["laser_state"]

        # Successful
        if t[cfg.bodypart]["trial_success"]:
            update(group, "Successful", laser_state, intensity)
        else : 
            update(group, "Rejected", laser_state, intensity)
        
        # No pad off
        if not t["pad_off"]:
            update(group, "No pad off", laser_state, intensity)

        # No reward
        if not t["reward"]:
            update(group, "No reward", laser_state, intensity)

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

            for laser_state in ["LaserOn", "LaserOff"]:
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
        margin_titles=True,
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
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "Count")
    g.set_xticklabels(["Successful", "Rejected", "No reward", "No pad off"])

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
report_path = output_dir / "trial_report_2.yaml"

# if not report_path.exists() : 
print("Making trial report ...")
all_trials = []
for i, metrics_path in enumerate(filenames) :
    metrics_path = Path(metrics_path) 
    print(metrics_path.stem)
    metrics = load_metrics(metrics_path)
    for trial in metrics : 
        all_trials.append(trial)

print("True number of trials :", len(all_trials))
report = trial_report(all_trials)

# save report
with open(output_dir / report_path, "w") as file :
    yaml.dump(report, file, default_flow_style=False, indent=4, sort_keys=False)


# plot report
print(f"Loading {report_path} and plotting")
plot_trial_report(output_dir / report_path,
                  output_dir / "trial_report.png")

