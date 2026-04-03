#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm


from rats_kinematics_utils.file_management import make_name_by_condition, open_DLC_results
from rats_kinematics_utils.trajectory_metrics import Trajectory, crop_xy, filter_likelihood, filter_outliers, interpolate_data
from rats_kinematics_utils.plot_preprocess import make_interpolation_figures, make_outlier_figures, _outlier_proportion
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, print_analysis_info, make_output_path

# ------------------------------------ plot choice ---------------------------------

FILTRATION = False
DISTRI = False
LIKELIHOOD = False
SUCCESS_DISTRI = True

# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Computing metrics")


RAT_NAME = cfg.rat_name

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)


filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))
# filenames = list((Path("data_V1") / "metrics_results" / RAT_NAME).glob("*.joblib"))
preprocess_counts_path = make_output_path(cfg.paths.metrics / "preprocessing" , f"{RAT_NAME}_preprocessing_removal_counts3.joblib")

# ------------------------------------ filtration ---------------------------------------


folder_names = {}

if FILTRATION : 
    print(f"Number of files : {len(filenames)}")
    for i, session in enumerate(filenames) : 
        folder_name = session.stem
        counts = []

        session_data = joblib.load(session)
        print(f"\nProcessing {folder_name}")
        for trial in tqdm(session_data, desc="Trials"):

            date = datetime.fromisoformat(trial["date"])
            if date.month == 5 : 
                continue
                
            # get all the path (coordinates, luminosity and video clips)
            coords_path = trial["filename_coords"]
            trial_name = trial["name"]

            # get coords + filtration
            raw_coords = open_DLC_results(coords_path)
            raw_coords = raw_coords[cfg.bodypart].copy()
            raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / cfg.fps)
            # print(f"\nraw : {len(raw_coords)}")
            
            # outlier_filtered_coords, peaks, speed = filter_outliers(raw_coords, stat_method='rolling_mad')
            outlier_filtered_coords, params = filter_outliers(raw_coords, stat_method='eucli')
            # print(f"after outlier filtration : {outlier_filtered_coords['x'].count()}")

            likelihood_filtered_coords, likelihood_threshold = filter_likelihood(outlier_filtered_coords, cfg.threshold)
            # print(f"after likelihood filtration : {likelihood_filtered_coords['x'].count()}")

            interpolated_coords = interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)
            # print(f"after interpolation : {interpolated_coords['x'].count()}")

            # save number of removal
            counts.append({"id": trial_name, "step": "raw", "n": 0})
            steps = {
                "outlier": outlier_filtered_coords,
                "likelihood": likelihood_filtered_coords,
                "interpolate": interpolated_coords,
            }

            for step, df in steps.items():
                counts.append({
                    "id": trial_name,
                    "step": step,
                    "n": len(raw_coords) - df['x'].count().sum(),
                })

        # final saving after the end of the loop
        if folder_name in folder_names.keys() : 
            folder_names[folder_name].extend(counts)
        else : 
            folder_names[folder_name] = counts

    joblib.dump(folder_names, preprocess_counts_path)

    ########################## distribution ######################

    print(f"\n---------DISTRIBUTION--------\n")
    sns.set_theme("paper", style="whitegrid", palette="pastel")

    all_counts = []

    for folder, count in folder_names.items() : 

        if len(count) == 0: 
            print(f"{folder}: no data")
            print(count)
            continue

        all_counts.extend(count)
        data = pd.DataFrame(count)

        fig, ax = plt.subplots()
        

        sns.boxplot(
            data=data,
            x="step",
            y="n",
            hue="step",
            showfliers=False,
            ax=ax
        )

        sns.stripplot(
            data=data,
            x="step",
            y="n",
            marker="X",
            size=3,
            alpha=0.7,
            color="black"
        )

        props = _outlier_proportion(data)
        y_max = data["n"].max()

        for i, step in enumerate(props.keys()):
            ax.text(
                i,
                0.92,
                props[step],
                ha="center",
                va="bottom",
                fontsize=9,
                transform=ax.get_xaxis_transform()
            )

        ax.set_title(f"Distribution of the number of coordinates of\n{folder}, thresh={likelihood_threshold:.2f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of removed points")

        if folder == "#525_CHR_Conti_RightHemi_H001_0,5mW_LaserOn" : 
            ax.set_ylim(-0.5, 50)
        else : 
            ax.set_ylim(-0.5, 20)

        # plt.xticks(rotation=45)
        # plt.tight_layout()
        fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME ,  f"{folder}_distri.png"))
        plt.close()


    all_data = pd.DataFrame(all_counts)

    fig, ax = plt.subplots()
    sns.set_theme("paper", style="whitegrid", palette="pastel")

    sns.boxplot(
        data=all_data,
        x="step",
        y="n",
        hue="step",
        showfliers=False,
        ax=ax
    )

    # sns.stripplot(
    #     data=all_data,
    #     x="step",
    #     y="n",
    #     marker="X",
    #     size=3,
    #     alpha=0.7,
    #     color="black"
    # )

    props = _outlier_proportion(all_data)
    y_max = all_data["n"].max()

    for i, step in enumerate(props.keys()):
        ax.text(
            i,
            0.92,
            props[step],
            ha="center",
            va="bottom",
            fontsize=9, 
            transform=ax.get_xaxis_transform()
        )

    ax.set_title(f"Distribution of the number of coordinates of {RAT_NAME}\nlikelihood threshold={likelihood_threshold:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of removed points")

    # ax.set_ylim(-0.5, 20)

    # plt.xticks(rotation=45)
    # plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME, f"{RAT_NAME}_point_removal_distribution.png"))
    plt.close()




    

if DISTRI : 


    # ------------------------------------ distribution ---------------------------------------

    likelihood_distri = []

    print(f"Number of files : {len(filenames)}")

    for i, session in enumerate(filenames):

        for trial in joblib.load(session) :
            date = datetime.fromisoformat(trial["date"])
            if date.month == 5 : 
                continue

            # loading
            coords_path = Path(trial['filename_coords'])
            raw_coords = open_DLC_results(coords_path)
            bodyparts = raw_coords.columns.get_level_values(0).unique()

            for bp in bodyparts[1:]:
                likelihoods = raw_coords[bp]["likelihood"]

                for val in likelihoods:
                    likelihood_distri.append({
                        "bodypart": bp,
                        "likelihood": val
                    })

    data = pd.DataFrame(likelihood_distri)


    fig, ax = plt.subplots()

    sns.violinplot(
        data=data,
        x="bodypart",
        y="likelihood",
        inner="quart",
        ax=ax
    )

    ax.axhline(likelihood_threshold, linestyle="--", color="red", label="likelihood threshold", lw="0.8")

    ax.set_title(f"Distribution of likelihood across bodyparts of rat {RAT_NAME}\nlikelihood threshold={likelihood_threshold:.2f}")
    ax.set_xlabel("Bodyparts")
    ax.set_ylabel("Likelihood")
    ax.legend(loc="lower right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME, f"{RAT_NAME}_bodypart_likelihood_distribution.png"))
    plt.close()



if LIKELIHOOD : 


    # ------------------------------------ likelihood distribution ---------------------------------------

    

    print(f"Number of files : {len(filenames)}")

    for i, session in enumerate(filenames):
        session_data = joblib.load(session)
        folder = session.stem

        print(f"Processing {folder}")

        for trial in tqdm(session_data, desc="Trials") :
            date = datetime.fromisoformat(trial["date"])
            if date.month == 5 : 
                continue

            # loading
            coords_path = Path(trial['filename_coords'])
            raw_coords = open_DLC_results(coords_path)
            bodyparts = raw_coords.columns.get_level_values(0).unique()

            raw_coords = raw_coords[cfg.bodypart]
            _, likelihood_threshold = filter_likelihood(raw_coords, cfg.threshold, percentile=0.80)

            data = pd.DataFrame(raw_coords)

            fig, ax = plt.subplots()

            ax.plot(data["likelihood"], color="lightblue")

            ax.axhline(likelihood_threshold, linestyle="--", color="red", 
                    label="likelihood threshold", lw="0.8")

            ax.set_title(f"likelihood distribution across frame\nclip number: {trial['nb_clip']}, likelihood threshold: {likelihood_threshold:.2f}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Likelihood")
            ax.legend(loc="lower right")

            plt.xticks()
            plt.tight_layout()
            fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / folder / "likelihood_distri", f"{trial['name']}_likelihood.png"))
            plt.close()





if SUCCESS_DISTRI : 

    all_state = {"raw" : 0,
                "interpolate": 0,
                "rejected": 0, 
                "None": 0}

    for session in filenames :
        
        for trial in joblib.load(session) :
            body = trial.get(cfg.bodypart)

            if body is None : 
                print("BODY NONE")
                continue

            success = body.get("trial_success")
            state = body.get("xy_state")

            if state is None : 
                print(trial["name"])
                all_state["None"] += 1
                continue

            all_state[state] += 1

    df = pd.DataFrame({
        "State": list(all_state.keys()),
        "Count": list(all_state.values())
    })

    order = ["rejected", "raw", "interpolate", "None"]

    fig = plt.figure(figsize=(8, 5))
    sns.set_theme("paper", style="whitegrid", palette="pastel")

    ax = sns.barplot(
        data=df,
        x="State",
        y="Count",
        palette="pastel",
        order=order,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%d')

    ax.set_title(f"Trial State Distribution of Rat {RAT_NAME}")
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Trials")

    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME, f"{RAT_NAME}_trial_state_distribution.png"))
    plt.show()
    plt.close()

print("Done !")