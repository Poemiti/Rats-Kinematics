#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from rats_kinematics_utils.file_management import make_name_by_condition, open_DLC_results
from rats_kinematics_utils.trajectory_metrics import Trajectory, crop_xy, filter_likelihood, filter_outliers, interpolate_data
from rats_kinematics_utils.plot_preprocess import make_interpolation_figures, make_outlier_figures, _outlier_proportion
from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_database, print_analysis_info, make_output_path

# ------------------------------------ plot choice ---------------------------------

FILTRATION = True
DISTRI = False


# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Computing metrics")


RAT_NAME = cfg.rat_name
DATABASE = load_database(cfg.paths.coords / RAT_NAME, cfg.paths.database, "csv")

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)


filenames = (
    DATABASE.sort_values(
        by=["rat_name", "rat_type", "condition", "task", "laser_intensity", "laser_on"],
        ascending=[True, True, True, True, True, True], 
    )
    ["filename"]
    .tolist()
)

# ------------------------------------ plotting function ---------------------------------------


# ------------------------------------ filtration ---------------------------------------

counts = []
folder_names = {}
old_foldername =  make_name_by_condition(Path(DATABASE["filename"][0]).stem)

if FILTRATION : 

    if (cfg.paths.metrics / "preprocessing" / f"{RAT_NAME}_preprocessing_removal_counts.joblib").exists() : 
        print("loading data")
        folder_names = joblib.load(cfg.paths.metrics / "preprocessing" / f"{RAT_NAME}_preprocessing_removal_counts.joblib")
    else : 
        print(f"Number of files : {len(filenames)}")
        for i, coords_path in enumerate(filenames[0:]) : 
            
            # get all the path (coordinates, luminosity and video clips)
            coords_path = Path(coords_path)
            
            trial_name = coords_path.stem.replace('pred_results_', '')
            print(f"\n[{i}/{len(filenames)}]\n{trial_name}\n")

            # get coords + filtration
            raw_coords = open_DLC_results(coords_path)
            raw_coords = raw_coords[cfg.bodypart].copy()
            raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / 125)
            print(f"\nraw : {len(raw_coords)}")
            
            # outlier_filtered_coords, peaks, speed = filter_outliers(raw_coords, stat_method='rolling_mad')
            outlier_filtered_coords, params = filter_outliers(raw_coords, stat_method='eucli')
            print(f"\nafter outlier filtration : {outlier_filtered_coords['x'].count().sum()}")

            likelihood_filtered_coords = filter_likelihood(outlier_filtered_coords, cfg.threshold)
            print(f"after likelihood filtration : {likelihood_filtered_coords['x'].count().sum()}")

            interpolated_coords = interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)
            print(f"\nafter interpolation : {interpolated_coords['x'].count().sum()}")

            # if new condition, save metrics.yaml + initialise metrics dictionary + make new folder
            new_foldername = make_name_by_condition(coords_path.stem)
            if new_foldername != old_foldername : 
                if old_foldername in folder_names.keys() : 
                    for c in counts :
                        folder_names[old_foldername].append(c)
                else : 
                    folder_names[old_foldername] = counts

                old_foldername = new_foldername
                counts = []
                print(f"File will be stored in {new_foldername}")

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

            offset = 10

            make_interpolation_figures(interpolated_coords, 
                               likelihood_filtered_coords,
                               outlier_filtered_coords,
                               raw_coords,
                               title=old_foldername, 
                               save_as=make_output_path(cfg.paths.figures / RAT_NAME / old_foldername / "preprocessing", f"{trial_name}_interpolation.png"))

            make_outlier_figures(raw_coords, params, 
                                 save_as=make_output_path(cfg.paths.figures / RAT_NAME / old_foldername /  "preprocessing", f"{trial_name}_outliers.png"))

        # final saving after the end of the loop
        if old_foldername in folder_names.keys() : 
            for c in counts :
                folder_names[old_foldername].append(c)
        else : 
            folder_names[old_foldername] = counts


        joblib.dump(folder_names, make_output_path(cfg.paths.metrics / "preprocessing" , f"{RAT_NAME}_preprocessing_removal_counts.joblib"))

    ########################## distribution ######################

    print(f"\n---------DISTRIBUTION--------\n")
    sns.set_theme("paper", style="whitegrid", palette="pastel")

    all_counts = []

    for folder, count in folder_names.items() : 
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
                y_max,
                props[step],
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_title(f"Distribution of the number of coordinates of\n{folder}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of removed points")

        # plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / folder / "preprocessing", f"point_removal_distribution.png"))
        plt.close()


    all_data = pd.DataFrame(all_counts)
    print(all_data.head)

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

    sns.stripplot(
        data=all_data,
        x="step",
        y="n",
        marker="X",
        size=3,
        alpha=0.7,
        color="black"
    )

    props = _outlier_proportion(all_data)
    y_max = all_data["n"].max()

    for i, step in enumerate(props.keys()):
        ax.text(
            i,
            y_max,
            props[step],
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_title(f"Distribution of the number of coordinates of {RAT_NAME}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of removed points")

    # plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "preprocessing", f"{RAT_NAME}_point_removal_distribution.png"))
    plt.close()




    

if DISTRI : 


    # ------------------------------------ distribution ---------------------------------------

    likelihood_distri = []

    print(f"Number of files : {len(filenames)}")

    for i, coords_path in enumerate(filenames[6:]):

        # loading
        coords_path = Path(coords_path)
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

    ax.set_title(f"Distribution of likelihood across bodyparts of rat {RAT_NAME}")
    ax.set_xlabel("Bodyparts")
    ax.set_ylabel("Likelihood")

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME, f"bodypart_likelihood_distribution.png"))
    plt.close()


