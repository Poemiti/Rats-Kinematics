#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from rats_kinematics_utils.file_management import make_name_by_condition, open_DLC_results
from rats_kinematics_utils.trajectory_metrics import Trajectory, crop_xy, filter_likelihood, filter_outliers, interpolate_data

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

def outlier_proportion(df):
    props = {}

    for step, values in df.groupby("step", sort=False)["n"]:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((values < lower) | (values > upper)).sum()
        props[step] = f"{outliers}/{len(values)}"
    return props



def plot_traj(coord, offset, label, color, ax: plt.axes = None, marker: str = None):

    x = coord["x"] - offset
    y = coord["y"] - offset

    if ax is not None : 
        ax.plot(x, y, label=label, color=color)
        if marker : 
            ax.scatter(x, y,marker=marker)
    else : 
        plt.plot(x, y, label=label, color=color)
        if marker : 
            ax.scatter(x, y,marker=marker)



def plot_xy(axes, coords, offset, color,  marker, label) -> None :
    
    t = coords["t"]
    x = coords["x"] + offset
    y = coords["y"] - offset

    axes[0].plot(t, x, marker=marker, color=color, label=label)
    axes[1].plot(t, y, marker=marker, color=color)


def plot_outliers(axes, coords, params) -> None : 

    t = coords["t"]
    x = coords["x"] 
    y = coords["y"] 

    dists, threshold, mask = params

    axes[0].plot(t, x, marker='|', color="#74a9cf", alpha=0.5 )
    axes[0].scatter(t[mask], x[mask], marker="x", color='red')
    axes[0].set_ylabel("x")
    axes[0].set_title("Outlier detection based on euclidian distance")

    # y(t)
    axes[1].plot(t, y, marker='|', color="#74a9cf", label="raw points", alpha=0.5)
    axes[1].scatter(t[mask], y[mask], marker="x", color='red', label="outliers")
    axes[1].set_ylabel("y")
    axes[1].invert_yaxis()
    axes[1].legend()

    # distances
    axes[2].plot(t, dists, marker='o', color="#0570b0", label="distance")
    axes[2].axhline(threshold, color="#fb6a4a", linestyle="--", linewidth=1 ,label="threshold")
    axes[2].set_ylabel("Euclidean distance (pixel)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[2].set_xticks([0, 0.5])
    axes[2].set_yticks([threshold])

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
            
            trial_name = coords_path.stem.replace('pred_results', '')
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

            fig = plt.figure(figsize=(10,6))
            gs = fig.add_gridspec(2, 2)

            ax_xt = fig.add_subplot(gs[0,0])      # x(t)
            ax_yt = fig.add_subplot(gs[1,0])      # y(t)
            ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows
            # ax_speed = fig.add_subplot(gs[2, 0])

            plot_xy([ax_xt, ax_yt], interpolated_coords, 3*offset,"#0570b0", "|", "3.interpolate")
            plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
            plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 1*offset, "#bdc9e1","|", "1.outlier")
            plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw")
            
            plot_traj(raw_coords[:63], 0*offset, "raw", "#d1cbdc", ax_traj)
            plot_traj(interpolated_coords[:63], 0*offset, "interpolate", "#0570b0" ,ax_traj, "|")

            # ax_speed.plot(raw_coords["t"], speed, label="speed", marker="|")
            # ax_speed.plot(raw_coords["t"][peaks], speed[peaks], "x")

            ax_xt.set_ylabel("x")
            ax_xt.legend()
            ax_xt.set_xlim(0, 0.5)
            ax_xt.set_title(old_foldername)

            ax_yt.set_ylabel("y")
            ax_yt.set_xlim(0, 0.5)
            ax_yt.invert_yaxis()
            ax_yt.set_xlabel("time (s)")

            ax_traj.set_title("Trajectory comparaison between\nraw and final interpolated")
            ax_traj.set_xlabel("x")
            ax_traj.set_ylabel("y")
            ax_traj.invert_yaxis()
            ax_traj.legend()
            ax_traj.set_aspect("equal")
            
            # ax_speed.set_ylabel("time")
            # ax_speed.set_xlabel("time (s)")
            # ax_speed.set_xlim(0, 0.5)
            # ax_speed.legend()

            for ax in [ax_xt, ax_yt, ax_traj]:
                ax.set_xticks([])
                ax.set_yticks([])

            ax_yt.set_xticks([0, 0.5])

            plt.tight_layout()
            fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / old_foldername / "preprocessing", f"{trial_name}_interpolation.png"))
            plt.close()

            fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
            plot_outliers(axes, raw_coords, params)
            plt.tight_layout()
            plt.gca().set_xlim(0,0.5)
            fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / old_foldername /  "preprocessing", f"{trial_name}_outliers.png"))
            plt.close()

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

        props = outlier_proportion(data)
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

    props = outlier_proportion(all_data)
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


