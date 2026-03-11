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


old_filename =  make_name_by_condition(Path(DATABASE["filename"][0]).stem)

filenames = (
    DATABASE.sort_values(
        by=["rat_name", "rat_type", "condition", "task", "laser_intensity", "laser_on"],
        ascending=[True, True, True, True, True, True], 
    )
    ["filename"]
    .tolist()
)

if FILTRATION : 
    # ------------------------------------ filtration ---------------------------------------

    print(f"Number of files : {len(filenames)}")
    for i, coords_path in enumerate(filenames[6:]) : 
        
        # get all the path (coordinates, luminosity and video clips)
        coords_path = Path(coords_path)
        
        trial_name = coords_path.stem.replace('pred_results', '')
        print(f"\nFilename {i}: {trial_name}\n")

        # get coords + filtration
        raw_coords = open_DLC_results(coords_path)
        raw_coords = raw_coords[cfg.bodypart].copy()
        raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / 125)
        print(f"\nraw : {len(raw_coords)}")
        print(raw_coords.head(5))
        
        

        # outlier_filtered_coords, peaks, speed = filter_outliers(raw_coords, stat_method='rolling_mad')
        outlier_filtered_coords = filter_outliers(raw_coords, stat_method='eucli')
        outlier_filtered_coords.to_csv("outlier_filtered.csv")
        print(f"\nafter outlier filtration : {outlier_filtered_coords['x'].count().sum()}")
        print(outlier_filtered_coords.head(5))

        likelihood_filtered_coords = filter_likelihood(outlier_filtered_coords, cfg.threshold, percentile=0.95)
        likelihood_filtered_coords.to_csv("likelihood_filtered.csv")
        print(f"after likelihood filtration : {likelihood_filtered_coords['x'].count().sum()}")
        print(likelihood_filtered_coords.head(5))

        interpolated_coords = interpolate_data(outlier_filtered_coords, method="spline", max_gap=5)
        interpolated_coords.to_csv("interpolated_filtered.csv")
        print(f"\nafter interpolation : {interpolated_coords['x'].count().sum()}")
        print(interpolated_coords.head(5))


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

        offset = 10

        def plot_xy(axes, coords, offset, color,  marker, label) :
            
            t = coords["t"]
            x = coords["x"] + offset
            y = coords["y"] - offset

            axes[0].plot(t, x, marker=marker, color=color, label=label)
            axes[1].plot(t, y, marker=marker, color=color)

        
        fig = plt.figure(figsize=(10,6))
        gs = fig.add_gridspec(2, 2)

        ax_xt = fig.add_subplot(gs[0,0])      # x(t)
        ax_yt = fig.add_subplot(gs[1,0])      # y(t)
        ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows
        # ax_speed = fig.add_subplot(gs[2, 0])

        plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw")
        plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 1*offset, "#bdc9e1","|", "1.outlier")
        plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
        plot_xy([ax_xt, ax_yt], interpolated_coords, 3*offset,"#0570b0", "|", "3.interpolate")

        plot_traj(raw_coords[:63], 0*offset, "raw", "#d1cbdc", ax_traj)
        # plot_traj(outlier_filtered_coords, 1*offset, "outlier", ax_traj)
        # plot_traj(likelihood_filtered_coords, 2*offset, "likelihood", ax_traj)
        plot_traj(interpolated_coords[:63], 0*offset, "interpolate", "#0570b0" ,ax_traj, "|")

        # ax_speed.plot(raw_coords["t"], speed, label="speed", marker="|")
        # ax_speed.plot(raw_coords["t"][peaks], speed[peaks], "x")

        ax_xt.set_ylabel("x")
        ax_xt.legend()
        ax_xt.set_xlim(0, 0.5)
        ax_xt.set_title("X and Y comparison along preprocessing")

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

        fig.savefig(make_output_path(cfg.paths.figures / RAT_NAME / "preprocessing", f"{trial_name}_interpolation.png"))
        plt.tight_layout()
        plt.show()
        plt.close()

    

if DISTRI : 


    # ------------------------------------ distribution ---------------------------------------

    likelihood_distri = []

    print(f"Number of files : {len(filenames)}")

    for i, coords_path in enumerate(filenames[6:]):

        coords_path = Path(coords_path)

        # load DLC results
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

    print(data.head())

    fig, ax = plt.subplots()

    sns.violinplot(
        data=data,
        x="bodypart",
        y="likelihood",
        palette="pastel",
        inner="quart",
        ax=ax
    )

    ax.set_title(f"Distribution of likelihood across bodyparts of rat {RAT_NAME}")
    ax.set_xlabel("Bodyparts")
    ax.set_ylabel("Likelihood")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
