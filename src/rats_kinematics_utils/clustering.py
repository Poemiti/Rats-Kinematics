#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import seaborn as sns

from rats_kinematics_utils.plot_comparative import plot_stacked_trajectories
from rats_kinematics_utils.pipeline_maker import load_metrics
from tslearn.metrics import frechet_path, dtw_path, ctw_path, lcss_path

sns.set_theme(style="darkgrid")

def plot_trajectories(cfg, trajectories, labels) : 
    fig, ax = plt.subplots(figsize=(9, 7))

    rows= []
    for (traj, label) in zip(trajectories, labels) : 
        H_px = 512  # pixels

        rows.append(pd.DataFrame({
            "x" : (H_px - traj[:, 0]) * cfg.cm_per_pixel,
            "y" : (H_px - traj[:, 1]) * cfg.cm_per_pixel,
            "condition" : label.split("_")[0]
        }))

    sns.lineplot(
        data=pd.concat(rows, ignore_index=True),
        x="x",
        y="y",
        hue="condition",
        estimator=None,  
        sort=False,
        alpha=0.7,
        ax=ax
    )

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)


    ax.tick_params(direction="out")

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"Stacked Trajectories, (n={len(trajectories)})")

    ax.invert_xaxis()
    return ax



def plot_all_trajectories(cfg, filenames: list[Path]) -> None: 
    n_trial = 0
    total_trial = 0

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, metrics_path in enumerate(filenames) :
        metrics_path = Path(metrics_path) 

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
    ax.set_title(f"Stacked Trajectories\nnumber of trial = {n_trial} / {total_trial}")

    ax.invert_xaxis()

    fig = ax.figure
    plt.show()
    plt.close(fig)




def display_distance_matrix(dist_mat, title: str) : 
    fig = plt.figure()
    plt.imshow(dist_mat)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Trajectory")
    plt.ylabel("Trajectory")
    return fig    


def plot_clustered_trajectories(cfg, trajectories, true_labels, pred_labels, show_noise=False, col_wrap=3):
    """
    trajectories : list of arrays (T,2)
    labels       : cluster labels from HDBSCAN
    show_noise   : whether to include noise (-1)
    col_wrap     : number of columns before wrapping
    """
    # Build long dataframe for seaborn
    rows = []
    for traj_id, (traj, t_label, p_label) in enumerate(zip(trajectories, true_labels, pred_labels)):
        if p_label == -1 and not show_noise:
            continue

        for t, (x, y) in enumerate(traj):

            # compute relative position
            H_px = 512  # pixels
            x = x * cfg.cm_per_pixel
            y = (H_px - y) * cfg.cm_per_pixel

            rows.append({
                "traj_id": traj_id,
                "x": x,
                "y": y,
                "cluster": str(p_label),
                "condition" : t_label.split("_")[0],
                "laser_state" : t_label.split("_")[1],
                "laser_intensity" : t_label.split("_")[2]
            })

    df = pd.DataFrame(rows)

    # Count cluster sizes
    cluster_sizes = df.groupby("cluster")["traj_id"].nunique()
    cluster_order = sorted(df["cluster"].unique()) 

    g = sns.FacetGrid(
        df,
        col="cluster",
        col_order=cluster_order,
        col_wrap=col_wrap,
        height=3
    )

    g.map_dataframe(
        sns.lineplot,
        x="x",
        y="y",
        hue="condition",
        style="laser_state",
        size="laser_intensity",
        estimator=None,
        alpha=0.7,
        sort=False,
        linewidth=1.5
    )

    g.add_legend(title="True Clusters")
    g._legend.set_loc("center right")


    # Set equal aspect and nicer titles
    for ax in g.axes.flat:
        cluster_label = ax.get_title().split("=")[-1].strip()
        n = cluster_sizes[cluster_label]
        ax.set_title(f"Cluster {cluster_label} (n={n})",)

    g.set_axis_labels("x (cm)", "y (cm)")

    return g
    




def plot_true_clustered_traj(cfg, trajectories, true_labels, pred_labels) : 
    """
    trajectories : list of arrays (T,2)
    labels       : cluster labels from HDBSCAN
    show_noise   : whether to include noise (-1)
    col_wrap     : number of columns before wrapping
    """
    # Build long dataframe for seaborn
    rows = []
    for traj_id, (traj, t_label, p_label) in enumerate(zip(trajectories, true_labels, pred_labels)):
        for t, (x, y) in enumerate(traj):

            # compute relative position
            H_px = 512  # pixels
            x = x * cfg.cm_per_pixel
            y = (H_px - y) * cfg.cm_per_pixel

            rows.append({
                "traj_id": traj_id,
                "x": x,
                "y": y,
                "cluster": str(p_label),
                "true_label" : t_label,
            })

    df = pd.DataFrame(rows)

    # Count cluster sizes
    cluster_sizes = df.groupby("true_label")["traj_id"].nunique()
    cluster_order = sorted(df["cluster"].unique()) 


    g = sns.FacetGrid(
        df,
        col="true_label",
        col_wrap=4,
        height=3
    )

    g.map_dataframe(
        sns.lineplot,
        x="x",
        y="y",
        hue="cluster",
        style="cluster",
        hue_order=cluster_order,
        estimator=None,
        alpha=0.7,
        sort=False,
        linewidth=1.5
    )

    g.add_legend(title="Founded Clusters")
    g._legend.set_loc("center right")

    # Set equal aspect and nicer titles
    for ax in g.axes.flat:
        cluster_label = ax.get_title().split("=")[-1].strip()
        n = cluster_sizes[cluster_label]
        ax.set_title(f"{cluster_label} (n={n})")


    g.set_axis_labels("x (cm)", "y (cm)")

    return g














###########################










def extract_trajectories(cfg, filenames: list[Path], coords: str) -> list[pd.DataFrame] : 
    all_traj = []
    true_labels = []
    for i, metrics_path in enumerate(filenames) :
        metrics_path = Path(metrics_path) 
        for trial in load_metrics(metrics_path) :

            # if coords == "xy_raw" : 
            #     xy = trial[coords]
            #     xy = xy[["x", "y"]].to_numpy()
            #     all_traj.append(xy)
            #     continue

            if not trial[cfg.bodypart]["trial_success"] : 
                continue

            # if not trial["reward"] : 
            #     continue

            xy = trial[cfg.bodypart][coords]
            xy = xy[["x", "y"]].to_numpy()

            label = trial['condition'] + "_" + trial["laser_intensity"]

            all_traj.append(xy)
            true_labels.append(label)

    return all_traj, true_labels



def make_distance_matrix(trajectories):
    from tqdm import tqdm

    n_traj = len(trajectories)
    dist_mat = np.zeros((n_traj, n_traj), dtype=np.float64)

    # total number of pair computations
    total_pairs = n_traj * (n_traj - 1) // 2

    with tqdm(total=total_pairs, desc="Computing distances matrix") as pbar:

        for i in range(n_traj - 1):
            p = trajectories[i]
            for j in range(i + 1, n_traj):
                q = trajectories[j]

                # path_frechet, frechet_dist = frechet_path(p, q)
                path_frechet, frechet_dist = dtw_path(p, q)

                dist_mat[i, j] = frechet_dist
                dist_mat[j, i] = frechet_dist

                pbar.update(1)

    return dist_mat