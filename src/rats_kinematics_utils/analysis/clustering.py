#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import seaborn as sns

from rats_kinematics_utils.analysis.plot_comparative import plot_stacked_trajectories
from rats_kinematics_utils.core.file_utils import load_trial_data
from tslearn.metrics import frechet_path, dtw_path, ctw_path, lcss_path


# ==================================== display hyperparameter ===========================================


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("paper", style="ticks", rc=custom_params, palette="pastel")

LASER_COLOR = "coral"
LINE_COLOR = "gray"
AVG_LINE_COLOR = "navy"
LINE_TRANSPARANCY = 0.3

REWARD_PALETTE = {"no": "black",
                  "yes": "green"}
LASER_INTENSITY_PALETTE = {"low" : "lightblue",
                            "high" : "salmon",
                            "NOstim" : "gray"}
LASER_INTENSITY_PALETTE_DARK = {"low" : "steelblue",
                            "high" : "tomato",
                            "NOstim" : "black"}
HUE_ORDER = ["low", "high"]


# ==================================== Plots for comparative analysis ===========================================

def plot_trajectories(cfg, trajectories, labels) : 
    fig, ax = plt.subplots(figsize=(9, 7))

    rows= []
    for i, (traj, label) in enumerate(zip(trajectories, labels) ): 
        H_px = 512  # pixels

        rows.append(pd.DataFrame({
            "x" : (H_px - traj[:, 0]) * cfg.cm_per_pixel,
            "y" : (H_px - traj[:, 1]) * cfg.cm_per_pixel,
            "condition" : label.split("_")[0],
            "traj_id" :i, 
        }))

    data = pd.concat(rows, ignore_index=True)

    sns.lineplot(
        data=data,
        x="x",
        y="y",
        hue="condition",
        units="traj_id",
        estimator=None,  
        sort=False,
        alpha=0.3,
        lw=1,
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

    # ax.set_xlim(0, cfg.frame_width_px * cfg.cm_per_pixel)
    # ax.set_ylim(0, cfg.frame_width_px * cfg.cm_per_pixel)

    ax.set_xlim(3, cfg.frame_width_px * cfg.cm_per_pixel)
    ax.set_ylim(2, 7)

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

        metrics = load_trial_data(metrics_path)
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

    return ax




def display_distance_matrix(dist_mat, title: str, show) : 
    fig = plt.figure()
    plt.imshow(dist_mat)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Trajectory")
    plt.ylabel("Trajectory")

    if show: 
        plt.show()

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










def extract_trajectories(cfg, filenames: list[Path]) -> list[pd.DataFrame] : 
    """
    Get all the trajectories
    Each trajectories must be the same lenght in order to compute matrix after!
    """
    all_traj = []
    true_labels = []
    for i, metrics_path in enumerate(filenames) :
        metrics_path = Path(metrics_path) 
        for trial in load_trial_data(metrics_path) :

            if not trial[cfg.bodypart]["trial_success"] or \
                trial[cfg.bodypart].get("behavior_state", "keep_all") != "keep_all": 
                continue

            xy = trial[cfg.bodypart]["xy_pad_off"]

            if trial["camera_view"] == "right": 
                xy["x"] = cfg.frame_width_px - xy["x"]  

            xy = xy[["x", "y"]].to_numpy()

            if not np.isfinite(xy).all():
                print(trial[cfg.bodypart]["xy_state"])
                continue

            label = trial['condition'] + "_" + trial["laser_state"] + "_" + trial["laser_intensity"]

            all_traj.append(xy)
            true_labels.append(label)

    return all_traj, true_labels






def make_distance_matrix(trajectories):
    from tqdm import tqdm
    import numpy as np

    n_traj = len(trajectories)
    dist_mat = np.full((n_traj, n_traj), np.nan)  

    for i in tqdm(range(n_traj), desc="Computing matrix"):
        p = trajectories[i]

        if len(p) == 0 or not np.isfinite(p).all():
            print(f"Bad trajectory i: {i}")
            print(p)
            continue

        for j in range(n_traj):
            q = trajectories[j]

            if len(q) == 0 or not np.isfinite(q).all():
                print(f"Bad trajectory j: {j}")
                print(q)
                continue

            try:
                _, dist = dtw_path(p, q)
            except Exception as e:
                print(f"DTW failed for ({i},{j}): {e}")
                continue

            if not np.isfinite(dist):
                print(f"NaN/inf distance at ({i},{j})")
            else:
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist

    print("Done")

    return dist_mat