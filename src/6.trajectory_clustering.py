#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import seaborn as sns

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, print_analysis_info
from rats_kinematics_utils.clustering import extract_trajectories, make_distance_matrix, plot_true_clustered_traj, plot_clustered_trajectories, display_distance_matrix, plot_trajectories

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import LabelEncoder


# ---------------------------- setup ----------------------------------

SHOW = False
NORMALIZE = False
cfg = load_config()
print_analysis_info(cfg, "Trajectory Clustering")


RAT_NAME = cfg.rat_name

matrix_dir = cfg.paths.metrics / "distance_matrix" / RAT_NAME
figures_dir = cfg.paths.figures / RAT_NAME / "distance_matrix" 
matrix_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
MATRIX = matrix_dir / "dtw_only_successful_raw.joblib"

filenames = sorted((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))

print(f"\nClustering of rat {RAT_NAME}")
print(f"Metrics used : ")
for file in filenames:
    print(f"   - {file.stem}")

# ------------------------------- get the trajectories -------------------------------------


all_traj, true_labels = extract_trajectories(filenames, coords="xy_raw")
print("Number of trajectories:", len(all_traj))


# ------------------------------- plot all the trajectories -------------------------------------


ax = plot_trajectories(cfg, all_traj, true_labels)
fig = ax.figure
fig.savefig(figures_dir / "stacked_traj_laserOn")
if SHOW : 
    plt.show()
plt.close()


# # ------------------------------- compute or load distance matrix -------------------------------------


if not MATRIX.exists() : 
    start = time.perf_counter()
    dist_matrix = make_distance_matrix(all_traj)

    stop = time.perf_counter()
    print(f"Processing time: {(stop - start) // 60} min")
    joblib.dump(dist_matrix, MATRIX)
else : 
    print(f"Loading {MATRIX}")
    dist_matrix = load_metrics(MATRIX)

# ------------------------------- display distance matrix -------------------------------------

fig = display_distance_matrix(dist_matrix, f"Distance Matrix of rat : {RAT_NAME}")
fig.savefig(figures_dir / "distance_matrix.png")
if SHOW : 
    fig.show()
plt.close(fig)


# ------------------------------- HDBSCAN -------------------------------------


if NORMALIZE : 
    dist_matrix = dist_matrix / dist_matrix.max()


hdbscan = HDBSCAN(
    metric='precomputed',  # for distance matrix
    min_cluster_size=8,
    min_samples=1, 
    cluster_selection_method='leaf',
    allow_single_cluster=True,
)

pred_labels = hdbscan.fit_predict(dist_matrix)

order = np.argsort(pred_labels)
sorted_mat = dist_matrix[order][:, order]
fig = display_distance_matrix(sorted_mat, f"Sorted distance Matrix of rat : {RAT_NAME}")
fig.savefig(figures_dir / "sorted_distance_matrix.png")
if SHOW : 
    fig.show()
plt.close(fig)


# ------------------------------- display cluster -------------------------------------

n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
print("Clusters (excluding noise):", n_clusters)

if n_clusters < 20 :
    fig = plot_clustered_trajectories(cfg, all_traj, 
                                      true_labels, 
                                      pred_labels, 
                                      show_noise=True,
                                      col_wrap=4 if n_clusters >= 4 else 3)
    fig.savefig(figures_dir / "dtw_clustsize6_by_predlabel.png")
    if SHOW : 
        plt.show()
    plt.close()


    fig = plot_true_clustered_traj(cfg, all_traj, true_labels, pred_labels)
    fig.savefig(figures_dir / "dtw_clustsize6_by_truelabel.png")
    if SHOW : 
        plt.show()
    plt.close()
