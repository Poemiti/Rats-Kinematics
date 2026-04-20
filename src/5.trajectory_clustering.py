#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import seaborn as sns
import sys

from rats_kinematics_utils.core.config import load_config
from rats_kinematics_utils.core.file_utils import print_analysis_info, make_output_path
import rats_kinematics_utils.analysis.clustering as c

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import LabelEncoder


# ---------------------------- setup ----------------------------------

SHOW = False
NORMALIZE = False
cfg = load_config()
print_analysis_info(cfg, "Trajectory Clustering")


matrix_dir = cfg.paths.metrics / "distance_matrix"
figures_dir = cfg.paths.analysis / "distance_matrix" 
matrix_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
matrix_metrics_path = matrix_dir / "dtw_only_successful_raw.joblib"

filenames = list((cfg.paths.metrics).glob("*.joblib"))

print(f"\nClustering of rat {cfg.rat_name}")
print(f"Metrics used : ")
for file in filenames:
    print(f"   - {file.stem}")

# ------------------------------- get the trajectories -------------------------------------


all_traj, true_labels = c.extract_trajectories(cfg, filenames, coords="xy_raw")
print("Number of trajectories:", len(all_traj))

# ------------------------------- plot all the trajectories -------------------------------------

# ax = c.plot_all_trajectories(cfg, filenames)
ax = c.plot_trajectories(cfg, all_traj, true_labels)
fig = ax.figure
fig.savefig(figures_dir / "stacked_traj_laserOn")
if SHOW : 
    plt.show()
plt.close()


# # ------------------------------- compute or load distance matrix -------------------------------------

if matrix_metrics_path.exists(): 
    print("\nDistance matrix has already been computed")
    res = input("Do you want to overwrite ? (y/n) : ")

if res == "y" or res == "Y" : 
    start = time.perf_counter()
    dist_matrix = c.make_distance_matrix(all_traj)
    stop = time.perf_counter()

    print(f"Processing time: {(stop - start) // 60} min")
    joblib.dump(dist_matrix, matrix_metrics_path)

if res == "n" or res == "n" : 
    print("Loading distance matrix...")
    dist_matrix = joblib.load(matrix_metrics_path)


# ------------------------------- display distance matrix -------------------------------------

fig = c.display_distance_matrix(dist_matrix, f"Distance Matrix of rat : {cfg.rat_name}", show=SHOW)
fig.savefig(figures_dir / "distance_matrix.png")


# ------------------------------- verify distance matrix -------------------------------------

D = dist_matrix  # shape (N, N)

row_means = np.nanmean(D, axis=1)
row_max   = np.nanmax(D, axis=1)
row_min   = np.nanmin(D, axis=1)

# suspicious indices
bad_mean = np.argmax(row_means)
bad_max  = np.argmax(row_max)
bad_min  = np.argmin(row_min)

print("Largest mean distance:", bad_mean)
print("Largest max distance:", bad_max)
print("Smallest min distance:", bad_min)

for i, traj in enumerate(all_traj):
    if not np.isfinite(traj).all():
        print("Bad traj:", i)
    elif len(traj) == 0:
        print(f"{i}: empty")
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
fig = c.display_distance_matrix(sorted_mat, f"Sorted distance Matrix of rat : {cfg.rat_name}", show=SHOW)
fig.savefig(figures_dir / "sorted_distance_matrix.png")



# ------------------------------- display cluster -------------------------------------

n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
print("Clusters (excluding noise):", n_clusters)

if n_clusters < 20 :
    fig = c.plot_clustered_trajectories(cfg, all_traj, 
                                      true_labels, 
                                      pred_labels, 
                                      show_noise=True,
                                      col_wrap=4 if n_clusters >= 4 else 3)
    fig.savefig(figures_dir / "dtw_clustsize6_by_predlabel.png")
    if SHOW : 
        plt.show()
    plt.close()


    fig = c.plot_true_clustered_traj(cfg, all_traj, true_labels, pred_labels)
    fig.savefig(figures_dir / "dtw_clustsize6_by_truelabel.png")
    if SHOW : 
        plt.show()
    plt.close()


print("Done !")
