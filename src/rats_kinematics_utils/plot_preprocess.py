#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _outlier_proportion(df):
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



def _plot_traj(coord, offset, label, color, ax: plt.axes = None, marker: str = None):

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



def _plot_xy(axes, coords, offset, color,  marker, label, time_pad_off= None) -> None :
    
    t = coords["t"]
    x = coords["x"] + offset
    y = coords["y"] - offset

    axes[0].plot(t, x, marker=marker, color=color, label=label)
    axes[1].plot(t, y, marker=marker, color=color)

    if time_pad_off : 
        axes[0].axvline(time_pad_off, color='k', lw=0.8, ls='--', label="time pad off")
        axes[1].axvline(time_pad_off, color='k', lw=0.8, ls='--')


def _plot_outliers(axes, coords, params) -> None : 

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



def make_interpolation_figures(interpolated_coords, 
                               likelihood_filtered_coords,
                               outlier_filtered_coords,
                               raw_coords,
                               time_pad_off,
                               title, 
                               save_as):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, 2)

    offset = 10 # pixel

    ax_xt = fig.add_subplot(gs[0,0])      # x(t)
    ax_yt = fig.add_subplot(gs[1,0])      # y(t)
    ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows
    # ax_speed = fig.add_subplot(gs[2, 0])

    _plot_xy([ax_xt, ax_yt], interpolated_coords, 3*offset,"#0570b0", "|", "3.interpolate")
    _plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
    _plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 1*offset, "#bdc9e1","|", "1.outlier")
    _plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw", time_pad_off)
    
    _plot_traj(raw_coords[:63], 0*offset, "raw", "#d1cbdc", ax_traj)
    _plot_traj(interpolated_coords[:63], 0*offset, "interpolate", "#0570b0" ,ax_traj, "|")

    # ax_speed.plot(raw_coords["t"], speed, label="speed", marker="|")
    # ax_speed.plot(raw_coords["t"][peaks], speed[peaks], "x")

    ax_xt.set_ylabel("x")
    ax_xt.legend()
    ax_xt.set_xlim(time_pad_off - 0.1, time_pad_off + 0.4)
    ax_xt.set_title(title)

    ax_yt.set_ylabel("y")
    ax_yt.set_xlim(time_pad_off - 0.1, time_pad_off + 0.4)
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

    ax_yt.set_xticks([time_pad_off - 0.1, time_pad_off + 0.4])

    plt.tight_layout()
    fig.savefig(save_as)
    plt.close()




def make_outlier_figures(raw_coords, params, save_as): 
    fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
    _plot_outliers(axes, raw_coords, params)
    plt.tight_layout()
    plt.gca().set_xlim(0,0.5)
    fig.savefig(save_as)
    plt.close()