from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from PIL import Image
import numpy as np

from utils.trajectory_metrics import define_StartEnd_of_trajectory

# --------------------------------- csv utils ----------------------------------


def open_clean_csv(csv_path : Path) -> pd.DataFrame : 
    """
    Load and clean a DeepLabCut CSV file.

    DeepLabCut CSV files contain a three-level header
    (scorer, bodypart, coordinate). This function removes the scorer
    level and drops the first data row to return a clean DataFrame.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with a two-level column index
        (bodypart, coordinate).
    """

    # DLC CSV has 3 header rows (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    clea_df = df.iloc[1:].reset_index(drop=True)

    return clea_df

# --------------------------------- plotting ----------------------------------


def plot_single_bodypart_trajectories(
    coords: pd.DataFrame,
    cm_per_pixel: float,
    frame_laser_on : float = None,
    ax: plt.axes = None,
    color: str = "red",
    transparancy: float = 0.7,
    marker: str = None,
    rat_background : bool=False, 
) -> plt.axes :
    """
    Plot body part trajectories from a DeepLabCut CSV file.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new figure is created.
    invert_y : bool, optional
        Whether to invert the y-axis (image coordinate convention).

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted trajectories.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if rat_background: 
        img = np.asarray(Image.open("/home/poemiti/Rats-Kinematics/data/rat_image.png"))
        ax.imshow(img, alpha=0.3)

    H_px = 512  # pixels

    x = (H_px - coords["x"]) * cm_per_pixel
    y = (H_px - coords["y"]) * cm_per_pixel
    # x = coords["x"]
    # y = coords["y"]

    ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        linestyle="-",
        alpha=transparancy,
    )

    if frame_laser_on:
        ax.plot(
            x[frame_laser_on: ],
            y[frame_laser_on: ],
            marker=marker,
            color="red",
            linestyle="-",
            alpha=0.7,
        )

    return ax









def plot_bodyparts_trajectories(
    coords: pd.DataFrame,
    ax: plt.axes = None,
    bodyparts: list[str] | None = None,
    invert_y: bool = True,
    transparancy: float = 0.7, 
    threshold: float = 0.5,
) -> plt.axes :
    """
    Plot body part trajectories from a DeepLabCut CSV file.

    Body part coordinates are filtered using a likelihood threshold
    and truncated to the active movement segment based on
    `define_StartEnd_of_trajectory`.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new figure is created.
    bodyparts : list of str, optional
        Body parts to plot. If None, all available body parts are used.
    invert_y : bool, optional
        Whether to invert the y-axis (image coordinate convention).
    threshold : float, optional
        Minimum likelihood required to include a coordinate.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted trajectories.
    """

    if ax is None:
        fig, ax = plt.subplots()

    all_bodyparts = coords.columns.get_level_values(0).unique()
    if bodyparts is None:
        bodyparts = list(all_bodyparts)

    for bp in bodyparts:
        if bp not in all_bodyparts:
            continue

        xy = coords[bp]
        mask = xy["likelihood"] >= threshold
        xy_filtered = xy[mask]

        start, end = define_StartEnd_of_trajectory(xy_filtered, lever_position=210)
        xy_filtered = xy_filtered.iloc[start : end]

        ax.plot(
            xy_filtered["x"],
            xy_filtered["y"],
            marker="o",
            linestyle="-",
            label=bp,
            alpha=transparancy,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(direction="out")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(0, 512) # video dimension
    ax.set_ylim(0, 512)

    if invert_y:
        ax.invert_yaxis()

    return ax






def plot_3D_traj(coords: pd.DataFrame, 
                time : pd.Series,
                laser_on : float | None,
                ax: plt.axes,
                color: str,
                transparancy: float,
                y_invert: bool=False) -> plt.axes :
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    # --- trajectory ---
    ax.plot3D(
        coords["x"],
        coords["y"],
        time,
        color=color,
        marker=".",
        linestyle="-",
        alpha=transparancy,
    )

    # --- laser ON region (3D replacement for axvspan) ---
    if laser_on is not None:
        laser_off = laser_on + 0.3

        x_min, x_max = 0, 512
        y_min, y_max = 0, 512

        verts = [[
            (x_min, y_min, laser_on),
            (x_max, y_min, laser_on),
            (x_max, y_min, laser_off),
            (x_min, y_min, laser_off),
        ]]

        laser_plane = Poly3DCollection(
            verts,
            facecolor="red",
            alpha=0.3
        )
        ax.add_collection3d(laser_plane)

        # manual legend entry
        ax.legend(handles=[Patch(color="red", label="laser on")])

    # --- labels ---
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_zlabel("time (sec)")

    # --- limits ---
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    if y_invert:
        ax.invert_yaxis()

    return ax






if __name__ == "__main__":
    
    # ---------------------------------------------- setup path -------------------------------------------------

    # inputs (should exist)
    GENERATED_DATA_DIR = Path("../exploration/data")
    DATABASE_PATH = GENERATED_DATA_DIR / "database/rat_517_H001.csv"  # if it does not exist, make one with make_database (in file_management.py)

    # get the path for a video (path in a premade database)
    database = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(database.iloc[0]["filename"])
    CSV_DIR = GENERATED_DATA_DIR / "csv_results" / VIDEO_EXEMPLE.stem
    INPUT_CSV_PATH = CSV_DIR/ f"pred_results_{VIDEO_EXEMPLE.stem}_clip_00.csv"

    # output
    OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "ANALYSIS_RESULTS" / VIDEO_EXEMPLE.stem
    OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------- plot single trajectory of bodypart -------------------------------------------------
    
    THRESHOLD = 0.5
    BODYPART = ["left_hand"]

    fig, ax = plt.subplots()

    for csv_path in CSV_DIR.iterdir() :

        fig, ax = plt.subplots() 

        output_fig_dir = OUTPUT_TRAJECTORY_PATH / "trajectory_per_clip"
        output_fig_dir.mkdir(parents=True, exist_ok=True)
        output_fig_path = output_fig_dir / csv_path.stem

        coords = open_clean_csv(csv_path)
        xy = coords[BODYPART]
        mask = xy["likelihood"] >= THRESHOLD
        xy_filtered = xy[mask]

        start, end = define_StartEnd_of_trajectory(xy_filtered, lever_position=210)
        xy_filtered = xy_filtered.iloc[start : end]

        print(len(xy_filtered))

        ax = plot_single_bodypart_trajectories(
                    coords=xy_filtered,
                    ax=None,
                    invert_y=True,
                    color="red",
                    transparancy=0.7
                )

        ax.set_title(f"Trajectory of {csv_path.stem}, threshold 0.5,\nL1, NoStim, Successful")
        # plt.show()
        fig.savefig(output_fig_path)   

        plt.close(fig) 
