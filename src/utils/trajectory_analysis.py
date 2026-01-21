from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_bodyparts_trajectories(
    csv_path: Path,
    ax: plt.axes = None,
    bodyparts: list[str] | None = None,
    invert_y: bool = True,
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

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    df = open_clean_csv(csv_path)

    all_bodyparts = df.columns.get_level_values(0).unique()
    if bodyparts is None:
        bodyparts = list(all_bodyparts)

    for bp in bodyparts:
        if bp not in all_bodyparts:
            continue

        xy = df[bp]
        mask = xy["likelihood"] >= threshold
        xy_filtered = xy[mask]
        start, end = define_StartEnd_of_trajectory(xy_filtered)
        xy_filtered = xy_filtered.iloc[start : end]

        ax.plot(
            xy_filtered["x"],
            xy_filtered["y"],
            marker="o",
            linestyle="-",
            label=bp,
            alpha=0.7,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(0, 512) # video dimension
    ax.set_ylim(0, 512)

    if invert_y:
        ax.invert_yaxis()

    return ax


def plot_stacked_trajectories(csv_dir: Path,
                            output_fig_path: Path | None = None,
                            bodyparts: list[str] | None = None,
                            invert_y: bool = True,
                            threshold: float = 0.5,
                            show: bool = False,
                        ) -> None:
    """
    Plot trajectories from multiple trials on a single figure.

    Each CSV file in the directory is treated as one trial, and all
    trajectories are overlaid on the same axes.

    Parameters
    ----------
    csv_dir : pathlib.Path
        Directory containing DeepLabCut CSV files.
    output_fig_path : pathlib.Path, optional
        Path where the output figure will be saved.
    bodyparts : list of str, optional
        Body parts to plot. If None, all body parts are used.
    invert_y : bool, optional
        Whether to invert the y-axis.
    threshold : float, optional
        Minimum likelihood required to include a coordinate.
    show : bool, optional
        Whether to display the figure interactively.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots()

    for csv_path in csv_dir.glob("*.csv"):
        plot_bodyparts_trajectories(
            csv_path=csv_path,
            ax=ax,
            bodyparts=bodyparts,
            invert_y=invert_y,
            threshold=threshold,
        )

    ax.set_title(f"Trajectories across frames of all trials\n{csv_dir.stem}")

    if output_fig_path:
        fig.savefig(output_fig_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_average_trajectories(csv_dir: Path,
                              output_fig_path: Path = None,
                              bodyparts : list[str] = None, 
                              invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    """
    Compute and plot average trajectories across multiple trials.

    This function is intended to aggregate body part trajectories
    across trials and display their average path.

    Parameters
    ----------
    csv_dir : pathlib.Path
        Directory containing DeepLabCut CSV files.
    output_fig_path : pathlib.Path, optional
        Path where the output figure will be saved.
    bodyparts : list of str, optional
        Body parts to include in the averaging.
    invert_y : bool, optional
        Whether to invert the y-axis.
    show : bool, optional
        Whether to display the figure interactively.
    threshold : float, optional
        Minimum likelihood required to include a coordinate.

    Returns
    -------
    None
    """

    # --------------------------- 1. compute the average xy coord -----------------------------    
    all_coords = []

    for csv_path in csv_dir.glob("*.csv") : 
        df = open_clean_csv(csv_path)

        all_bodyparts = df.columns.get_level_values(0).unique()
        if bodyparts is None:
            bodyparts = list(all_bodyparts)

        for bp in bodyparts : 
            if bp not in all_bodyparts:
                continue

            xy = df[bp]
            mask = xy["likelihood"] >= threshold
            xy_filtered = xy[mask]

        print(xy_filtered)

    # --------------------------- 2. plot the average trajectory -----------------------------    

    # TODO
    pass


# --------------------------------- metrics measurements ----------------------------------



def define_StartEnd_of_trajectory(coords : pd.DataFrame) : 
    """
    Determine the start and end indices of a movement trajectory.

    A trajectory is defined as the movement from the pad to the lever.
    The start is fixed at the beginning of the sequence, and the end
    is detected when the y-coordinate crosses a predefined lever position twice.

    Lever position is set at 210 pixels

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame containing ``x`` and ``y`` coordinates.

    Returns
    -------
    tuple of int
        Start and end indices of the trajectory.
    """
    lever_position = 210
    crossed = False
    t_start = 0
    print(t_start)
    t_end = len(coords)

    for t, row in coords.iterrows():
        if t == 0 : 
            print(f"not crossed, y = {row['y']}, t = {t}")

        if row["y"] < lever_position and not crossed:
            print(f"crossed, y = {row['y']}, t = {t}")
            crossed = True
            continue

        if row["y"] > lever_position and crossed:
            print(f"crossed again, y = {row['y']}, t = {t}")
            t_end = t-1
            break

    return t_start, t_end


def get_instantaneous_velocity(coords: pd.DataFrame) -> pd.Series:
    """
    Compute instantaneous velocity from coordinate data. 
    fps is set at 125

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame containing ``x`` and ``y`` coordinates.

    Returns
    -------
    pandas.Series
        Instantaneous velocity in pixels per second.
    """
    fps = 125
    diffs = coords.diff().dropna() # compute the difference between 2 row (the actual vs the previous)
    displacement = diffs.pow(2).sum(axis=1).pow(0.5)    # compute the power(2) of each columns + the sqrt (pow(0.5))
    print(displacement)

    return displacement * fps  # pixels / second


def get_velocity(coords: pd.DataFrame) -> float : 
    """
    Compute average velocity over a trajectory.

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame containing ``x`` and ``y`` coordinates.

    Returns
    -------
    float
        Average velocity in pixels per second.
    """

    fps = 125
    diffs = coords.diff().dropna()
    distance = (diffs.pow(2).sum(axis=1).pow(0.5)).sum()

    duration_sec = len(coords) / fps
    return distance / duration_sec  # pixel / sec


def get_distance(coords: pd.DataFrame) :
    """
    Compute total traveled distance along a trajectory.

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame containing ``x`` and ``y`` coordinates.

    Returns
    -------
    float
        Total distance in pixels.
    """

    distance = 0
    diffs = coords.diff().dropna()
    distance = (diffs.pow(2).sum(axis=1).pow(0.5)).sum()
    return distance


def compute_metric(csv_path: Path,
                    bodyparts : str,
                    metric,
                    threshold : float = 0.5) -> float : 
    """
    Compute a kinematic metric from DeepLabCut predictions for one clip.

    Coordinates are filtered by likelihood and restricted to the
    active movement segment before computing the metric.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.
    bodyparts : str
        Name of the body part to analyze.
    metric : callable
        Function applied to the trajectory (e.g., distance, velocity).
    threshold : float, optional
        Minimum likelihood required to include a coordinate.

    Returns
    -------
    float
        Computed metric value.
    """
    # get data from the bodypart
    df = open_clean_csv(csv_path)
    xy = df[bodyparts]
    xy = xy[xy["likelihood"] >= threshold]
    print(f"filtered coord : \n {xy}")

    # filter to get only the trajectory we want
    t_start, t_end = define_StartEnd_of_trajectory(xy)
    true_coords = xy.iloc[t_start : t_end].reset_index(drop=True)
    print(f"coords after threshold : {len(xy)}, coords after finding traj : {len(true_coords)}")

    if len(true_coords) <= 1 : 
        print("ERROR : no movement have been found in this clip")
        return 0

    return metric(true_coords[["x", "y"]])



if __name__ == "__main__":
    
    # ---------------------------------------------- setup path -------------------------------------------------

    # inputs (should exist)
    GENERATED_DATA_DIR = Path("../../exploration/data")
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

        plot_bodyparts_trajectories(
            csv_path=Path(csv_path),
            ax=ax,
            bodyparts=BODYPART,
            invert_y=True,
            threshold=THRESHOLD,
        )

        ax.set_title(f"Trajectory of {csv_path.stem}, threshold 0.5,\nL1, NoStim, Successful")
        # plt.show()
        fig.savefig(output_fig_path)   

        plt.close(fig) 


    # ---------------------------------------------- plot stacked + average trajectory of bodypart -------------------------------------------------


    plot_stacked_trajectories(csv_dir= INPUT_CSV_PATH , 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_stacked.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)


    plot_average_trajectories(csv_dir= INPUT_CSV_PATH, 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_average.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)

    # ---------------------------------------------- compute metrics -------------------------------------------------

    print(f"\nComputing metric of {INPUT_CSV_PATH} :")

    # distance
    distance = compute_metric(csv_path=INPUT_CSV_PATH,
                              bodyparts="left_hand",
                              metric=get_distance)
    
    # velocity
    velocity = compute_metric(csv_path=INPUT_CSV_PATH,
                              bodyparts="left_hand",
                              metric=get_velocity)

    # instantaneous velocity
    instant_velocity = compute_metric(csv_path=INPUT_CSV_PATH,
                              bodyparts="left_hand",
                              metric=get_instantaneous_velocity)
    

    print(f"\ndistance = {distance:.02f}")
    print(f"velocity = {velocity:.02f}")
    print(f"instaneous velocity = \n{instant_velocity}")

    # distance = 223.18
    # velocity = 680.43
    # instaneous velocity = 
    # 1        20.745121
    # 2        10.700683
    # XX       ....
    # 40       77.380144
