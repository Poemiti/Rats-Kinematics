from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from trajectory_ploting import open_clean_csv

def define_StartEnd_of_trajectory(coords : pd.DataFrame, lever_position : float = 210) : 
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
    crossed = False
    t_start = 0
    t_end = len(coords)

    for t, row in coords.iterrows():
        if t == 0 : 
            pass
            # print(f"not crossed, y = {row['y']}, t = {t}")

        if row["y"] < lever_position and not crossed:
            # print(f"crossed, y = {row['y']}, t = {t}")
            crossed = True
            continue

        if row["y"] > lever_position and crossed:
            # print(f"crossed again, y = {row['y']}, t = {t}")
            t_end = t-2
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
    # print(f"filtered coord : \n {xy}")

    # filter to get only the trajectory we want
    t_start, t_end = define_StartEnd_of_trajectory(xy)
    true_coords = xy.iloc[t_start : t_end].reset_index(drop=True)
    # print(f"coords after threshold : {len(xy)}, coords after finding traj : {len(true_coords)}")

    if len(true_coords) <= 1 : 
        print("ERROR : no movement have been found in this clip")
        return 0

    return metric(true_coords[["x", "y"]])



# ------------------------ metrics plotting -----------------------------------

def plot_violin_distribution(NoStim, conti, beta : pd.DataFrame) -> plt.axes : 
    """
    Does a Violin Plot of the distribution of velocity or acceleration for the 3 conditions, for 1 clip.
    The velocity given must be from 0.25 sec to 0.325 sec, which correspond to
    the time range of a laser stimulus (even when there is NoStim for comparaison purpose)

    Parameters
    ----------
    NoStim : pd.Dataframe
        instantaneous velocity for 1 clip  
    conti : pd.Dataframe
        instantaneous velocity for 1 clip  
    beta : pd.Dataframe
        instantaneous velocity for 1 clip  

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted velocities.
    """

    
    pass

def plot_metric_time(metric: pd.DataFrame) -> None : 
    """
    Plot the velocity or acceleration in time, for 1 condition (either NoStim, Beta or conti), for 1 clip
    The plot is highlighted between time 0.25 sec and 0.325 which correspond to
    the time range of a laser stimulus.
    No Highlight for NoStim condition

    Parameters
    ----------
    metric : pd.Dataframe
        instantaneous velocity or acceleration for 1 clip  

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted velocities.
    """

    pass





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
