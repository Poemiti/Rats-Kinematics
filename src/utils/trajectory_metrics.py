from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks


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


# V1
def define_StartEnd_of_trajectory(coords : pd.DataFrame, lever_position) -> float : 
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

# V2
# def define_StartEnd_of_trajectory(coords : pd.DataFrame, lever_position) -> float : 
#         t_start = 0
#         t_end = len(coords)

#         for frame, y in enumerate(coords["y"]) : 
#             if y < lever_position : 
#                 t_end = frame + 45
#                 break

#         return t_start, t_end


# V3
# def define_End_of_trajectory(coords : pd.DataFrame, lever_position) -> float : 
#         crossed = False
#         t_end = len(coords)

#         for t, row in coords.iterrows():
#             if t == 0 : 
#                 pass
#                 # print(f"not crossed, y = {row['y']}, t = {t}")

#             if row["y"] < lever_position and not crossed:
#                 # print(f"crossed, y = {row['y']}, t = {t}")
#                 crossed = True
#                 continue

#             if row["y"] > lever_position and crossed:
#                 # print(f"crossed again, y = {row['y']}, t = {t}")
#                 t_end = t-2
#                 break

#         return t_end

# V4
def define_End_of_trajectory(coords : pd.DataFrame, lever_position) -> float : 
    t_end = len(coords)

    peaks, properties = find_peaks(-coords["y"], height=-lever_position)
    # peaks_below = peaks[coords["y"].iloc[peaks] < lever_position]
    print(f"peeaks : {peaks}")

    if len(peaks) == 0 : 
        return 0
    
    fig, ax = plt.subplots()

    ax.plot(coords["x"], coords["y"], label="trajectory")
    ax.plot(coords["x"].iloc[peaks[0]],
            coords["y"].iloc[peaks[0]],
            "o", label="peaks below lever")


    ax.axhline(lever_position, color="k", lw=0.8, ls="--", label="lever position")

    ax.tick_params(direction="out")

    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    ax.set_xlim(0, 512)  # video dimension
    ax.set_ylim(0, 512)

    ax.invert_yaxis()
    ax.legend()

    plt.show()

    return t_end





# --------------------------------- metrics calculation ----------------------------------

class Trajectory:
    def __init__(
                self,
                coords: pd.DataFrame,
                reaching_coords: pd.DataFrame,
                fps: int = 125,
                m_per_pixel: float | None = None,
    ):
        """
        Parameters
        ----------
        coords : pd.DataFrame
            All x, y coordinates in pixels
        reaching_coords : pd.DataFrame
            x, y coordinates of the reaching movement in pixels
        fps : int
            Frames per second
        cm_per_pixel : float, optional
            Spatial scale (cm / pixel). If None, values stay in pixels.
        """
        self.coords_px = coords  # pixel
        self.reaching_coords = reaching_coords  # pixel, actual trajectory where we compute metrics
        self.fps = fps
        self.dt = 1 / fps
        self.m_per_pixel = m_per_pixel # m

    def _scale(self, values: pd.Series | pd.DataFrame):
        if self.m_per_pixel is None:
            return values
        return values * self.m_per_pixel
    
    def _displacements(self) -> pd.DataFrame:
        diffs = self.reaching_coords[["x", "y"]].diff()
        disp_px = (diffs.pow(2).sum(axis=1) ** 0.5)
        disp = self._scale(disp_px)

        return pd.DataFrame({
                    "t": self.reaching_coords["t"],
                    "displacement": disp
                })

    # ------------------- calculation ------------------

    def velocity_vector(self) -> pd.DataFrame:
        v_px = self.reaching_coords[["x", "y"]].diff() / self.dt
        v = self._scale(v_px)

        return pd.DataFrame({
            "t": self.reaching_coords["t"],
            "vx": v["x"],
            "vy": v["y"]
        })

    def instant_velocity(self) -> pd.DataFrame:
        v = self.velocity_vector()
        speed = (v[["vx", "vy"]].pow(2).sum(axis=1) ** 0.5)

        return pd.DataFrame({
            "t": v["t"],
            "velocity": speed
        })

    def acceleration(self) -> pd.Series:
        v = self.velocity_vector()[["vx", "vy"]]
        a = v.diff() / self.dt
        acc = (a.pow(2).sum(axis=1) ** 0.5)

        return pd.DataFrame({
            "t": self.reaching_coords["t"],
            "acceleration": acc
        })


    def distance(self) -> float:
        return self._displacements()["displacement"].sum()

    def mean_velocity(self) -> float:
        duration = self.reaching_coords["t"].iloc[-1] - self.reaching_coords["t"].iloc[0]
        return self.distance() / duration

    

# ------------------------ metrics plotting -----------------------------------


def plot_metric_time(metric: pd.Series, 
                     time : pd.Series,
                     laser_on : float | None,
                     ax: plt.axes,
                     color: str,
                     transparancy: float=0.7,
                     y_invert: bool=False) -> plt.axes : 
    """
    Plot the velocity or acceleration in time, for 1 condition (either NoStim, Beta or conti), for 1 clip
    The plot is highlighted between time 0.25 sec and 0.325 which correspond to
    the time range of a laser stimulus.
    No Highlight for NoStim condition

    Parameters
    ----------
    metric : pd.Series
        instantaneous velocity or acceleration for 1 clip  

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted velocities.
    """
    fps = 125

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, metric, color= color, alpha=transparancy, marker='.')
    if laser_on : 
        laser_off = laser_on +  0.3 # sec or 37.5 frame
        ax.axvspan(laser_on, laser_off, color='red', alpha=0.3, label="laser on")

    # add line for pad off 
    # ax.axvline(time.iloc[0], color='k', lw=0.8, ls='--', label="pad off")
    # ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(direction="out")

    if y_invert: 
        ax.invert_yaxis()

    return ax



def create_trajectory_object(coords_path, bodypart, threshold, m_per_pixel) -> Trajectory : 
    # get data from the bodypart
    data = open_clean_csv(coords_path)
    data = data[bodypart]
    data = data[data["likelihood"] >= threshold].reset_index(drop=True)
    
    # filter to get only the trajectory we want
    t_start, t_end = define_StartEnd_of_trajectory(data, 220)
    # print(f"t_end = {t_end}, len_data = {len(data)}")
    xy = data.iloc[t_start : t_end].reset_index(drop=True)

    if len(xy) <= 1 : 
        print("ERROR : no movement have been found in this clip")
        return 0

    data = data[["x", "y"]]
    xy = xy[["x", "y"]][:-1]
    return Trajectory(coords=data,
                      reaching_coords=xy, 
                      fps=125, 
                      m_per_pixel=m_per_pixel)









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

    # ---------------------------------------------- parameters -------------------------------------------------


    BODYPART = "left_hand"
    THRESHOLD = 0.5
    SHOW = True

    # define cm per pixel
    frame_width_m = 0.12 # m
    frame_width_px = 512 # pixel
    M_PER_PIXEL = frame_width_px / frame_width_m

    # ---------------------------------------------- compute metrics -------------------------------------------------

    print(f"\nComputing metric of {INPUT_CSV_PATH} :")
    metrics : list[dict] = []

    
    Traj : Trajectory = create_trajectory_object(INPUT_CSV_PATH, BODYPART, THRESHOLD, M_PER_PIXEL)

    distance = Traj.distance()
    mean_velo = Traj.mean_velocity()
    instant_velo = Traj.instant_velocity()
    acceleration = Traj.acceleration()
    y_position = Traj._scale(Traj.coords_px[["y"]])

    print(f"\ndistance = {distance:.02f} m")
    print(f"mean velocity = {mean_velo:.02f} m.s")
    print(f"instaneous velocity = \n{instant_velo}")
    print(f"acceleration = \n{acceleration}")

    metrics.append({"file" : INPUT_CSV_PATH,
                    "distance" : distance,
                    "velocity" : mean_velo
                    })
    

    # save computed metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_TRAJECTORY_PATH / "metrics_per_clips.csv", index=False)
    instant_velo.to_csv(OUTPUT_TRAJECTORY_PATH / "instant_velocities.csv", index=False)
    acceleration.to_csv(OUTPUT_TRAJECTORY_PATH / "acceleration.csv", index=False)

    # ---------------------------------------- plot metrics ---------------------------------------

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    plot_metric_time(instant_velo,
                     ax = axs[0], 
                     title="Velocity over time of a trial",
                     ylabel="Velocity (m.s$^{-1}$)",)
    
    plot_metric_time(acceleration, 
                     ax = axs[1],
                     title="Acceleration over time of a trial",
                     ylabel="Acceleration (m.s$^{-2}$)",)
    

    plot_metric_time(y_position, 
                     ax = axs[2],
                     title="Height position over time of a trial",
                     ylabel="position (m)", 
                     y_invert=True)
    
    if SHOW :
        plt.show()
    plt.close(fig)

