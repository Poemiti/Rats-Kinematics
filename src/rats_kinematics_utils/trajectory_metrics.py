import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class Trajectory:
    def __init__(
        self,
        coords: pd.DataFrame,
        fps: int = 125,
        cm_per_pixel: float | None = None,
    ):
        """
        Parameters
        ----------
        coords : pd.DataFrame
            Must contain columns: ["x", "y", "t"]
        fps : int
            Frames per second
        cm_per_pixel : float | None
            Spatial scale (cm / pixel). If None → stays in pixels.
        """
        self.coords = coords.reset_index(drop=True)
        self.fps = fps
        self.dt = 1 / fps
        self.cm_per_pixel = cm_per_pixel

    # ------------------- internal helpers -------------------

    def _scale(self, values):
        if self.cm_per_pixel is None:
            return values
        return values * self.cm_per_pixel

    def _displacement_steps(self, coords: pd.DataFrame):
        diffs = coords[["x", "y"]].diff()
        step_dist = np.sqrt((diffs**2).sum(axis=1))
        return self._scale(step_dist)

    # ------------------- geometric metrics -------------------

    def path_length(self, coords: pd.DataFrame | None = None) -> float:
        """Total traveled distance (path length)."""
        if coords is None:
            coords = self.coords
        return self._displacement_steps(coords).sum()

    def net_displacement(self, coords: pd.DataFrame | None = None) -> float:
        """Straight-line distance from start to end."""
        if coords is None:
            coords = self.coords

        start = coords[["x", "y"]].iloc[0]
        end = coords[["x", "y"]].iloc[-1]

        disp = self._scale(end - start)
        return np.linalg.norm(disp)

    def tortuosity(self, coords: pd.DataFrame | None = None) -> float:
        """Path length / straight-line distance."""
        if coords is None:
            coords = self.coords

        direct = self.net_displacement(coords)
        if direct == 0:
            return np.nan
        return self.path_length(coords) / direct

    # ------------------- time -------------------

    def duration(self, coords: pd.DataFrame | None = None) -> float:
        if coords is None:
            coords = self.coords
        return coords["t"].iloc[-1] - coords["t"].iloc[0]

    # ------------------- velocity -------------------

    def velocity_vector(self, coords: pd.DataFrame | None = None) -> pd.DataFrame:
        """Instantaneous velocity components."""
        if coords is None:
            coords = self.coords

        v = coords[["x", "y"]].diff() / self.dt
        v = self._scale(v)

        return pd.DataFrame({
            "t": coords["t"],
            "vx": v["x"],
            "vy": v["y"]
        })

    def instant_velocity(self, coords: pd.DataFrame | None = None) -> pd.Series:
        """Instantaneous velocity"""
        if coords is None:
            coords = self.coords

        v = self.velocity_vector(coords)
        velo = np.sqrt(v["vx"]**2 + v["vy"]**2)
        return pd.DataFrame({'t': coords["t"],
                             "velocity": velo})

    def mean_speed(self, coords: pd.DataFrame | None = None) -> float:
        """
        Average speed = path length / duration
        """
        if coords is None:
            coords = self.coords

        return self.path_length(coords) / self.duration(coords)

    def net_average_velocity(self, coords: pd.DataFrame | None = None) -> dict:
        """
        Net average velocity vector (displacement / time)
        """
        if coords is None:
            coords = self.coords

        start = coords[["x", "y"]].iloc[0]
        end = coords[["x", "y"]].iloc[-1]

        dt = self.duration(coords)

        displacement = self._scale(end - start)

        vx_avg = displacement["x"] / dt
        vy_avg = displacement["y"] / dt

        return {
            "vx_avg": vx_avg,
            "vy_avg": vy_avg,
            "v_avg": np.sqrt(vx_avg**2 + vy_avg**2)
        }

    def peak_speed(self, coords: pd.DataFrame | None = None) -> float:
        """ Peak instantaneous speed """
        if coords is None:
            coords = self.coords

        s = self.instant_velocity(coords)
        peaks, _ = find_peaks(s["velocity"].dropna())
        if len(peaks) == 0:
            return np.nan
        return s.iloc[peaks].max()

    # ------------------- acceleration -------------------

    def acceleration(self, coords: pd.DataFrame | None = None) -> pd.Series:
        """ Instantaneous acceleration magnitude """
        if coords is None:
            coords = self.coords

        v = self.velocity_vector(coords)[["vx", "vy"]]
        a = v.diff() / self.dt
        acc= np.sqrt(a["vx"]**2 + a["vy"]**2)
        return pd.DataFrame({'t': coords["t"],
                             "acceleration": acc})



def crop_xy(xy: pd.DataFrame, start: float, end: float) :  
    """
    Crop coordinates from [start : end]
    
    :param xy: coordinates
    :type xy: pd.DataFrame
    :param start: starting time (in sec)
    :type start: float
    :param end: ending time (in sec)
    :type end: float
    """
    return xy.loc[
        (xy["t"] >= start) &
        (xy["t"] <= end)
    ].reset_index(drop=True)
