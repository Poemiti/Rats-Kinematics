import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class Trajectory:
    def __init__(
                self,
                coords: pd.DataFrame,
                reaching_coords: pd.DataFrame,
                laserOn_coords : pd.DataFrame,
                fps: int = 125,
                cm_per_pixel: float | None = None,
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
        ccm_per_pixel : float, optional
            Spatial scale (cm / pixel). If None, values stay in pixels.
        """
        self.coords_px = coords  # pixel
        self.reaching_coords = reaching_coords  # pixel, traj [ pad of -> laser off ]
        self.laserOn_coords = laserOn_coords    # pixel, traj [ laser on -> laser off ]
        self.fps = fps
        self.dt = 1 / fps
        self.cm_per_pixel = cm_per_pixel # cm

    def _scale(self, values: pd.Series | pd.DataFrame):
        if self.cm_per_pixel is None:
            return values
        return values * self.cm_per_pixel
    
    def _displacements(self) -> pd.DataFrame:
        diffs = self.laserOn_coords[["x", "y"]].diff()
        disp_px = (diffs.pow(2).sum(axis=1).pow(0.5))
        disp = self._scale(disp_px)

        return pd.DataFrame({
                    "t": self.laserOn_coords["t"],
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
        duration = self.laserOn_coords["t"].iloc[-1] - self.laserOn_coords["t"].iloc[0]
        return self.distance() / duration
    
    def peak(self) -> float : 
        v: pd.Series = self.instant_velocity()["velocity"]
        peaks, _ = find_peaks(v)
        return v.iloc[peaks].max()
    
    def tortuosity(self) -> float : 
        actual_path_length = self.distance()
        start = self.laserOn_coords[["x", "y"]].iloc[0]
        end = self.laserOn_coords[["x", "y"]].iloc[-1]
        direct_path_length = np.linalg.norm(end - start)  # calculate the norm √((x-x)² + (y-y²)
        return actual_path_length / (direct_path_length * self.cm_per_pixel)


    