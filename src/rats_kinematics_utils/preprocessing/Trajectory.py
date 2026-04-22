import numpy as np
import pandas as pd
from scipy.signal import find_peaks



class Trajectory:
    def __init__(
        self,
        coords: pd.DataFrame,
        fps: int = 125,
        cm_per_pixel: float | None = None,
        lever_position: float | None = None, 
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
        self.lever_position = lever_position if lever_position is not None else (55, 230)

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
    

    def relative_mean_speed(self, coords: pd.DataFrame | None = None) -> float: 
        """
        Relative Average speed = sum(relative distance between eeach points) / duration
        The distance between each point takes into account if the paw goes up or down
        """
        if coords is None:
            coords = self.coords

        if len(coords) < 2:
            return 0.0

        # Cmpute distance between each points
        dx = coords["x"].diff()
        dy = coords["y"].diff()
        dx = self._scale(dx)
        dy = self._scale(dy)

        step_dist = (dx**2 + dy**2) ** 0.5

        # Sign rule:
        # +1 if both increasing or unchanged
        # -1 if either x or y decreases
        sign = ((dx >= 0) & (dy >= 0)).astype(int).replace({0: -1})
        length = (step_dist * sign).iloc[1:].sum()

        return length / self.duration(coords)
    

    def relative_speed(self, coords: pd.DataFrame | None = None) -> pd.DataFrame: 
        """
        Relative speed :
        The distance between each point takes into account if the paw goes up or down
        """
        if coords is None:
            coords = self.coords

        if len(coords) < 2:
            return pd.DataFrame({"t": [], "velocity": []})

        dx = coords["x"].diff()
        dy = coords["y"].diff()
        dt = coords["t"].diff()

        dx = self._scale(dx)
        dy = self._scale(-dy)

        # signed displacement (direction preserved)
        displacement = dx + dy  
        velocity = displacement / dt

        return pd.DataFrame({
            "t": coords["t"],
            "velocity": velocity
        }).dropna()




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

        v = self.instant_velocity(coords)
        peaks, _ = find_peaks(v["velocity"].dropna())
        if len(peaks) == 0:
            return np.nan
        p = v.iloc[peaks].max()
        return p["velocity"]
    


    def pre_post_velocity(self, time_pad_off: float, 
                          coords: pd.DataFrame | None = None) -> tuple[float, float]:
        """
        Return a tuple with the velocity 0.025 sec before the laser and 
        0.025 sec after the beginning of the laser.
        It uses the time of the pad off, since the laser gets on 0.025 sec after.
        and in the case a 'LaserOff' trial, we don't have a laserOn time
        """
        if coords is None:
            coords = self.coords

        time_laser_on = time_pad_off + 0.025
        pre_coords = coords.loc[(coords["t"] >= time_pad_off - 0.075) & 
                                (coords["t"] <= time_laser_on)].reset_index(drop=True)
        post_coords = coords.loc[(coords["t"] >= time_laser_on) & 
                                 (coords["t"] <= time_laser_on + 0.1)].reset_index(drop=True)

        pre_velo = self.mean_speed(pre_coords)
        post_velo = self.mean_speed(post_coords)

        return pre_velo, post_velo

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


    def lever_bodypart_distance(self, coords: pd.DataFrame | None = None, 
                                lever_pos: tuple[int, int] = None ) -> pd.Series: 
        """Compute the straight/net distance between the lever 
        and the bodypart choosen"""

        if coords is None:
            coords = self.coords
        if lever_pos is None: 
            lever_pos = self.lever_position

        xy = coords[["x", "y"]]

        disp = self._scale(xy - lever_pos)
        distance = np.linalg.norm(disp, axis=1)

        return pd.DataFrame({
                "t": coords["t"],
                "distance": distance,
        })