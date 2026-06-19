import numpy as np
import pandas as pd
from scipy.signal import find_peaks



class Trajectory:
    def __init__(
                self,
                coords: pd.DataFrame,
                fps: int = 125,
                frame_width: float = 512,
                cm_per_pixel: float | None = None,
                lever_position: float | None = (55, 230), 
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
        self.lever_position = lever_position
        self.frame_width = frame_width  # in pixel

    # ------------------- internal helpers -------------------

    def _scale(self, values):
        if self.cm_per_pixel is None:
            return values
        return values * self.cm_per_pixel

    def _displacement_steps(self, coords: pd.DataFrame):
        diffs = coords[["x", "y"]].diff()
        step_dist = np.sqrt((diffs**2).sum(axis=1))
        return self._scale(step_dist)
    
    def _cartesian_xy(self, view: str, coords: pd.DataFrame) : 
        """change to a correct cartesian axis
        If the camera view is left : 
            (0, 0) becomes the bottom left corner
        Elif the camera view is right :
            (0, 0) becomes the bottom right corner
        """
        if view == "left" :  # invert y axis
            y = self.frame_width - coords["y"]
            x = coords["x"]
        else :              # invert both x and y axis
            y = self.frame_width - coords["y"]
            x = self.frame_width - coords["x"]
        return x, y

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
    

    def _get_lever_reaching_time(self, coords: pd.DataFrame, lever_pos: tuple[int, int]) -> float: 
        """Return the time at which the coordinates where as close 
        a possible to the lever position"""

        lever_x, lever_y = lever_pos

        # measure distance between coords and lever
        distances = np.sqrt(
            (coords["x"] - lever_x) ** 2 +
            (coords["y"] - lever_y) ** 2
        )

        # return the time at which the distance was the smallest 
        return coords.loc[distances.idxmin(), "t"]
    

    def _show_traj(self, coords: pd.DataFrame):
        import matplotlib.pyplot as plt

        x = coords["x"]
        y = coords["y"]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(x, y, lw=1.5)
        
        ax.set_ylim(0, 512)
        ax.set_xlim(0, 512)

        ax.set_title("Trajectory")
        ax.set_xlabel("X position (px)")
        ax.set_ylabel("Y position (px)")
        ax.invert_yaxis()

        plt.show()
        plt.close()

    def tortuosity(self, coords: pd.DataFrame | None = None, 
                   time_pad_off: float = 0,
                   lever_pos: tuple[int, int] = None,
                   fixed_period: float = None) -> float:
        """Path length to get to the lever / straight-line distance from pad off to lever."""
        if coords is None:
            coords = self.coords
        if lever_pos is None: 
            lever_pos = self.lever_position

        if fixed_period is not None: 
            # crop from pad off to the fixed period (laser off generally)
            cropped_coords = coords.loc[(coords["t"] >= time_pad_off) &
                                        (coords["t"] <= time_pad_off + fixed_period)].reset_index(drop=True)
            
            # calculated straigth distance between pad off and lever position
            dist = self._scale(lever_pos - cropped_coords[["x", "y"]].iloc[0])
            direct = np.linalg.norm(dist)

            # self._show_traj(cropped_coords)  # for debugging
            
        else : 

            # crop at pad off to skip beginning
            pad_off_coords = coords.loc[coords["t"] >= time_pad_off].reset_index(drop=True)

            # crop coordinates from the pad off to the lever reach
            time_reach_lever =  self._get_lever_reaching_time(pad_off_coords, lever_pos)
            cropped_coords = pad_off_coords.loc[coords["t"] <= time_reach_lever].reset_index(drop=True)
            
            direct = self.net_displacement(cropped_coords)

            # self._show_traj(cropped_coords)  # for debugging
        
        if direct == 0:
            print("\nDIRECT IS 0")
            print(cropped_coords[["x", "y"]].iloc[0], lever_pos)
            return 0

        return self.path_length(cropped_coords) / direct

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
    

    def relative_mean_speed(self, view: str, coords: pd.DataFrame | None = None) -> float: 
        """
        Relative Average speed = sum(relative distance between eeach points) / duration
        The distance between each point takes into account if the paw goes up or down
        """
        if coords is None:
            coords = self.coords

        if len(coords) < 2:
            return 0.0

        x, y = self._cartesian_xy(view, coords)

        # Cmpute distance between each points
        dx = x.diff()
        dy = y.diff()
        dx = self._scale(dx)
        dy = self._scale(dy)

        step_dist = (dx**2 + dy**2) ** 0.5

        # Sign rule:
        # +1 if both increasing or unchanged
        # -1 if either x or y decreases
        sign = ((dx >= 0) & (dy >= 0)).astype(int).replace({0: -1})
        length = (step_dist * sign).iloc[1:].sum()

        return length / self.duration(coords)
    

    def relative_speed(self, view: str, coords: pd.DataFrame | None = None) -> pd.DataFrame: 
        """
        Relative speed :
        The distance between each point takes into account if the paw goes up or down
        """
        if coords is None:
            coords = self.coords

        if len(coords) < 2:
            return pd.DataFrame({"t": [], "velocity": []})

        x, y = self._cartesian_xy(view, coords)

        # Cmpute distance between each points
        dx = x.diff()
        dy = y.diff()
        dt = coords["t"].diff()

        dx = self._scale(dx)
        dy = self._scale(dy)

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
    


