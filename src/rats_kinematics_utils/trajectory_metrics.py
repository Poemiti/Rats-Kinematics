import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.stats.stattools import medcouple
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import UnivariateSpline, make_splrep, splev


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

        v = self.instant_velocity(coords)
        peaks, _ = find_peaks(v["velocity"].dropna())
        if len(peaks) == 0:
            return np.nan
        p = v.iloc[peaks].max()
        return p["velocity"]

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



def define_likelihood_threshold(coords: pd.DataFrame, thresh: float, percentile: float = 0.05) -> float : 
    """
    Define the threshold of low likelihood coordinates
    Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
    """
    if percentile is not None:
        computed_tresh = coords["likelihood"].quantile(percentile / 100.0)
    else:
        computed_tresh = thresh

    return computed_tresh


def filter_likelihood(coords: pd.DataFrame, thresh: float, percentile: float = None) -> pd.DataFrame : 
    """
    Returns the filtered coordinates based on the threshold of low likelihood coordinates.
    Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
    Low confidance point will be set to NaN
    """
    computed_thresh = define_likelihood_threshold(coords, thresh, percentile)
    # print(f"\nthreshold set to: {computed_thresh}")
    mask = coords["likelihood"] > computed_thresh
    filtered_coords = coords.copy()
    filtered_coords.loc[~mask, ["x", "y"]] = np.nan
    return filtered_coords













def _remove_consecutiv_outliers(mask, max_len=2):
    clean_mask = mask.copy()
    mask_size = len(mask)

    i = 0
    while i < mask_size:
        if clean_mask[i]:
            start = i

            while i < mask_size and clean_mask[i]:
                i += 1

            end = i
            length = end - start

            left_bad = start > 0 and not clean_mask[start-1]
            right_bad = end < mask_size and not clean_mask[end]

            if left_bad and right_bad and length <= max_len:
                clean_mask[start:end] = False
        else:
            i += 1

    return clean_mask






def filter_outliers(coords: pd.DataFrame, stat_method: str = 'mad') -> pd.DataFrame : 
    """
    Detect outliers in coordinates and put them to NaN

    :param coords: coordinates (must contain x, y columns)
    :param stat_method:
      - 'regression': dists distance to computed polynomial regression < threshold
      - 'eucli': euclidian distance between 2 consecutive points > threshold
    """
    filtered_coords = coords.copy()

    t = coords["t"].to_numpy(dtype=float)
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)


    if stat_method == "regression" : 
        s_factor=3.0
        k=2
        s=1000

        # Fit splines 
        tck_x = make_splrep(t, x, k=k, s=s)
        tck_y = make_splrep(t, y, k=k, s=s)

        # Predicted smooth trajectory
        x_pred = splev(t, tck_x)
        y_pred = splev(t, tck_y)

        # Compute dists distance
        dists = np.sqrt((x - x_pred)**2 + (y - y_pred)**2)
        thresh = np.nanmean(dists) + s_factor * np.nanstd(dists)
        mask = dists < thresh

        params = (dists, thresh, mask)


    if stat_method == "eucli" : 
        threshold = 40 # pixel

        # compute displacement
        diffs = coords[["x","y"]].diff()
        dists = np.sqrt((diffs**2).sum(axis=1))

        # remove consecutive outliers
        mask = dists < threshold
        mask = _remove_consecutiv_outliers(mask, max_len=3)
        mask = ~mask

        params = (dists, threshold, mask)

    else:
        raise ValueError(f"Unknown stat_method '{stat_method}'")
    
    filtered_coords.loc[mask, ["x", "y"]] = np.nan
    return filtered_coords, params












def interpolate_data(coords: pd.DataFrame, method: str, max_gap: int, displacement_threshold: float | None = None) -> pd.DataFrame:
    """
    Interpolates missing values (NaN) in coordinates dataframe.
    Only for a number of consecutive missing values under max_gap

    :param method: 
        - 'zero'
        - 'linear'
        - 'splinear'
        - 'cubic'
        - 'spline'
    """
    coords_interpolated = coords.copy() 

    # Minimum valid points required per method
    min_points = {
        'zero': 2,
        'linear': 2,
        'slinear': 2,
        'cubic': 4,
        'spline': 4
    }

    for col in ["x", "y"]:
        series = coords[col]
        before_nans = series.isna().sum()
        valid = series.dropna()

        # Determine if fallback to linear is needed
        use_method = method
        if len(valid) < min_points.get(method, 2):
            print(f"Column {col} has only {len(valid)} valid points; falling back to linear interpolation.")
            use_method = 'linear'

        # Perform interpolation for interior gaps
        if use_method == 'spline':
            # Use a cubic spline of order 3
            interp_series = series.interpolate(
                method='spline',
                order=3,
                limit=max_gap,
                limit_direction='both'
            )
        else:
            interp_series = series.interpolate(
                method=use_method,
                limit=max_gap,
                limit_direction='both'
            )
        # Fill leading/trailing small gaps via backward/forward fill
        interp_series = interp_series.bfill(limit=max_gap)
        interp_series = interp_series.ffill(limit=max_gap)

        after_nans = interp_series.isna().sum()
        # print(f"Number of NaNs before interpolation : {before_nans}, after : {after_nans}")

        coords_interpolated[col] = interp_series

    # Revert large displacements to NaN if threshold is set
    if displacement_threshold is not None:
        dx = coords_interpolated["x"].diff()
        dy = coords_interpolated["y"].diff()
        displacement = (dx ** 2 + dy ** 2) ** 0.5
        exceed = displacement > displacement_threshold
        coords_interpolated.loc[exceed, "x"] = float('nan')
        coords_interpolated.loc[exceed, "y"] = float('nan')
        print(f"{exceed.sum()} frames exceeded displacement threshold and were reverted to NaN")

    return coords_interpolated