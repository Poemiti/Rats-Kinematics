#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import yaml 
import numpy as np
import joblib
from datetime import datetime
import sys
from statsmodels.stats.stattools import medcouple
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import UnivariateSpline, make_splrep, splev


from rats_kinematics_utils.core.file_utils import parse_filename, get_date, get_condition, get_clip_number, get_laser_intensity


def open_DLC_results(csv_path : Path) -> pd.DataFrame : 
    """
    Load and clean a DeepLabCut CSV file.
    """

    # DLC CSV has 3 header rows (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    clean_df = df.iloc[1:].reset_index(drop=True)

    return clean_df





def init_metadata(coords: Path, lum: Path, clip: Path) : 
    metadata = parse_filename(clip.stem)
    date = get_date(coords.stem).date().isoformat()
    return {
            "name" : clip.stem,
            "filename_coords" : str(coords),
            "filename_luminosity" : str(lum),
            "filename_clips" : str(clip),
            "date" : date, # datetime object
            "condition" : metadata["condition"],
            "nb_clip" : metadata["clip"],
            "laser_intensity" : metadata["laser_intensity"],
            "rat_type": metadata["rat_type"], 
            "stim_location": metadata["stim_location"],
        }


# ------------------------------------------ verifiaction step ------------------------------------


def check_lost_coords(xy_filtered, coords, max_lost=10):
    n_lost = len(coords) - len(xy_filtered)
    if n_lost > max_lost:
        print(f"  ! Too much lost coords: {n_lost}")
        return False
    return True


def check_times(time_pad_off, time_laser_on, laser_duration):
    if time_pad_off is None:
        # print("  ! Pad off time is None")
        return False

    if time_laser_on is not None and time_laser_on + laser_duration > 3:
        # print(f"  ! Laser window out of bounds (laser_on={time_laser_on})")
        return False

    return True


def check_non_empty(xy_filtered, time_pad_off):
    if len(xy_filtered) == 0:
        print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
        return False
    return True



def check_reward(time_reward) : 
    if time_reward is None : 
        print("  ! Reward time is None")
        return False
    return True


# ------------------------------------------ function for trajectory processing ------------------------------------


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



def define_likelihood_threshold(coords: pd.DataFrame, thresh: float, percentile: float = None) -> float : 
    """
    Define the threshold of low likelihood coordinates
    Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
    """
    if percentile is not None:
        computed_tresh = coords["likelihood"].quantile(percentile / 100.0)
    else:
        computed_tresh = thresh

    return computed_tresh


def filter_likelihood(coords: pd.DataFrame, thresh: float, percentile: float = None) -> tuple[pd.DataFrame, float] : 
    """
    Returns the filtered coordinates based on the threshold of low likelihood coordinates.
    Coords must be a dataframe containing the following columns ['x', 'y', 'likelihood']
    Low confidance point will be set to NaN
    """
    computed_thresh = define_likelihood_threshold(coords, thresh, percentile)
    # print(f"\nthreshold set to: {computed_thresh}")
    mask = coords["likelihood"] > computed_thresh
    filtered_coords: pd.DataFrame = coords.copy()
    filtered_coords.loc[~mask, ["x", "y"]] = np.nan
    return filtered_coords, computed_thresh













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
    filtered_coords: pd.DataFrame = coords.copy()

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
        
