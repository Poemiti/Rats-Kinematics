#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import yaml 
import numpy as np
import joblib

from utils.file_management import make_database, is_csv, is_video
import utils.database_filter as db
import utils.figures_maker as fg

# ------------------------------------ make database out of directory ---------------------------------------

def load_video_database(cfg) -> pd.DataFrame:
    return make_database(cfg.paths.clips, is_video)


def load_csv_database(cfg) -> pd.DataFrame:
    return make_database(cfg.paths.coords, is_csv)

def load_database(cfg, source: str): 
    if source == "video" :
        raw_database = make_database(cfg.paths.clips, is_video)
    elif source == "csv" : 
        raw_database = make_database(cfg.paths.coords, is_csv)
    else : 
        raise ValueError("Invalid source. Available source are :\n\tcsv \n\tvideo \nChange source type in 'config.yaml'")

    model = db.Model(raw_database, cfg.paths.database)
    view = db.View()
    controller = db.Controller(model, view)
    view.mainloop()

    database = controller.filtered_dataset.reset_index(drop=True)
    print(database)

    # save database
    if controller.dataset_name.get() :
        dataset_name = f"{controller.dataset_name.get().strip()}.csv"
        database.to_csv(cfg.paths.database / dataset_name)
        print(f"\nFiltered dataset saved as : {cfg.paths.database / dataset_name}")

    print(f"\nNumber of files in database : {len(database)}")

    return database



def load_figure_maker(cfg, rat_name) : 

    model = fg.Model(cfg.paths.metrics / rat_name)
    view = fg.View()
    controller = fg.Controller(model, view)
    view.mainloop()

    filenames : list[Path] = controller.selected_files
    plot_choice : list[bool] = controller.selected_functions

    return filenames, plot_choice



# ------------------------------------------ store metrics ------------------------------------


def init_metrics(coords: Path, lum: Path, clip: Path) : 
    return {
            "filename_coords" : coords,
            "filename_luminosity" : lum,
            "filename_clips" : clip,

            "trial_success" : False,
            "lost_coords" : 0,
            "pad_off" : 0.0,
            "laser_on" : 0.0,

            "average_distance" : 0,
            "average_velocity" : 0,
            "peak_velocity" : 0,
            "tortuosity" : 0,
            "instant_velocity" : None,
            "acceleration" : None,

            "xy_raw" : None,
            "xy_filtered" : None,
            "xy_pad_off" : None,
            "xy_laser_on" : None
        }


def to_yaml(obj):
    if isinstance(obj, np.generic):  
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj



# ------------------------------------------ verifiaction step ------------------------------------


def check_lost_coords(xy_filtered, coords, max_lost=10):
    n_lost = len(coords) - len(xy_filtered)
    if n_lost > max_lost:
        print(f"  ! Too much lost coords: {n_lost}")
        return False
    return True


def check_times(time_pad_off, time_laser_on, n_frames, laser_duration):
    if time_pad_off is None:
        print("  ! Pad off time is None")
        return False

    if time_laser_on is not None and time_laser_on + laser_duration > n_frames - 1:
        print(f"  ! Laser window out of bounds (laser_on={time_laser_on})")
        return False

    return True


def check_non_empty(xy_filtered, time_pad_off):
    if len(xy_filtered) == 0:
        print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
        return False
    return True



# ------------------------------------------ load metrics ------------------------------------


def load_metrics(path: Path) -> dict :
    try:
        metrics = joblib.load(path)
        return metrics
    except FileNotFoundError:
        print(f"ERROR : {path} does not exist")
    except yaml.YAMLError as exc:
        print(f"Error when trying to open {path} : ", exc)
    except Exception as e:
        print("Unexpected error : ", e)



def make_output_path(base_dir, folder_name, file_name):
    path = base_dir / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path / file_name