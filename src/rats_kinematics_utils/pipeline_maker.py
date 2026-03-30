#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import yaml 
import numpy as np
import joblib
import sys

from rats_kinematics_utils.file_management import make_database, is_csv, is_video, get_date, get_condition, get_clip_number, get_laser_intensity

# ------------------------------------ make database out of directory ---------------------------------------


def load_database(files_path, database_path, source: str): 
    import rats_kinematics_utils.database_filter as db

    if source == "video" :
        raw_database = make_database(files_path, is_video)
    elif source == "csv" : 
        raw_database = make_database(files_path, is_csv)
    else : 
        raise ValueError("Invalid source. Available source are :\n\tcsv \n\tvideo \nChange source type in 'config.yaml'")    

    model = db.Model(raw_database, database_path)
    view = db.View()
    controller = db.Controller(model, view)
    view.mainloop()

    if controller.filtered_dataset is None: 
        print("No database seletected, stop !")
        sys.exit()

    database = controller.filtered_dataset.reset_index(drop=True)
    print(database)

    # save database
    if controller.dataset_name.get() :
        dataset_name = f"{controller.dataset_name.get().strip()}.csv"
        database.to_csv(database_path / dataset_name)
        print(f"\nFiltered dataset saved as : {database_path / dataset_name}")

    print(f"\nNumber of files in database : {len(database)}")

    return database



def load_figure_maker(dir, single_plot: bool) : 
    import rats_kinematics_utils.figures_maker as fg

    model = fg.Model(dir, single_plot)
    view = fg.View()
    controller = fg.Controller(model, view)
    view.mainloop()

    filenames : list[Path] = controller.selected_files
    plot_choice : list[bool] = controller.selected_functions

    return filenames, plot_choice



def load_preprocess_validator(dir) : 
    import rats_kinematics_utils.preprocess_validator as pv

    model = pv.Model(dir) 
    view = pv.View()
    controller = pv.Controller(model, view)
    view.bind_keys(controller)
    view.mainloop()

    if view.stop_requested : 
        return

    return model.validation


# ------------------------------------------ store metrics ------------------------------------


def init_metadata(coords: Path, lum: Path, clip: Path) : 
    return {
            "name" : clip.stem,
            "filename_coords" : coords,
            "filename_luminosity" : lum,
            "filename_clips" : clip,
            "date" : get_date(coords.stem), # datetime object
            "condition" : get_condition(coords.stem),
            "nb_clip" : get_clip_number(coords.stem),
            "laser_intensity" : get_laser_intensity(coords.stem),
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


def check_times(time_pad_off, time_laser_on, laser_duration):
    if time_pad_off is None:
        print("  ! Pad off time is None")
        return False

    if time_laser_on is not None and time_laser_on + laser_duration > 3:
        print(f"  ! Laser window out of bounds (laser_on={time_laser_on})")
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



def check_analysis_choice(files, choice) : 
    if not choice or all(v is False for v in choice.values()):
        print("No plotting option selected ! Stop")
        sys.exit()

    if len(files) == 0 : 
        print("No files selected ! Stop")
        sys.exit()



def check_trial_success(cfg, trial, may_restriction: bool = False) : 
    if not trial[cfg.bodypart]["trial_success"] :
        print("Trial success : ", trial[cfg.bodypart]["trial_success"])
        return False 
    if may_restriction : 
        if "202405" in  trial["filename_clips"].as_posix() or \
            "052024" in trial["filename_clips"].as_posix() : 
            print("Trial made in may 2024, skipped")
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



def make_output_path(base_dir, file_name):
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / file_name


# ----------------- display info ---------------------------

def print_analysis_info(cfg, analysis) : 
    print(f"\n====== {analysis} of {cfg.rat_name} ======")
    print(f"  bodypart : {cfg.bodypart}")
    print(f"  view : {cfg.view}")
    print(f"  task : {cfg.task} -> {cfg.task_pad}")
    print(f"  cm per pixel : {cfg.cm_per_pixel}")
    print(f"============================================\n")


def print_interRat_analysis_info(filenames: list[Path], available_functions: dict) : 
    print("Available rats: ")
    rats = [file.parent.stem for file in filenames]
    seen = []
    for r in rats : 
        if not r in seen : 
            seen.append(r)
            print("  ", r)

    print("\nAvailable files:")
    for file in filenames : 
        print(f"  {file.parent.name}/{file.stem}" )

    print("\nAvailable functions")
    for function_name in available_functions.keys() : 
        print("  ", function_name)
    print("==============================================\n")



def dataframe_report(df: pd.DataFrame, include_na=False, sort=True):
    """
    Analyze categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    include_na : bool (default=False)
        Whether to include NaN values in counts
    sort : bool (default=True)
        Whether to sort categories by frequency (descending)
        
    Returns:
    --------
    dict
        Dictionary where keys are column names and values are
        DataFrames containing counts and percentages per category.
    """
    
    results = {}
    
    # Select categorical and object columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=not include_na)
        percentages = df[col].value_counts(normalize=True, dropna=not include_na) * 100
        
        summary = pd.DataFrame({
            'count': counts,
            'percentage (%)': percentages.round(2)
        })
        
        if sort:
            summary = summary.sort_values(by='count', ascending=False)
        
        results[col] = {
            'summary': summary,
            'total_non_null': df[col].count(),
            'num_unique_categories': df[col].nunique(dropna=not include_na)
        }
    
    return results