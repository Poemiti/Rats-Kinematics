#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.file_management import make_database, is_csv, is_video, is_left_view, verify_exist
from utils.database_filter import Model, Controller, View
from utils.led_detection import get_luminosity, rename_file, define_cue_type, is_led_on
from config import load_config


# ------------------------------------ setup path ---------------------------------------

cfg = load_config()

# ------------------------------------ make database out of directory ---------------------------------------

def load_video_database() -> pd.DataFrame:
    return make_database(cfg.paths.clips, is_video)


def load_csv_database() -> pd.DataFrame:
    return make_database(cfg.paths.coords, is_csv)

def load_database(source: str): 
    if source == "video" :
        raw_database = make_database(cfg.paths.clips, is_video)
    elif source == "csv" : 
        raw_database = make_database(cfg.paths.coords, is_csv)
    else : 
        raise ValueError("Invalid source. Available source are :\n\tcsv \n\tvideo \nChange source type in 'config.yaml'")

    model = Model(raw_database, cfg.paths.database)
    view = View()
    controller = Controller(model, view)
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





# ------------------------------------------ store metrics ------------------------------------


def init_overall_metrics() : 
    return {
        "average distance" : 0,
        "average velocity" : 0,
        "peak velocity" : 0,
        "tortuosity" : 0,
        "metrics_per_trial" : []
    }

def init_trial_metrics(coords: Path, lum: Path, clip: Path) : 
    return {
            "filename coords" : coords,
            "filename luminosity" : lum,
            "filename clips" : clip,

            "trial success" : False,
            "pad off" : 0.0,
            "laser on" : 0.0,
            "instant velocity" : None,
            "acceleration" : None,

            "xy raw" : None,
            "xy filtered" : None,
            "xy pad off" : None,
            "xy laser on" : None
        }



db = load_database(cfg.source)
RAT_NAME = db['rat_name'][0]

lum_dir = cfg.paths.luminosity / RAT_NAME
lum_dir.mkdir(parents=True, exist_ok=True)

print(lum_dir)

