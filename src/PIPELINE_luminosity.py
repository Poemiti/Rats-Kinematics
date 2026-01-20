#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sort_files import make_database, is_csv, is_video
from database_filter import Model, Controller, View
from trajectory_analysis import get_luminosity
from led_detection import classify_clip



# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
PREDICTION_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database" 

# ------------------------------------ make database ---------------------------------------

raw_database = make_database(CLIP_DIR, is_video)

model = Model(raw_database) # or DATABASE_PRED
view = View()
controller = Controller(model, view)
view.mainloop()

DATABASE = controller.filtered_dataset.reset_index(drop=True)
print(DATABASE)

# save database
dataset_name = f"{controller.dataset_name.get().strip()}.csv"
DATABASE.to_csv(DATABASE_DIR / dataset_name)
print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")
print(f"Number of files in {dataset_name} : {len(DATABASE)}")

# DATABASE = pd.read_csv(GENERATED_DATA_DIR / "database/#517_CHR_beta_H001.csv")

RAT_NAME = DATABASE['rat_name'][0]

OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "trajectory_figures" / RAT_NAME
LUMINOSITY_DIR = GENERATED_DATA_DIR / "luminosity" / RAT_NAME
OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)
LUMINOSITY_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------ get luminosity + classify ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 1

for video_path in DATABASE["filename"] : 
    if COUNTER >= COUNTER_LIMIT : 
        break


    video_path = Path(video_path)

    html_output_path = LUMINOSITY_DIR / f"{video_path.stem}_luminosity.html"
    csv_output_path = LUMINOSITY_DIR / f"{video_path.stem}_luminosity.csv"

    luminosities: pd.DataFrame = get_luminosity(annotation_num=1802,        
                                video_path= video_path,
                                fig_output_path= html_output_path,
                                csv_ouput_path = csv_output_path,
                                max_n_frames=None,
                                label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
    )

    # clean dataframe
    luminosities.columns = luminosities.columns.droplevel(0) # columns = LED_1 ...
    luminosities = luminosities.drop([1]).reset_index(drop=True)  # remove useless row

    print(luminosities)

    classify_clip(video_path, luminosities)

    COUNTER += 1


