#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.file_management import make_database, is_csv, is_video, is_left_view
from utils.database_filter import Model, Controller, View
from utils.led_detection import get_luminosity, rename_file, define_cue_type, is_led_on



# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
TRAJECTORY_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database" 

# ------------------------------------ make database out of the clip directory ---------------------------------------

raw_database = make_database(CLIP_DIR, is_video)

model = Model(raw_database, DATABASE_DIR)
view = View()
controller = Controller(model, view)
view.mainloop()

DATABASE = controller.filtered_dataset.reset_index(drop=True)
print(DATABASE)

# save database
if controller.dataset_name.get() :
    dataset_name = f"{controller.dataset_name.get().strip()}.csv"
    DATABASE.to_csv(DATABASE_DIR / dataset_name)
    print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")

print(f"\nNumber of files in database : {len(DATABASE)}")

RAT_NAME = DATABASE['rat_name'][0]

LUMINOSITY_DIR = GENERATED_DATA_DIR / "luminosity" / RAT_NAME
LUMINOSITY_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------ get luminosity + classify ---------------------------------------

COUNTER = 0
COUNTER_LIMIT = 10

for video_path in DATABASE["filename"] : 
    if COUNTER >= COUNTER_LIMIT : 
        break

    print(f"\nGetting luminosity of {video_path}")
    video_path = Path(video_path)
    
    output_dir = LUMINOSITY_DIR / video_path.parent.stem  # get the folder name
    output_dir.mkdir(parents=True, exist_ok=True)

    html_output_path = output_dir / f"luminosity_{video_path.stem}.html"
    csv_output_path = output_dir / f"luminosity_{video_path.stem}.csv"

    # choose annotation number (label_studio) based on the view (voir readme)

    if is_left_view(str(video_path.stem)) : 
        label_studio_annotation = 1814
    else : 
        label_studio_annotation = 1811
    print(f"label studio annotation : {label_studio_annotation}\n")

    luminosities: pd.DataFrame = get_luminosity(annotation_num=label_studio_annotation,        
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

    # rename both clip AND csv (for later filtering)
    clip_number = str(video_path.stem)[-8:]
    trajectory_csv_dir = TRAJECTORY_DIR / str(video_path.stem)[:-8]
    trajectory_csv_path = trajectory_csv_dir / f"{str(video_path.stem)[:-8]}_pred_results{clip_number}.csv"
    
    if not trajectory_csv_path.exists() : 
        raise FileExistsError(f'This trajectory csv file does not exist : {trajectory_csv_path}')

    cue_type = define_cue_type(luminosities["LED_1"])
    led_on = is_led_on(luminosities["LED_4"])

    rename_file(video_path, laser_on=led_on, new_cue=cue_type)
    rename_file(trajectory_csv_path, laser_on=led_on, new_cue=cue_type)

    COUNTER += 1
