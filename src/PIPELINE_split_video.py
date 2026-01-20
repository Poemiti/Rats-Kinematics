#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from sort_files import is_video, make_database
from split_video import split_video

from database_filter import Model, View, Controller



# ------------------------------------ setup path ---------------------------------------

INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")

GENERATED_DATA_DIR = Path("../data")
DATABASE_DIR = GENERATED_DATA_DIR / "database"

GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------ setup parameters ---------------------------------------

# duration of the resulting clip
DURATION = 12.5  # sec

# ------------------------------------ filter video ---------------------------------------

raw_database = make_database(INPUT_VIDEO_DIR, is_video)

model = Model(raw_database) # or DATABASE_PRED
view = View()
controller = Controller(model, view)
view.mainloop()

DATABASE = controller.filtered_dataset.reset_index(drop=True)

RAT_NAME = DATABASE["rat_name"][0]
OUTPUT_CLIP_DIR = GENERATED_DATA_DIR / "clips" / RAT_NAME
OUTPUT_CLIP_DIR.mkdir(parents=True, exist_ok=True)

print(DATABASE)
# save database
dataset_name = f"{controller.dataset_name.get().strip()}.csv"
DATABASE.to_csv(DATABASE_DIR / dataset_name)
print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")
print(f"Number of files in {dataset_name} : {len(DATABASE)}")


# ------------------------------------ do the spliting on the filtered database ---------------------------------------


COUNTER = 0
COUNTER_LIMIT = 1

for video_path in DATABASE["filename"] : 
    if COUNTER >= COUNTER_LIMIT : 
        break

    video_path = Path(video_path)
    output_clip_dir = OUTPUT_CLIP_DIR / video_path.stem
    output_clip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting video {video_path.stem}\n")

    split_video(input_path = video_path,
                output_path = output_clip_dir,
                CLIP_DURATION = DURATION)

    COUNTER += 1
