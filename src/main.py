from pathlib import Path
import pandas as pd
import split_video_by_trial as split
import dlc_prediction as dlc

# ------------------------------------ paths and parameters ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
GENERATED_FRAMES_DIR = Path("../data/frames")
GENERATED_VIDEOS_DIR = Path("../data/clips")
INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
DATABASE_PATH = "../exploration/no_KO_video_list.csv"

MODEL_PATH = f"/media/filer2/T4b/Models/DLC/DLC-Project-2025-03-13/"
TEMPORARY_DIR = Path("/media/filer2/T4b/Temporary")
OUTPUT_H5_PATH = f"/media/filer2/T4b/UserFolders/Réjane/result-predict-h5.h5"
OUTPUT_CSV_PATH = f"/media/filer2/T4b/UserFolders/Réjane/result-predict-csv.csv"

DURATION = 3  # sec
FPS = 125

# ------------------------------------ setup directories ---------------------------------------

GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_FRAMES_DIR.mkdir(exist_ok=True)
GENERATED_VIDEOS_DIR.mkdir(exist_ok=True)
TEMPORARY_DIR.mkdir(parents=True, exist_ok=True)

DATABASE = pd.read_csv(DATABASE_PATH)


# ------------------------------------ split videos ---------------------------------------
 
for file_path in DATABASE["filename"]:
    file_path = Path(file_path)

    frame_per_clip, n_clip = split.diplay_video_properties(file_path, DURATION)
    split.extract_frames(file_path, GENERATED_FRAMES_DIR, DURATION, FPS)


# ------------------------------------ frames to video clips mp4 ---------------------------------------

for file_path in DATABASE["filename"]:
    file_path = Path(file_path)
    frames_dir = GENERATED_FRAMES_DIR / file_path.name

    for clip_path in frames_dir.iterdir():
        output_name = GENERATED_VIDEOS_DIR / f"{file_path.stem}_{clip_path.name}.mp4"
        split.frames_to_video(clip_path, output_name, FPS)


# ------------------------------------ DLC prediction ---------------------------------------

for clip_path in GENERATED_VIDEOS_DIR.iterdir():

    analysis_output_path = dlc.run_deeplabcut_analysis(
        MODEL_PATH,
        clip_path,
        TEMPORARY_DIR,
        save_as_csv=True,
    )

    dlc.move_outputs(analysis_output_path, OUTPUT_H5_PATH, OUTPUT_H5_PATH)  ### verify path
    dlc.cleanup_temp_directory(analysis_output_path)
