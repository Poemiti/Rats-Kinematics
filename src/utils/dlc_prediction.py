import deeplabcut
import shutil
from pathlib import Path
import time
import xarray as xr
import pandas as pd

from deeplabcut.pose_estimation_pytorch import set_load_weights_only

def dlc_predict_Rejane(
    model_path: Path,
    input_video_path: Path,
    temporary_path: Path,
    output_h5_path: Path,
    output_csv_path: Path,
    save_as_csv: bool = True,
) -> Path:
    """
    Run DeepLabCut analysis on a single video and return the temporary output directory.
    """

    analysis_output_path = temporary_path / f"analysis_{input_video_path.stem}"
    analysis_output_path.mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_videos(
        f"{model_path}/config.yaml",
        [str(input_video_path)],
        save_as_csv=save_as_csv,
        destfolder=str(analysis_output_path),
    )

    move_outputs(analysis_output_path, output_h5_path, output_csv_path)   
    cleanup_temp_directory(analysis_output_path)




def move_outputs(
    analysis_output_path: Path,
    output_h5_path: Path | None = None,
    output_csv_path: Path | None = None,
):
    """
    Move DeepLabCut output files from the temporary directory to final destinations.
    """
    if not output_h5_path and not output_csv_path:
        raise ValueError(
            "At least one of output_h5_path or output_csv_path must be provided."
        )

    h5_file = next(analysis_output_path.glob("*.h5"), None)
    csv_file = next(analysis_output_path.glob("*.csv"), None)

    if output_h5_path and h5_file:
        shutil.move(str(h5_file), str(output_h5_path))

    if output_csv_path and csv_file:
        shutil.move(str(csv_file), str(output_csv_path))



def cleanup_temp_directory(analysis_output_path: Path):
    """
    Remove the temporary analysis directory.
    """
    if analysis_output_path.exists():
        shutil.rmtree(analysis_output_path)



def dlc_predict_Julien(model_path: Path, 
                       video_path: Path, 
                       output_csv_path : Path = None) -> xr.DataArray:
    import deeplabcut, tempfile

    with tempfile.TemporaryDirectory() as dlc_dest:
        print(dlc_dest)
        deeplabcut.analyze_videos(
            f'{model_path}/config.yaml',
            [str(video_path)],
            save_as_csv=False,
            # gputouse=0,
            destfolder=dlc_dest
        )

        h5_file = next(Path(dlc_dest).glob("*.h5"), None)
        df = pd.read_hdf(h5_file)

    df.index.name="frame_num"

    if output_csv_path : 
        print(df)
        df.to_csv(output_csv_path)

    res =  df.stack("scorer").stack("bodyparts").stack("coords").to_xarray()

    if res.sizes["scorer"] !=1:
        raise Exception(f"Multiple scorers not supported, got {res.sizes['scorer']}")
    res = res.isel(scorer=0, drop=True)

    return res






if __name__ == "__main__":

    # Disable "weights only" before analyzing
    set_load_weights_only(False)

    # -------------------------------------- setup path ------------------------------------

    # inputs (should exist)
    GENERATED_DATA_DIR = Path("../../exploration/data")
    DATABASE_PATH = GENERATED_DATA_DIR / "database/rat_517_H001.csv"  # if it does not exist, make one with make_database (in file_management.py)
    INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
    MODEL_PATH = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/DLC-project-2025-06-18")

    # get the path for a video (path in a premade database)
    database = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(database.iloc[0]["filename"])
    INPUT_VIDEO_PATH = GENERATED_DATA_DIR / "clips" / VIDEO_EXEMPLE.stem / f"{VIDEO_EXEMPLE.stem}_clip_00.mp4"

    # outputs
    OUTPUT_H5_PATH = GENERATED_DATA_DIR / "dlc_results" / VIDEO_EXEMPLE.stem 
    OUTPUT_CSV_PATH = GENERATED_DATA_DIR / "csv_results" / VIDEO_EXEMPLE.stem 
    OUTPUT_VIDEO_PATH = GENERATED_DATA_DIR / "video_annotation" / VIDEO_EXEMPLE.stem

    # Temporary directory
    TEMPORARY_PATH = GENERATED_DATA_DIR / "temporary"

    OUTPUT_H5_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIDEO_PATH.mkdir(parents=True, exist_ok=True)
    TEMPORARY_PATH.mkdir(parents=True, exist_ok=True)

    # -------------------------------------- make prediction : Rejane version ------------------------------------
    
    print("\nDLC prediction - Rejane Methode ...\n")

    Rej_start = time.perf_counter()

    analysis_output_path = dlc_predict_Rejane(MODEL_PATH, 
                                              INPUT_VIDEO_PATH,
                                              TEMPORARY_PATH,
                                              OUTPUT_H5_PATH / f"pred_results_{INPUT_VIDEO_PATH.stem}.h5",
                                              OUTPUT_CSV_PATH / f"pred_results_{INPUT_VIDEO_PATH.stem}.csv", 
                                              save_as_csv=True,)


    Rej_end = time.perf_counter()

    # -------------------------------------- make prediction : Julien version ------------------------------------
    
    print("\nDLC prediction - Julien Methode ...\n")

    Jul_start = time.perf_counter()

    dlc_points_xr = dlc_predict_Julien(MODEL_PATH, 
                                       INPUT_VIDEO_PATH, 
                                       OUTPUT_CSV_PATH / f"pred_results_{INPUT_VIDEO_PATH.stem}.csv")

    Jul_end = time.perf_counter()


    # -------------------------------------- display performance ------------------------------------

    Rej_pred_time = Rej_end - Rej_start
    Jul_pred_time = Jul_end - Jul_start

    total_n_clip = 27692               # 27692 : calculated in split_video_by_trial.py
    total_time_pred_Rej = ((Rej_pred_time * total_n_clip) / 60 ) / 60 # h
    total_time_pred_Jul = ((Jul_pred_time * total_n_clip) / 60 ) / 60 # h

    print(f"\nPerformance : ")
    print(f"  Prediction time (for 1 clip) - Rejane Method  :  {Rej_pred_time:.2f} sec")
    print(f"  Prediction time (for 1 clip) - Julien Method  :  {Jul_pred_time:.2f} sec")

    print(f"\nOVERALL PERFOMANCE PREDICTION")
    print(f"  Prediction time - Rejane Method  :  {total_time_pred_Rej:.2f} h")
    print(f"  Prediction time - Julien Method  :  {total_time_pred_Jul:.2f} h")

    # Performance : 
    #   Prediction time (for 1 clip) - Rejane Method  :  30.74 sec
    #   Prediction time (for 1 clip) - Julien Method  :  30.63 sec

    # OVERALL PERFOMANCE PREDICTION
    #   Prediction time - Rejane Method  :  236.47 h
    #   Prediction time - Julien Method  :  235.63 h


