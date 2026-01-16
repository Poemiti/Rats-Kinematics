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



def dlc_predict_Julien(model_path: Path, video_path: Path, output_csv_path : Path = None) -> xr.DataArray:
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


def annotate_video_from_xr(video_path: Path, output_path: Path, pose: xr.DataArray, radius=5, likelihood_threshold: int = 0.5):
    import cv2
    import numpy as np
    import xarray as xr
    import matplotlib.cm as cm
    import numba

    radius=5

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    num_bodyparts = pose.sizes["bodyparts"]
    num_frames = pose.sizes["frame_num"]

    # Colors per bodypart
    cmap = cm.get_cmap("jet", num_bodyparts)
    colors = np.array([tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_bodyparts)])

    # Precompute circle offsets
    
    def circle_offsets(radius):
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = x**2 + y**2 <= radius**2

        ys, xs = np.where(circle)  # matching shapes (N,)
        ys = ys - radius           # convert grid index back to coordinates
        xs = xs - radius

        return np.column_stack((xs, ys))


    circle_coords = [circle_offsets(radius) for _ in range(num_bodyparts)]

    # Body part coordinates
    x = (
        pose.sel(coords="x")
        .transpose("frame_num", "bodyparts")
        .fillna(-radius-1)
        .to_numpy()
        .astype(int)
    )

    y = (
        pose.sel(coords="y")
        .transpose("frame_num", "bodyparts")
        .fillna(-radius-1)
        .to_numpy()
        .astype(int)
    )

    p = pose.sel(coords="likelihood").transpose("frame_num", "bodyparts").to_numpy()

    @numba.njit
    def stamp_circles(frame, xs, ys, ps, coords_list, colors, threshold= likelihood_threshold):
        num_bodyparts = xs.shape[0]
        frame_h, frame_w, _ = frame.shape

        for bp in range(num_bodyparts):
            if ps[bp] <= threshold:
                continue

            cx = xs[bp]
            cy = ys[bp]

            if cx <= -radius or cy <= -radius:
                continue

            coords = coords_list[bp]
            color = colors[bp]

            for k in range(coords.shape[0]):
                xi = cx + coords[k, 0]
                yi = cy + coords[k, 1]

                if 0 <= xi < frame_w and 0 <= yi < frame_h:
                    for c in range(3):
                        frame[yi, xi, c] = color[c]

    # Main loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        stamp_circles(frame, x[i], y[i], p[i], circle_coords, colors)

        out.write(frame)

    cap.release()
    out.release()



def annotate_video_from_csv(video_path: Path, csv_path: Path, output_path: Path, radius=5, likelihood_threshold=0.8, ):
    import cv2
    import numpy as np
    import matplotlib.cm as cm
    import numba

    
    # Load CSV
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Drop scorer level to get only (bodypart, coord)
    df.columns = df.columns.droplevel(0)

    bodyparts = list(df.columns.get_level_values(0).unique())
    bodyparts.remove("bodyparts")
    num_bodyparts = len(bodyparts)
    num_frames = len(df)

    # Extract arrays: (frames, bodyparts)
    x = np.stack([df[bp]["x"].to_numpy() for bp in bodyparts], axis=1).astype(int)
    y = np.stack([df[bp]["y"].to_numpy() for bp in bodyparts], axis=1).astype(int)
    p = np.stack([df[bp]["likelihood"].to_numpy() for bp in bodyparts], axis=1)

    # Replace NaNs with off-screen values
    x[np.isnan(x)] = -radius - 1
    y[np.isnan(y)] = -radius - 1

    # Video IO
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        str(output_path), fourcc, fps, (frame_width, frame_height)
    )

    # Colors per bodypart
    cmap = cm.get_cmap("jet", num_bodyparts)
    colors = np.array(
        [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_bodyparts)],
        dtype=np.uint8,
    )

    # Precompute circle offsets

    def circle_offsets(radius):
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = x**2 + y**2 <= radius**2

        ys, xs = np.where(circle)  # matching shapes (N,)
        ys = ys - radius           # convert grid index back to coordinates
        xs = xs - radius

        return np.column_stack((xs, ys))

    circle_coords = [circle_offsets(radius) for _ in range(num_bodyparts)]

    # Fast stamping
    @numba.njit
    def stamp_circles(frame, xs, ys, ps, coords_list, colors, threshold):
        frame_h, frame_w, _ = frame.shape

        for bp in range(xs.shape[0]):
            if ps[bp] < threshold:
                continue

            cx = xs[bp]
            cy = ys[bp]

            if cx < 0 or cy < 0:
                continue

            coords = coords_list[bp]
            color = colors[bp]

            for k in range(coords.shape[0]):
                xi = cx + coords[k, 0]
                yi = cy + coords[k, 1]

                if 0 <= xi < frame_w and 0 <= yi < frame_h:
                    frame[yi, xi, 0] = color[0]
                    frame[yi, xi, 1] = color[1]
                    frame[yi, xi, 2] = color[2]

    # Main loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        stamp_circles(frame, x[i], y[i], p[i], 
                      circle_coords, colors, likelihood_threshold)

        out.write(frame)

    cap.release()
    out.release()






if __name__ == "__main__":

    # Disable "weights only" before analyzing
    set_load_weights_only(False)

    # -------------------------------------- setup path ------------------------------------

    DATABASE_PATH = "../exploration/no_KO_video_list.csv"
    DATABASE = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(DATABASE.iloc[0]["filename"])

    # inputs (they should already exist)
    GENERATED_DATA_DIR = Path("../exploration/data")
    MODEL_PATH = Path("/media/filer2/T4b/Models/DLC/REJANE_rat_right_model-2025-06-18/DLC-project-2025-06-18")
    INPUT_VIDEO_PATH = GENERATED_DATA_DIR / "direct_clips" / VIDEO_EXEMPLE.stem / "clip_00.mp4"

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


    # -------------------------------------- video anotation ------------------------------------

    print("\nVideo Annotation ...\n")

    anot_start = time.perf_counter()

    annotate_video_from_xr(INPUT_VIDEO_PATH, 
                           OUTPUT_VIDEO_PATH / f"annotated_{INPUT_VIDEO_PATH.stem}.mp4", 
                           dlc_points_xr, radius=5, likelihood_threshold=0.5) # default radius

    anot_end = time.perf_counter()

    # -------------------------------------- video anotation ------------------------------------

    anot_start_csv = time.perf_counter()

    annotate_video_from_csv(INPUT_VIDEO_PATH, 
                            OUTPUT_CSV_PATH / f"pred_results_{INPUT_VIDEO_PATH.stem}.csv", 
                            OUTPUT_VIDEO_PATH / f"annotated_csv_{INPUT_VIDEO_PATH.stem}.mp4", 
                            radius=5, likelihood_threshold=0.5)
    
    anot_end_csv = time.perf_counter()

    # -------------------------------------- display performance ------------------------------------

    Rej_pred_time = Rej_end - Rej_start
    Jul_pred_time = Jul_end - Jul_start
    annotation_time = anot_end - anot_start
    annotation_time_csv = anot_end_csv - anot_start_csv

    total_n_clip = 27692               # 27692 : calculated in split_video_by_trial.py
    total_time_pred_Rej = ((Rej_pred_time * total_n_clip) / 60 ) / 60 # h
    total_time_pred_Jul = ((Jul_pred_time * total_n_clip) / 60 ) / 60 # h
    total_time_annotation = ((annotation_time * total_n_clip) / 60 ) / 60 # h
    total_time_annotation_csv = ((annotation_time_csv * total_n_clip) / 60 ) / 60 # h

    print(f"\nPerformance : ")
    print(f"  Prediction time (for 1 clip) - Rejane Method  :  {Rej_pred_time:.2f} sec")
    print(f"  Prediction time (for 1 clip) - Julien Method  :  {Jul_pred_time:.2f} sec")
    print(f"  Video annotation time (xarray)                :  {annotation_time:.2f} sec")
    print(f"  Video annotation time (csv)                   :  {annotation_time_csv:.2f} sec\n")

    print(f"\nOVERALL PERFOMANCE PREDICTION")
    print(f"  Prediction time - Rejane Method  :  {total_time_pred_Rej:.2f} h")
    print(f"  Prediction time - Julien Method  :  {total_time_pred_Jul:.2f} h")
    print(f"  Video annotation time (xarray)   :  {total_time_annotation:.2f} h")
    print(f"  Video annotation time (csv)      :  {total_time_annotation_csv:.2f} h\n")

    # Performance : 
    #   Prediction time (for 1 clip) - Rejane Method  :  30.74 sec
    #   Prediction time (for 1 clip) - Julien Method  :  30.63 sec
    #   Video annotation time (xarray)                :  1.06 sec
    #   Video annotation time (csv)                   :  0.79 sec


    # OVERALL PERFOMANCE PREDICTION
    #   Prediction time - Rejane Method  :  236.47 h
    #   Prediction time - Julien Method  :  235.63 h
    #   Video annotation time (xarray)   :  8.16 h
    #   Video annotation time (csv)      :  6.05 h


