import deeplabcut
import shutil
from pathlib import Path
import time
import xarray as xr

from deeplabcut.pose_estimation_pytorch import set_load_weights_only

def run_deeplabcut_analysis(
    model_path: str,
    input_video_path: str,
    temp_base_dir: str,
    save_as_csv: bool = True,
) -> Path:
    """
    Run DeepLabCut analysis on a single video and return the temporary output directory.
    """
    input_video_path = Path(input_video_path)

    analysis_output_path = (
        Path(temp_base_dir) / f"analysis_{input_video_path.stem}"
    )
    analysis_output_path.mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_videos(
        f"{model_path}/config.yaml",
        [str(input_video_path)],
        save_as_csv=save_as_csv,
        destfolder=str(analysis_output_path),
    )

    return analysis_output_path


def move_outputs(
    analysis_output_path: Path,
    output_h5_path: str | None = None,
    output_csv_path: str | None = None,
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



def annotate_video(video_path: Path, output_path: Path, pose: xr.DataArray, radius=5):
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
    def stamp_circles(frame, xs, ys, ps, coords_list, colors, threshold=0.8):
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



if __name__ == "__main__":

    # Disable "weights only" before analyzing
    set_load_weights_only(False)

    model_path = f"/media/filer2/T4b/Models/DLC/DLC-Project-2025-03-13/"
    input_video_path = "./clips/clip_00.mp4"
    # input_video_path = "../exploration/ouput_3sec_125fps.mp4"
    

    output_h5_path = f"../data/dlc_results/result_predict_h5.h5"
    output_csv_path = f"../data/csv_results/result_predict_csv_3.csv"

    temp_base_dir = "../data/temporary"

    start = time.perf_counter()
    # 1. make a prediction
    analysis_output_path = run_deeplabcut_analysis(model_path, input_video_path,
                                                   temp_base_dir, save_as_csv=True,)

    # 2. move the outputs 
    move_outputs(analysis_output_path, output_h5_path, output_csv_path)

    # remove temporary directory
    cleanup_temp_directory(analysis_output_path)
    end = time.perf_counter()


    print(f"\nPerformance : ")
    print(f"  Prediction time (for 1 clip)   :  {(end - start):2f} sec")
