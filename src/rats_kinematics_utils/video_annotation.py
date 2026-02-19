from pathlib import Path
import time
import xarray as xr
import pandas as pd
import cv2
import numpy as np
import matplotlib
import numba



def annotate_video_from_xr(video_path: Path, output_path: Path, pose: xr.DataArray, radius=5, likelihood_threshold: int = 0.5):
    """
    Annotate a video with pose estimation data stored in an xarray DataArray.

    This function overlays colored dots on each frame to represent
    detected body part positions. One color is assigned per body part.
    Annotations are applied only when the likelihood exceeds a given threshold.

    Parameters
    ----------
    video_path : pathlib.Path
        Path to the input video file.
    output_path : pathlib.Path
        Path where the annotated video will be saved.
    pose : xarray.DataArray
        Pose estimation data with dimensions:
        ``(frame_num, bodyparts, coords)``,
        where ``coords`` includes ``x``, ``y``, and ``likelihood``.
    radius : int, optional
        Radius (in pixels) of the circles drawn for each body part.
        Default is 5.
    likelihood_threshold : float, optional
        Minimum likelihood required to draw a body part.
        Default is 0.5.

    Returns
    -------
    None
    """

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
    # cmap = cm.get_cmap("jet", num_bodyparts)    old version
    cmap = matplotlib.colormaps.get_cmap("jet").resampled(num_bodyparts)
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
    """
    Annotate a video with pose estimation data stored in csv.

    This function overlays colored dots on each frame to represent
    detected body part positions. One color is assigned per body part.
    Annotations are applied only when the likelihood exceeds a given threshold.

    Parameters
    ----------
    video_path : pathlib.Path
        Path to the input video file.
    csv_path : pathlib.Path
        csv file where data has a multi-index header with:
        ``(frame_num, bodyparts, coords)``,
        where ``coords`` includes ``x``, ``y``, and ``likelihood``.
    output_path : pathlib.Path
        Path where the annotated video will be saved.
    radius : int, optional
        Radius (in pixels) of the circles drawn for each body part.
        Default is 5.
    likelihood_threshold : float, optional
        Minimum likelihood required to draw a body part.
        Default is 0.5.

    Returns
    -------
    None
    """

    # Load CSV
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    
    # Drop scorer level to get only (bodypart, coord)
    df.columns = df.columns.droplevel(0)
    df = df.iloc[1:].reset_index(drop=True)

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
    # cmap = cm.get_cmap("jet", num_bodyparts)    old version
    cmap = matplotlib.colormaps.get_cmap("jet").resampled(num_bodyparts)
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


def annotate_single_bodypart(video_path: Path,
                            csv_path: Path,
                            output_path: Path,
                            bodypart_name: str,
                            radius=5,
                            likelihood_threshold=0.8,
                        ):
    """
    Annotate a video with the trajectory of a single body part.

    The body part position is accumulated over time and displayed
    as a persistent trajectory overlay. Only detections exceeding
    the likelihood threshold are drawn.

    Parameters
    ----------
    video_path : pathlib.Path
        Path to the input video file.
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.
    output_path : pathlib.Path
        Path where the annotated video will be saved.
    bodypart_name : strs
        Name of the body part to visualize.
    radius : int, optional
        Radius (in pixels) of the drawn trajectory points.
        Default is 5.
    likelihood_threshold : float, optional
        Minimum likelihood required to draw a point.
        Default is 0.8.

    Returns
    -------
    None
    """

    # Load CSV
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    df = df.iloc[1:].reset_index(drop=True)

    if bodypart_name not in df:
        raise ValueError(f"{bodypart_name} not found in CSV")

    num_frames = len(df)

    # Extract only the selected bodypart
    x = df[bodypart_name]["x"].to_numpy().astype(int)
    y = df[bodypart_name]["y"].to_numpy().astype(int)
    p = df[bodypart_name]["likelihood"].to_numpy()

    # Video IO
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path), fourcc, fps, (frame_width, frame_height)
    )

    # Fixed color for the trajectory
    color = np.array([0, 0, 255], dtype=np.uint8)  # red (BGR)

    # Persistent overlay (trajectory)
    overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Circle offsets
    def circle_offsets(radius):
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        ys, xs = np.where(mask)
        return np.column_stack((xs - radius, ys - radius))

    circle_coords = circle_offsets(radius)

    @numba.njit
    def stamp_circle(frame, cx, cy, prob, coords, color, threshold):
        if prob < threshold or cx < 0 or cy < 0:
            return

        h, w, _ = frame.shape

        for k in range(coords.shape[0]):
            xi = cx + coords[k, 0]
            yi = cy + coords[k, 1]

            if 0 <= xi < w and 0 <= yi < h:
                frame[yi, xi, 0] = color[0]
                frame[yi, xi, 1] = color[1]
                frame[yi, xi, 2] = color[2]

    # Main loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Stamp onto the persistent overlay
        stamp_circle(
            overlay,
            x[i],
            y[i],
            p[i],
            circle_coords,
            color,
            likelihood_threshold,
        )

        # Combine original frame + trajectory (overlay)
        output_frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        out.write(output_frame)

    cap.release()
    out.release()




if __name__ == "__main__" : 

    print("no main")