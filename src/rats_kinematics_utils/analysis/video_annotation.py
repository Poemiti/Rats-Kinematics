from pathlib import Path
import xarray as xr
import pandas as pd
import cv2
import numpy as np
import matplotlib
import numba
from skimage.draw import line_aa
import yaml


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




def _get_skeleton(path: Path) -> list[list] : 
    """
    Open a yaml config file with the bodyparts pairs
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg.get("skeleton", [])



def annotate_video_from_csv(video_path: Path, csv_path: Path, output_path: Path, radius=5, likelihood_threshold=0.8, draw_skeleton: bool = False):
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
                    frame[yi, xi, 0] = color[0]     # red
                    frame[yi, xi, 1] = color[1]     # green
                    frame[yi, xi, 2] = color[2]     # blue


    def stamp_skeleton(frame, xs, ys, ps, skeleton, bodypart_to_idx, threshold):
        for bp1, bp2 in skeleton:

            i1 = bodypart_to_idx[bp1]
            i2 = bodypart_to_idx[bp2]

            # Check likelihood
            if ps[i1] < threshold or ps[i2] < threshold:
                continue

            x1, y1 = xs[i1], ys[i1]
            x2, y2 = xs[i2], ys[i2]

            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                continue

            # Draw line (BGR: black)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)


    # Main loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        stamp_circles(frame, x[i], y[i], p[i], 
                      circle_coords, colors, likelihood_threshold)
        
        if draw_skeleton :
            skeleton = _get_skeleton("./info_skeleton.yaml")
            bodypart_to_idx = {bp: i for i, bp in enumerate(bodyparts)}

            stamp_skeleton(
                frame, x[i], y[i], p[i],
                skeleton,
                bodypart_to_idx,
                likelihood_threshold
            )

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







def annotate_behavior_box(csv_path, video_path, bodypart_name, output_path, boxes, lever_pos, pad_pos):

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


    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # persistent trajectory canvas (IMPORTANT)
    traj_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    radius = 3
    alpha = 1
    color = np.array([0, 0, 255], dtype=np.uint8)  # red (BGR)

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


    def draw_alpha_rect(img, pt1, pt2, color, alpha):
        overlay = img.copy()

        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


    i = 0

    while True:
        ret, frame = cap.read()
        if not ret or i >= num_frames:
            break

        overlay = frame.copy()

        # ---------------------------------------------------
        # static UI overlays
        # ---------------------------------------------------
        cv2.drawMarker(overlay, lever_pos, (100, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 3)
        cv2.drawMarker(overlay, pad_pos, (100, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 3)

        reach = boxes["reach"]
        open = boxes["open"]
        grasp = boxes["grasp"]
        press = boxes["press"]

        draw_alpha_rect(overlay, reach[0:2], reach[2:4], (0, 255, 255), 0.2)  # reach
        draw_alpha_rect(overlay, open[0:2], open[2:4], (0, 165, 255), 0.2)    # open
        draw_alpha_rect(overlay, grasp[0:2], grasp[2:4], (255, 0, 0), 0.2)    # grasp
        draw_alpha_rect(overlay, press[0:2], press[2:4], (0, 255, 0), 0.2)    # press

        # frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)

        # ---------------------------------------------------
        # trajectory stamping (FIXED)
        # ---------------------------------------------------
        stamp_circle(
            traj_overlay,
            int(x[i]),
            int(y[i]),
            float(p[i]),
            circle_coords,
            color,
            0.1,
        )

        output_frame = cv2.addWeighted(frame, 1.0, traj_overlay, 1.0, 0)

        out.write(output_frame)

        i += 1

    cap.release()
    out.release()

    # print("Saved:", output_path)







def annotate_traj_with_laser(coords: pd.DataFrame, 
                             laser_on: bool, 
                             time_pad_off: float, 
                             video_path: Path, 
                             output_path: Path): 
    
    num_frames = len(coords)
    frame_laser_start = int((time_pad_off + 0.025) * 125)
    frame_laser_end = int((time_pad_off + 0.3) * 125)

    # Extract only the selected bodypart
    x = coords["x"].to_numpy().astype(int)
    y = coords["y"].to_numpy().astype(int)

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
    radius=3
    default_color = np.array([255, 0, 0], dtype=np.uint8)  # Blue (BGR)
    laserOFF_color = np.array([0, 0, 255], dtype=np.uint8)  # Red (BGR) -> Laser Off
    laserON_color = np.array([0, 255, 0], dtype=np.uint8)  # Green (BGR) -> Laser On

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
    def stamp_circle(frame, cx, cy, circle_coords, color):
        h, w, _ = frame.shape

        for k in range(circle_coords.shape[0]):
            xi = cx + circle_coords[k, 0]
            yi = cy + circle_coords[k, 1]

            if 0 <= xi < w and 0 <= yi < h:
                frame[yi, xi, 0] = color[0]
                frame[yi, xi, 1] = color[1]
                frame[yi, xi, 2] = color[2]

    # Main loop
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if laser_on and frame_laser_start <= i <= frame_laser_end: 
            color = laserON_color
        elif not laser_on and frame_laser_start <= i <= frame_laser_end: 
            color = laserOFF_color
        else : 
            color = default_color

        # Stamp onto the persistent overlay
        stamp_circle(
            overlay,
            x[i],
            y[i],
            circle_coords,
            color,
        )

        # Combine original frame + trajectory (overlay)
        output_frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        out.write(output_frame)

    cap.release()
    out.release()





if __name__ == "__main__" : 

    print("no main")