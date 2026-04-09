from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from PIL import Image
import numpy as np
import cv2
import numba
import matplotlib.animation as animation
import seaborn as sns

# ==================================== display hyperparameter ===========================================


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("talk", style="ticks", rc=custom_params)

LASER_COLOR = "lightpink"
LINE_COLOR = "gray"
AVG_LINE_COLOR = "navy"


# --------------------------------- plotting ----------------------------------
def plot_video_from_csv(video_path: Path, csv_path: Path, output_path: Path, radius=5, likelihood_threshold=0.8, ):
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

def plot_annotated_video(video_path: Path,
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
    def stamp_circle(frame, cx, cy, prob, coords, threshold):
        if prob < threshold or cx < 0 or cy < 0:
            return

        h, w, _ = frame.shape

        r = int(255 * (1 - prob))
        g = int(255 * prob)
        b = 0

        for k in range(coords.shape[0]):
            xi = cx + coords[k, 0]
            yi = cy + coords[k, 1]

            if 0 <= xi < w and 0 <= yi < h:
                frame[yi, xi, 0] = b
                frame[yi, xi, 1] = g
                frame[yi, xi, 2] = r

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
            likelihood_threshold,
        )

        # Combine original frame + trajectory (overlay)
        output_frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

        out.write(output_frame)

    cap.release()
    out.release()





def plot_single_bodypart_trajectories(
    coords: pd.DataFrame,
    cm_per_pixel: float,
    frame_laser_on : float = None,
    ax: plt.axes = None,
    color: str = "red",
    transparancy: float = 0.7,
    marker: str = None,
    rat_background : bool=False, 
) -> plt.axes :
    """
    Plot body part trajectories from a DeepLabCut CSV file.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new figure is created.
    invert_y : bool, optional
        Whether to invert the y-axis (image coordinate convention).

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted trajectories.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if rat_background: 
        img = np.asarray(Image.open("/home/poemiti/Rats-Kinematics/data/rat_image.png"))
        ax.imshow(img, alpha=0.3)

    H_px = 512  # pixels

    x = (H_px - coords["x"]) * cm_per_pixel
    y = (H_px - coords["y"]) * cm_per_pixel
    # x = coords["x"]
    # y = coords["y"]


    if frame_laser_on:
        ax.plot(
            x[0: frame_laser_on+int(0.3*125)],
            y[0: frame_laser_on+int(0.3*125)],
            marker=marker,
            color=color,
            linestyle="-",
            alpha=transparancy,
        )
    else : 
        ax.plot(
            x,
            y,
            marker=marker,
            color=color,
            linestyle="-",
            alpha=transparancy,
        )

    
    if frame_laser_on : 
        ax.plot(
            x[frame_laser_on: frame_laser_on+int(0.3*125)],
            y[frame_laser_on: frame_laser_on+int(0.3*125)],
            marker=marker,
            color="red",
            linestyle="-",
            alpha=0.7,
            label="Laser on"
        )

    

    return ax





def plot_3D_traj(coords: pd.DataFrame, 
                time : pd.Series,
                laser_on : float | None,
                ax: plt.axes,
                color: str,
                transparancy: float,
                y_invert: bool=False) -> plt.axes :
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    # --- trajectory ---
    ax.plot3D(
        coords["x"],
        coords["y"],
        time,
        color=color,
        marker=".",
        linestyle="-",
        alpha=transparancy,
    )

    # --- laser ON region (3D replacement for axvspan) ---
    if laser_on is not None:
        laser_off = laser_on + 0.3

        x_min, x_max = 0, 512
        y_min, y_max = 0, 512

        verts = [[
            (x_min, y_min, laser_on),
            (x_max, y_min, laser_on),
            (x_max, y_min, laser_off),
            (x_min, y_min, laser_off),
        ]]

        laser_plane = Poly3DCollection(
            verts,
            facecolor="red",
            alpha=0.3
        )
        ax.add_collection3d(laser_plane)

        # manual legend entry
        ax.legend(handles=[Patch(color="red", label="laser on")])

    # --- labels ---
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_zlabel("time (sec)")

    # --- limits ---
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    if y_invert:
        ax.invert_yaxis()

    return ax






def plot_metric_time(metric: pd.Series, 
                     time : pd.Series,
                     laser_on : float | None,
                     ax: plt.axes,
                     color: str,
                     transparancy: float=0.7) -> plt.axes : 
    """
    Plot the velocity or acceleration in time, for 1 condition (either NoStim, Beta or conti), for 1 clip
    The plot is highlighted between time 0.25 sec and 0.325 which correspond to
    the time range of a laser stimulus.
    No Highlight for NoStim condition

    Parameters
    ----------
    metric : pd.Series
        instantaneous velocity or acceleration for 1 clip  

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plotted velocities.
    """
    fps = 125

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, metric, color= color, alpha=transparancy, marker='.')
    if laser_on : 
        laser_off = laser_on +  0.3 # sec or 37.5 frame
        ax.axvspan(laser_on, laser_off, color='red', alpha=0.3, label="laser on")

        # add line for pad off 
        ax.axvline(laser_on - 0.025, color='k', lw=0.8, ls='--', label="pad off")
        ax.legend()

    return ax




def plot_animation(data: pd.Series, 
                time : pd.Series,
                laser_on : float | None,
                ax: plt.axes
                ) -> plt.axes :

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Velocity over time", pad=15)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Velocity")
    else:
        fig = ax.figure

    laser_off = laser_on + 0.3 if laser_on is not None else None
    line_base, = ax.plot([], [], lw=2, color="blue", label="Velocity")

    if laser_on : 
        ax.axvline(laser_on - 0.025, color='k', lw=0.8, ls='--', label="pad off")
        ax.axvspan(laser_on, laser_off, color='red', alpha=0.3, label="laser on")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    ax.set_ylim(min(data)-10, max(data)+10)
    ax.set_xlim(0, max(time) + 0.5)

    ax.legend()

    def animate(frame):
        x = time.iloc[:frame]
        y = data.iloc[:frame]

        line_base.set_data(x, y)
        return line_base

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(data),
        interval=5,
        repeat=False,
        blit=False
    )

    return anim
    
