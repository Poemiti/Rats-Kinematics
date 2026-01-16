from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import tqdm

def get_luminosity(annotation_num, video_path, fig_output_path, csv_ouput_path: Path, max_n_frames, label_studio_url, api_key):
    from label_studio_sdk import LabelStudio
    import pandas as pd, cv2
    import xarray as xr

    ls_client = LabelStudio(base_url=label_studio_url, api_key=api_key)
    data = ls_client.annotations.get(id=annotation_num).result
    led_info = {}
    for item in data:
        label = item["value"]["ellipselabels"][0]
        led_info[label] = {"x_per": item["value"]["x"], "y_per": item["value"]["y"], "radiusX_per": item["value"]["radiusX"], "radiusY_per": item["value"]["radiusY"]}
    leds = pd.DataFrame(led_info).T.to_xarray().rename(index="led_name")
    print(leds)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    image = xr.Dataset()
    image["y"] = xr.DataArray(np.arange(h), dims="y")
    image["x"] = xr.DataArray(np.arange(w), dims="x")
    image["mask"] = ((image["x"] - leds["x_per"]*w/100)**2/(leds["radiusX_per"]*w/100)**2 + (image["y"] - leds["y_per"]*h/100)**2/(leds["radiusY_per"]*h/100)**2) < 1
    print(image)

    if fig_output_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        fig = px.imshow(frame)
        image["color"] = xr.DataArray(["r", "g", "b", "a"], dims="color")
        image["mask_color"] = xr.DataArray([0, 200, 0, 0.5], dims="color")
        rgba_mask = image["mask"] * image["mask_color"]
        import plotly.graph_objects as go
        for i in range(rgba_mask.sizes["led_name"]):
            fig.add_trace(go.Image(z=rgba_mask.isel(led_name=i).transpose("y", "x", "color"), colormodel="rgba"))
        fig.write_html(fig_output_path)

    #Highly optimized code part, we convert everything to basic numpy and list, taking care of ordering
    n_leds = image.sizes["led_name"]
    mask_low_x = image["x"].where(image["mask"].any("y")).min("x").astype(int).to_numpy().tolist()
    mask_high_x = (image["x"].where(image["mask"].any("y")).max("x").astype(int).to_numpy()+1).tolist()
    mask_low_y = image["y"].where(image["mask"].any("x")).min("y").astype(int).to_numpy().tolist()
    mask_high_y = (image["y"].where(image["mask"].any("x")).max("y").astype(int).to_numpy()+1).tolist()
    cropped_masks = [image["mask"].isel(led_name=i).transpose("y", "x").to_numpy()[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]] for i in range(n_leds)]
    mask_low_x, mask_high_x, mask_low_y, mask_high_y

    cap = cv2.VideoCapture(video_path)
    luminosities = []

    if max_n_frames is None: 
        max_n_frames = num_frames
    else:
        max_n_frames = min(max_n_frames, num_frames)

    for i in tqdm.tqdm(range(max_n_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lum = [np.sum(np.where(cropped_masks[i], gray[mask_low_y[i]:mask_high_y[i], mask_low_x[i]:mask_high_x[i]], 0)) for i in range(n_leds)]
        luminosities.append(lum)

    cap.release()
    #End of highly optimized code part

    luminosities = xr.DataArray(luminosities, dims=["t", "led_name"], name="luminosity")
    luminosities["t"] = np.arange(luminosities.sizes["t"])/fps
    luminosities["t"].attrs["fs"] = fps
    luminosities = luminosities/image["mask"].sum(["y", "x"])

    # save as a csv file
    if csv_ouput_path is not None : 
        luminosity_df = luminosities.to_dataframe(name="luminosity").unstack("led_name")
        luminosity_df.to_csv(csv_ouput_path)

    return luminosities



def plot_bodyparts_trajectories(
    csv_path: Path,
    ax: plt.axes = None,
    bodyparts: list[str] | None = None,
    invert_y: bool = True,
    threshold: float = 0.5,
) -> plt.axes :
    """
    Plot bodypart trajectories from a DLC CSV onto a Matplotlib Axes.
    If ax is None, a new figure and axes are created (standalone use).
    """

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    # DLC CSV has 3 header rows (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    df = df.iloc[1:].reset_index(drop=True)

    all_bodyparts = df.columns.get_level_values(0).unique()
    if bodyparts is None:
        bodyparts = list(all_bodyparts)

    for bp in bodyparts:
        if bp not in all_bodyparts:
            continue

        xy = df[bp]
        mask = xy["likelihood"] >= threshold
        xy_filtered = xy[mask]

        ax.plot(
            xy_filtered["x"],
            xy_filtered["y"],
            marker="o",
            linestyle="-",
            label=bp,
            alpha=0.7,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(0, 512) # video dimension
    ax.set_ylim(0, 512)

    if invert_y:
        ax.invert_yaxis()

    return ax


def plot_stacked_trajectories(
        csv_dir: Path,
        output_fig_path: Path | None = None,
        bodyparts: list[str] | None = None,
        invert_y: bool = True,
        threshold: float = 0.5,
        show: bool = False,
    ) -> None:

    csv_paths = [p for p in Path(csv_dir).glob("*.csv") if p.is_file()]

    fig, ax = plt.subplots()

    for csv_path in csv_paths:
        plot_bodyparts_trajectories(
            csv_path=csv_path,
            ax=ax,
            bodyparts=bodyparts,
            invert_y=invert_y,
            threshold=threshold,
        )

    ax.set_title(f"Trajectories across frames of all trials")

    if output_fig_path:
        fig.savefig(output_fig_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_average_trajectories(csv_dir: Path,
                              output_fig_path: Path = None,
                              bodyparts : list[str] = None, 
                              invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    
    csv_path_list = [x for x in (csv_dir.glob("*.csv")) if x.is_file()]

    # --------------------------- 1. compute the average xy coord -----------------------------    
    all_coords = []

    for csv_path in csv_path_list : 
        # DLC CSV has 3 header rows (scorer, bodyparts, coords)
        df = pd.read_csv(csv_path, header=[0, 1, 2])

        # clean dataframe
        df.columns = df.columns.droplevel(0)  # remove scorer row
        df = df.iloc[1:].reset_index(drop=True)

        all_bodyparts = df.columns.get_level_values(0).unique()
        if bodyparts is None:
            bodyparts = list(all_bodyparts)

        for bp in bodyparts : 
            if bp not in all_bodyparts:
                continue

        xy = df[bp]
        mask = xy["likelihood"] >= threshold
        xy_filtered = xy[mask]

        print(xy_filtered)

    # --------------------------- 2. plot the average trajectory -----------------------------    

    pass


def annotate_single_bodypart(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    bodypart_name: str,
    radius=5,
    likelihood_threshold=0.8,
):
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    import numba

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


if __name__ == "__main__":
    
    # ---------------------------------------------- setup path -------------------------------------------------

    GENERATED_DATA_DIR = Path("../exploration/data") # root

    DATABASE_PATH = "../exploration/no_KO_video_list.csv"
    DATABASE = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(DATABASE.iloc[0]["filename"])

    INPUT_VIDEO_PATH = GENERATED_DATA_DIR / "direct_clips" / VIDEO_EXEMPLE.stem / "clip_00.mp4"
    BODYPART_POINTS_PATH = GENERATED_DATA_DIR / "csv_results" / VIDEO_EXEMPLE.stem
    OUTPUT_LUMINOSITY_PATH = GENERATED_DATA_DIR / "luminosity_figures" / VIDEO_EXEMPLE.stem
    OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "trajectory_figures" / VIDEO_EXEMPLE.stem

    OUTPUT_LUMINOSITY_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------- plot trajectory of bodypart -------------------------------------------------

    bodyparts_point = pd.read_csv(BODYPART_POINTS_PATH / "pred_results_clip_00.csv")
    print(bodyparts_point)
    
    bodyparts = ['elbow_l', 'elbow_r', 'finger_l_1', 
                 'finger_l_2', 'finger_l_3', 'finger_r_1', 'finger_r_2', 
                 'finger_r_3', 'left_hand', 'left_wrist', 'muzzle', 
                 'right_hand', 'right_wrist', 'shoulder_l', 'shoulder_r', 
                 'soft_pad_l', 'soft_pad_r']
    
    # clean dataframe from useless info
    bodyparts_point.columns = bodyparts_point.columns.droplevel(0) # remove scorer row
    bodyparts_point = bodyparts_point.iloc[1:]                     # remove num_frame row
    bodyparts_point = bodyparts_point.reset_index(drop=True)  
    print("\nclean dataframe :\n", bodyparts_point)

    plot_bodyparts_trajectories(csv_path= bodyparts_point / "pred_results_clip_00.csv", 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_{INPUT_VIDEO_PATH.stem}.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)
    

    plot_stacked_trajectories(csv_dir= bodyparts_point , 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_stacked.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)


    plot_average_trajectories(csv_dir= bodyparts_point, 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_average.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)

    # ---------------------------------------------- get luminosity info -------------------------------------------------

    luminosity = get_luminosity(annotation_num=1802,        
                                video_path= INPUT_VIDEO_PATH,
                                fig_output_path= OUTPUT_LUMINOSITY_PATH / f"luminosity_{INPUT_VIDEO_PATH.stem}.html",
                                csv_ouput_path = OUTPUT_LUMINOSITY_PATH / f"luminosity_{INPUT_VIDEO_PATH.stem}.csv",
                                max_n_frames=None,
                                label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
    )


    
    
