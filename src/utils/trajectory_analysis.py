from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import tqdm
import math

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

    luminosity_df = luminosities.to_dataframe(name="luminosity").unstack("led_name")

    # save as a csv file
    if csv_ouput_path is not None : 
        luminosity_df.to_csv(csv_ouput_path)

    return luminosity_df


def open_clean_csv(csv_path : Path) -> pd.DataFrame : 
    # DLC CSV has 3 header rows (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    clea_df = df.iloc[1:].reset_index(drop=True)

    return clea_df


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

    df = open_clean_csv(csv_path)

    all_bodyparts = df.columns.get_level_values(0).unique()
    if bodyparts is None:
        bodyparts = list(all_bodyparts)

    for bp in bodyparts:
        if bp not in all_bodyparts:
            continue

        xy = df[bp]
        mask = xy["likelihood"] >= threshold
        xy_filtered = xy[mask]
        start, end = define_StartEnd_of_trajectory(xy_filtered)
        xy_filtered = xy_filtered.iloc[start : end]

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

    fig, ax = plt.subplots()

    for csv_path in csv_dir.glob("*.csv"):
        plot_bodyparts_trajectories(
            csv_path=csv_path,
            ax=ax,
            bodyparts=bodyparts,
            invert_y=invert_y,
            threshold=threshold,
        )

    ax.set_title(f"Trajectories across frames of all trials\n{csv_dir.stem}")

    if output_fig_path:
        fig.savefig(output_fig_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_average_trajectories(csv_dir: Path,
                              output_fig_path: Path = None,
                              bodyparts : list[str] = None, 
                              invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    

    # --------------------------- 1. compute the average xy coord -----------------------------    
    all_coords = []

    for csv_path in csv_dir.glob("*.csv") : 
        df = open_clean_csv(csv_path)

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




def define_StartEnd_of_trajectory(coords : pd.DataFrame) : 
    """
    return the start time and end time of the trajectory
    a trajectory is considered to go from the pad to the trigger
    """
    lever_position = 210
    crossed = False
    t_start = 0
    print(t_start)
    t_end = len(coords)

    for t, row in coords.iterrows():
        if t == 0 : 
            print(f"not crossed, y = {row['y']}, t = {t}")

        if row["y"] < lever_position and not crossed:
            print(f"crossed, y = {row['y']}, t = {t}")
            crossed = True
            continue

        if row["y"] > lever_position and crossed:
            print(f"crossed again, y = {row['y']}, t = {t}")
            t_end = t-1
            break

    return t_start, t_end


def get_instantaneous_velocity(coords: pd.DataFrame) -> pd.Series:
    fps = 125
    diffs = coords.diff().dropna() # compute the difference between 2 row (the actual vs the previous)
    displacement = diffs.pow(2).sum(axis=1).pow(0.5)    # compute the power(2) of each columns + the sqrt (pow(0.5))
    print(displacement)

    return displacement * fps  # pixels / second


def get_velocity(coords: pd.DataFrame) -> float : 
    fps = 125
    diffs = coords.diff().dropna()
    distance = (diffs.pow(2).sum(axis=1).pow(0.5)).sum()

    duration_sec = len(coords) / fps
    return distance / duration_sec  # pixel / sec


def get_distance(coords: pd.DataFrame) :
    distance = 0
    diffs = coords.diff().dropna()
    distance = (diffs.pow(2).sum(axis=1).pow(0.5)).sum()
    return distance


def compute_metric(csv_path: Path,
                    bodyparts : str,
                    metric,
                    threshold : float = 0.5) -> float : 
    """
    compute the velocity during 1 clip from the dlc prediction point 
    (stored in csv)
    """
    # get data from the bodypart
    df = open_clean_csv(csv_path)
    xy = df[bodyparts]
    xy = xy[xy["likelihood"] >= threshold]
    print(f"filtered coord : \n {xy}")

    # filter to get only the trajectory we want
    t_start, t_end = define_StartEnd_of_trajectory(xy)
    true_coords = xy.iloc[t_start : t_end].reset_index(drop=True)
    print(f"coords after threshold : {len(xy)}, coords after finding traj : {len(true_coords)}")

    if len(true_coords) <= 1 : 
        print("ERROR : no movement have been found in this clip")
        return 0

    return metric(true_coords[["x", "y"]])



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

    # ---------------------------------------------- plot single trajectory of bodypart -------------------------------------------------
    
    THRESHOLD = 0.5
    BODYPART = ["left_hand"]

    fig, ax = plt.subplots()

    for csv_path in BODYPART_POINTS_PATH.iterdir() :

        fig, ax = plt.subplots() 

        output_fig_dir = OUTPUT_TRAJECTORY_PATH / "single_trajectory"
        output_fig_dir.mkdir(parents=True, exist_ok=True)
        output_fig_path = output_fig_dir / csv_path.stem

        plot_bodyparts_trajectories(
            csv_path=Path(csv_path),
            ax=ax,
            bodyparts=BODYPART,
            invert_y=True,
            threshold=THRESHOLD,
        )

        ax.set_title(f"Trajectory of {csv_path.stem}, threshold 0.5,\nL1, NoStim, Successful")
        # plt.show()
        fig.savefig(output_fig_path)   

        plt.close(fig) 

    # ---------------------------------------------- annotate single trajectory of bodypart -------------------------------------------------

    

    # ---------------------------------------------- plot stacked + average trajectory of bodypart -------------------------------------------------


    # plot_stacked_trajectories(csv_dir= BODYPART_POINTS_PATH , 
    #                             output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_stacked.png",
    #                             bodyparts= ["left_hand"], 
    #                             invert_y=True,
    #                             show=False,
    #                             threshold=0.5)


    # plot_average_trajectories(csv_dir= BODYPART_POINTS_PATH, 
    #                             output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_average.png",
    #                             bodyparts= ["left_hand"], 
    #                             invert_y=True,
    #                             show=False,
    #                             threshold=0.5)

    # # ---------------------------------------------- get luminosity info -------------------------------------------------

    # luminosity = get_luminosity(annotation_num=1802,        
    #                             video_path= INPUT_VIDEO_PATH,
    #                             fig_output_path= OUTPUT_LUMINOSITY_PATH / f"luminosity_{INPUT_VIDEO_PATH.stem}.html",
    #                             csv_ouput_path = OUTPUT_LUMINOSITY_PATH / f"luminosity_{INPUT_VIDEO_PATH.stem}.csv",
    #                             max_n_frames=None,
    #                             label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
    #                             api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
    # )

    # ---------------------------------------------- compute metrics -------------------------------------------------

    print(f"\nComputing metric of {BODYPART_POINTS_PATH / 'pred_results_clip_00.csv'} :")

    # # distance
    # distance = compute_metric(csv_path=BODYPART_POINTS_PATH / "pred_results_clip_00.csv",
    #                           bodyparts="left_hand",
    #                           metric=get_distance)
    
    # # velocity
    # velocity = compute_metric(csv_path=BODYPART_POINTS_PATH / "pred_results_clip_00.csv",
    #                           bodyparts="left_hand",
    #                           metric=get_velocity)

    # instantaneous velocity
    instant_velocity = compute_metric(csv_path=BODYPART_POINTS_PATH / "pred_results_clip_00.csv",
                              bodyparts="left_hand",
                              metric=get_instantaneous_velocity)
    

    # print(f"  distance = {distance:.02f}")
    # print(f"  velocity = {velocity:.02f}")
    print(f"  instaneous velocity = {instant_velocity}")