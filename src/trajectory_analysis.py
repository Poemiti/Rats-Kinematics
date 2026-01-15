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



def plot_bodyparts_trajectories(csv_path: Path,
                                output_fig_path: Path = None,
                                bodyparts : list[str] = None, 
                                invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    # DLC CSV has 3 header rows
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe from useless info
    df.columns = df.columns.droplevel(0) # remove scorer row
    df = df.iloc[1:]                     # remove num_frame row
    df = df.reset_index(drop=True)  
    print("\nclean dataframe :\n", df)

    # Get body parts automatically
    all_bodyparts = df.columns.get_level_values(0).unique()

    if bodyparts is None:
        bodyparts = list(all_bodyparts)

    print(f"\nbodypart : {bodyparts}")

    plt.figure()

    for bp in bodyparts:
        if bp not in bodyparts:
            continue

    
        # Select x,y only (ignore likelihood)
        xy = df[bp]
        print(f"\nbodypart : {bp}, coord : \n{xy}")
        
        # Create mask for points above threshold
        mask = xy["likelihood"] >= threshold

        # Apply mask
        xy_filtered = xy[mask]
        print(f"filtered coor : \n{xy_filtered}")

        plt.plot(
            xy_filtered["x"],
            xy_filtered["y"],
            marker="o",
            linestyle="-",
            label=bp
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(ncol=2, fontsize=8)

    if invert_y:
        plt.gca().invert_yaxis()

    if output_fig_path: 
        plt.title("Body part trajectories across frames of one trial")
        plt.savefig(output_fig_path)

    if show : 
        plt.show()



def plot_stacked_trajectories(csv_dir: Path,
                              output_fig_path: Path = None,
                              bodyparts : list[str] = None, 
                              invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    
    csv_path_list = [x for x in (Path(csv_dir).glob("*.csv")) if x.is_file()]
    print(f"\n CSV LIST : {csv_path_list}\n")

    for csv_path in csv_path_list : 
        plot_bodyparts_trajectories(csv_path=csv_path,
                                    output_fig_path= None,  # we don't save individual fig
                                    bodyparts=bodyparts,
                                    show=False,             # we don"t want to show each individual fig
                                    invert_y=invert_y, threshold=threshold
                                    )
        
    if output_fig_path: 
        plt.title("Body part trajectories across frames of all the trial of this video")
        plt.savefig(output_fig_path)

    if show : 
        plt.show()

    pass


def plot_average_trajectories(csv_dir: Path,
                              output_fig_path: Path = None,
                              bodyparts : list[str] = None, 
                              invert_y: bool=True, show: bool = False, threshold: int = 0.5) -> None:
    
    csv_path_list = [x for x in (Path(csv_dir).glob("*.csv")) if x.is_file()]
    
    for csv_path in csv_path_list : 
        pass
    pass


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


    plot_bodyparts_trajectories(csv_path= BODYPART_POINTS_PATH / "pred_results_clip_00.csv", 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_{INPUT_VIDEO_PATH.stem}.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)
    

    plot_stacked_trajectories(csv_dir= BODYPART_POINTS_PATH , 
                                output_fig_path= OUTPUT_TRAJECTORY_PATH / f"trajectory_stacked.png",
                                bodyparts= ["left_hand"], 
                                invert_y=True,
                                show=False,
                                threshold=0.5)


    plot_average_trajectories(csv_dir= BODYPART_POINTS_PATH, 
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


    
    
