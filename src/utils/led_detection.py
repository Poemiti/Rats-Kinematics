import numpy as np 
import pandas as pd
from pathlib import Path
import re
import cv2
import xarray as xr
import plotly.express as px
import tqdm

from utils.file_management import is_left_view


def get_luminosity(annotation_num, video_path, fig_output_path, csv_ouput_path: Path, max_n_frames, label_studio_url, api_key):
    from label_studio_sdk import LabelStudio

    ls_client = LabelStudio(base_url=label_studio_url, api_key=api_key)
    data = ls_client.annotations.get(id=annotation_num).result
    led_info = {}
    for item in data:
        label = item["value"]["ellipselabels"][0]
        led_info[label] = {"x_per": item["value"]["x"], "y_per": item["value"]["y"], "radiusX_per": item["value"]["radiusX"], "radiusY_per": item["value"]["radiusY"]}
    leds = pd.DataFrame(led_info).T.to_xarray().rename(index="led_name")
    # print(leds)

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
    # print(image)

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




def define_cue_type(luminosities) :
    """
    Determine the cue type based on LED luminosity over time.
    Note : Must be applied only on LED_1

    The function analyzes a sequence of luminosity values and counts
    the number of sustained light activations. A cue is detected when
    luminosity exceeds a fixed threshold (50) for more than a given number
    of consecutive frames (5 frames).

    Cue classification:
    - 1 activation  -> ``CueL1``
    - 2 activations -> ``CueL2``
    - otherwise     -> ``NoCue``

    Parameters
    ----------
    luminosities : array like
        Array luminosity values (e.g., from get_luminosity())

    Returns
    -------
    str
        Detected cue type: ``"CueL1"``, ``"CueL2"``, or ``"NoCue"``.
    """

    cue_type = 'NoCue'
    time = 0
    cue_count = 0

    for t, luminosity in enumerate(luminosities) :
        luminosity = float(luminosity) 

        if luminosity >= 50 : 
            time += 1

        if luminosity < 50 and time > 5 : 
            cue_count += 1
            time = 0

    if cue_count == 1 :
        cue_type = "CueL1"

    elif cue_count == 2 : 
        cue_type = "CueL2"

    print(f"\ncue count : {cue_count}")
        
    return cue_type


def is_led_on(luminosities) -> bool : 
    """
    Determine if a LED is active during a clip.

    The function counts the total number of frames for which the LED
    luminosity exceeds a fixed threshold (50). If this count exceeds a
    minimum duration (5 frames), the LED is considered ON.

    Parameters
    ----------
    luminosities : iterable
        Sequence of luminosity values (e.g., LED intensity measurements).

    Returns
    -------
    bool
        True if the LED is considered ON, False otherwise.
    """

    led_on = False
    time = 0

    for t, luminosity in enumerate(luminosities) :
            luminosity = float(luminosity)  

            if luminosity > 50 : 
                time += 1

    if time > 10 : 
        led_on = True

    return led_on



def rename_file(file_path: Path, laser_on: bool, new_cue: str):
    """
    Rename a video clip based on detected cue and laser state.

    The function replaces any existing cue token in the filename
    with a new cue label and appends a laser status suffix
    (``LaserOn`` or ``LaserOff``).

    Parameters
    ----------
    clip_path : pathlib.Path
        Path to the video clip to rename.
    laser_on : bool
        Whether the laser was active during the clip.
    new_cue : str
        Cue label to insert into the filename (e.g., ``CueL1``, ``CueL2``).

    Returns
    -------
    None
    """

    CUE_PATTERN = r"(" + "|".join([
        "onlyL1LeftHand", "onlyL2", "onlyL1",
        "L1L26040", "L1L25050", "L1L2",
        "L1-60", "L2-40",
        "L1", "L2"
    ]) + r")"


    name = file_path.stem
    suffix = file_path.suffix

    # Replace cue if present
    name, cue_count = re.subn(CUE_PATTERN, new_cue, name)
    if cue_count == 0:
        print(f"[WARN] No cue token found in: {file_path.name}")

    if laser_on :
        name += "_LaserOn"
    else : 
        name += "_LaserOff"

    new_path = file_path.with_name(name + suffix)

    if new_path.exists():
        raise FileExistsError(f"Target already exists: {new_path}")

    # /!\ WARNING this line rename file, must uncomment manually 
    file_path.rename(new_path)

    print(f"\nRenamed {file_path.name} into : \n\t{new_path.name}")





if __name__ == "__main__" :

    # ---------------------------------------------- setup path -------------------------------------------------

    # inputs (should exist)
    GENERATED_DATA_DIR = Path("../exploration/data")
    DATABASE_PATH = GENERATED_DATA_DIR / "database/rat_517_H001.csv"  # if it does not exist, make one with make_database (in file_management.py)

    # get the path for a video (path in a premade database)
    database = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(database.iloc[0]["filename"])
    INPUT_VIDEO_PATH = GENERATED_DATA_DIR / "clips" / VIDEO_EXEMPLE.stem / f"{VIDEO_EXEMPLE.stem}_clip_00.mp4"

    # outputs
    OUTPUT_LUMINOSITY_DIR = GENERATED_DATA_DIR / "luminosity" /  VIDEO_EXEMPLE.stem 
    OUTPUT_LUMINOSITY_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------- get luminosisty -------------------------------------------------

    output_path = OUTPUT_LUMINOSITY_DIR / f"luminosity_{VIDEO_EXEMPLE.stem}"

    # verify if the clip is left or right view
    if is_left_view(str(VIDEO_EXEMPLE.stem)) : 
        label_studio_annotation = 1812
    else : 
        label_studio_annotation = 1811
    print(f"\nlabel studio annotation : {label_studio_annotation}")
    

    luminosity = get_luminosity(annotation_num=label_studio_annotation,        
                                video_path= INPUT_VIDEO_PATH,
                                fig_output_path= f"{output_path}.html",
                                csv_ouput_path = f"{output_path}.csv",
                                max_n_frames=None,
                                label_studio_url= "http://l-t4-mamserver.imn.u-bordeaux2.fr/labelstudioapp",
                                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3NTE1MDkzNCwiaWF0IjoxNzY3OTUwOTM0LCJqdGkiOiI4OGEwYTE5NDZkODM0NTlhYjQyMzIzN2I1MTQ0N2ZlYiIsInVzZXJfaWQiOiIyNCJ9.dNTu0zJNPHax5tnfYWanvZlH8SZ9VHQvOGZ_GEyN0l8"
    )

    print(f"\nNumber of frame : {len(luminosity)}")
    
    # ---------------------------------------------- classify clip depending on led  -------------------------------------------------

    luminosity.columns = luminosity.columns.droplevel(0) # columns = LED_1 ...
    luminosity = luminosity.drop([1]).reset_index(drop=True)  # remove useless row

    print(luminosity)
