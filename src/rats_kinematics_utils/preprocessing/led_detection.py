import numpy as np 
import pandas as pd
from pathlib import Path
import re
import cv2
import xarray as xr
import plotly.express as px
import tqdm
import operator



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




def define_cue_type(luminosities: pd.Series, threshold=100, min_duration=5) -> str :
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

        if luminosity >= threshold : 
            time += 1

        if luminosity < threshold and time > min_duration : 
            cue_count += 1
            time = 0

    if cue_count == 1 :
        cue_type = "CueL1"

    elif cue_count == 2 : 
        cue_type = "CueL2"

    # print(f"\ncue count : {cue_count}")
        
    return cue_type


def led_state(luminosities: pd.Series,
              threshold: float = 100,
              min_duration: int = 10,
              comparator: operator = operator.lt,) -> tuple[bool, int]:
    
    consecutive = 0
    start_index = None

    for t, value in enumerate(luminosities):
        value = float(value)

        if comparator(value, threshold):
            if consecutive == 0:
                start_index = t
            consecutive += 1

            if consecutive > min_duration:
                return True, start_index
        else:
            consecutive = 0
            start_index = None

    return False, None



def get_time_led_state(
                    luminosity_path: Path, 
                    luminosities: pd.DataFrame = None,
                    LED: str = "LED_3", 
                    state: str = "ON",
                    min_duration: int = 5,
                    in_sec: bool = False, 
                    fps: int =125) -> float | int :

    if luminosities is None : 
        luminosities = pd.read_csv(luminosity_path)
        luminosities.columns = luminosities.iloc[0]      # use first row as column names
        luminosities = luminosities.drop(0).reset_index(drop=True) # remove useless row
        luminosities = luminosities[luminosities.iloc[:, 0] != 't']

    if state == "ON" : 
        _, first_frame = led_state(luminosities[LED], min_duration=min_duration, comparator=operator.gt)  # gt: greater than = ON
    else : 
        _, first_frame = led_state(luminosities[LED], min_duration=min_duration, comparator=operator.lt)  # lt: less than = OFF
    
    if first_frame and in_sec : 
        first_time = first_frame / fps  
        # print(f"Laser one at {time_laser_off} sec, {frame_laser_off} frame")
    else : 
        first_time = first_frame
    
    return first_time












if __name__ == "__main__" :

    print("No main")