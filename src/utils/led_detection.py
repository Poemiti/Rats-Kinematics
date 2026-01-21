import numpy as np 
import os
import pandas as pd
from pathlib import Path
import re


def define_cue_type(luminosities) :
    cue_type = 'NoCue'
    time = 0
    cue_count = 0

    for t, luminosity in enumerate(luminosities) :
        luminosity = float(luminosity) 

        if luminosity > 50 : 
            time += 1

        if time > 5 : 
            cue_count += 1
            time = 0

    if cue_count == 1 :
        cue_type = "CueL1"

    elif cue_count == 2 : 
        cue_type = "CueL2"
        
    return cue_type


def is_led_on(luminosities) -> bool : 
    led_on = False
    time = 0

    for t, luminosity in enumerate(luminosities) :
            luminosity = float(luminosity)  

            if luminosity > 50 : 
                time += 1

    if time > 10 : 
        led_on = True

    return led_on



def rename_clip(clip_path: Path, laser_on: bool, new_cue: str):
    CUE_PATTERN = r"(" + "|".join([
        "onlyL1LeftHand", "onlyL2", "onlyL1",
        "L1L26040", "L1L25050", "L1L2",
        "L1-60", "L2-40",
        "L1", "L2"
    ]) + r")"


    name = clip_path.stem
    suffix = clip_path.suffix

    # Replace cue if present
    name, cue_count = re.subn(CUE_PATTERN, new_cue, name)
    if cue_count == 0:
        print(f"[WARN] No cue token found in: {clip_path.name}")

    if laser_on :
        name += "LaserOn"
    else : 
        name += "LaserOff"

    new_path = clip_path.with_name(name + suffix)

    if new_path.exists():
        raise FileExistsError(f"Target already exists: {new_path}")

    # clip_path.rename(new_path)

    print(f"\nRenamed {clip_path.name} into : \n\t{new_path.name}")



def classify_clip(clip_path: Path, luminosities) : 

    cue_type = define_cue_type(luminosities["LED_1"])
    led_on = is_led_on(luminosities["LED_4"])

    rename_clip(clip_path, led_on, cue_type)


if __name__ == "__main__" :

    # ---------------------------------------------- setup path -------------------------------------------------

    GENERATED_DATA_DIR = Path("../data") # root
    DATABASE_DIR = GENERATED_DATA_DIR / "database" 
    CLIPS_DIR = GENERATED_DATA_DIR / "clips"
    LUMINOSITY_DIR = GENERATED_DATA_DIR / "luminosity"

    df = pd.read_csv(LUMINOSITY_DIR / "#517" / "Rat_#517Ambidexter_20240624_BetaMT300_RightHemiCHR_L1L25050_C001H001S0002_clip_20_luminosity.csv")
    clip_path = CLIPS_DIR / "Rat_#517Ambidexter_20240624_BetaMT300_RightHemiCHR_L1L25050_C001H001S0002" / "Rat_#517Ambidexter_20240624_BetaMT300_RightHemiCHR_L1L25050_C001H001S0002_clip_20.mp4"
    # ---------------------------------------------- define setting  -------------------------------------------------

    # verify clip path is correct
    print(clip_path.exists())

    df.columns = df.iloc[0] # columns = LED_1 ...
    df = df.drop([0,1]).reset_index(drop=True)  # remove useless row

    classify_clip(clip_path, df)

