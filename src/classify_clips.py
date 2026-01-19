import numpy as np 
import os
import pandas as pd
from pathlib import Path
import re


def define_cue_type(luminosities) :
    cue_type = 'NoCue'
    time = 0

    for t, luminosity in enumerate(luminosities) :
        luminosity = float(luminosity) 

        if luminosity > 50 : 
            time += 1

        if luminosity > 50 and cue_type == 'NoCue' and time > 5:
            cue_type = "CueL1"
            time = 0
            continue

        if luminosity > 50 and cue_type == "L1" and time > 5 : 
            cue_type = "CueL2"
            break
        
    return cue_type


def define_stim_type(luminosities) : 
    stim_type = "NoStim"
    time = 0
    count = 0

    for t, luminosity in enumerate(luminosities) :
            luminosity = float(luminosity)  

            if luminosity > 50 : 
                time += 1

            if time == 5 and luminosity < 50 : 
                count += 1
                time = 0
                continue

            if time > 20 and count == 0 : 
                count += 1
                break

    if count > 5 : 
        stim_type = "Beta"
    elif count == 1 : 
        stim_type == "Conti"

    return stim_type


def define_reward(luminosities) : 
    pass



def rename_clip(clip_path: Path, new_stim: str, new_cue: str):
    STIM_PATTERN = r"(ContiMT300|NOstim|BetaMT300)"

    CUE_PATTERN = r"(" + "|".join([
        "onlyL1LeftHand", "onlyL2", "onlyL1",
        "L1L26040", "L1L25050", "L1L2",
        "L1-60", "L2-40",
        "L1", "L2"
    ]) + r")"


    name = clip_path.stem
    suffix = clip_path.suffix

    # Replace stim if present
    name, stim_count = re.subn(STIM_PATTERN, new_stim, name) # search and replace pattern
    name, cue_count = re.subn(CUE_PATTERN, new_cue, name)

    if stim_count == 0:
        print(f"[WARN] No stim token found in: {clip_path.name}")
    if cue_count == 0:
        print(f"[WARN] No cue token found in: {clip_path.name}")

    new_path = clip_path.with_name(name + suffix)

    if new_path.exists():
        raise FileExistsError(f"Target already exists: {new_path}")

    # clip_path.rename(new_path)

    print(f"\nRenamed {clip_path.name} into : \n\t{new_path.name}")




def classify_clip(clip_path: Path, luminosities) : 

    led_1 = luminosities["LED_1"]
    led_4 = luminosities["LED_4"]

    cue_type = define_cue_type(led_1)
    stim_type = define_stim_type(led_4)

    rename_clip(clip_path, stim_type, cue_type)


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

