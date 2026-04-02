import pandas as pd
import re
from pathlib import Path

# ----------------------------------- basic utility --------------------------------------

def open_DLC_results(csv_path : Path) -> pd.DataFrame : 
    """
    Load and clean a DeepLabCut CSV file.

    DeepLabCut CSV files contain a three-level header
    (scorer, bodypart, coordinate). This function removes the scorer
    level and drops the first data row to return a clean DataFrame.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the DeepLabCut CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with a two-level column index
        (bodypart, coordinate).
    """

    # DLC CSV has 3 header rows (scorer, bodyparts, coords)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # clean dataframe
    df.columns = df.columns.droplevel(0)  # remove scorer row
    clean_df = df.iloc[1:].reset_index(drop=True)

    return clean_df

def is_video(filename : str) -> bool : 
    """
    Check if a file is a video file.
    Supported video formats are ``.avi`` and ``.mp4``.

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    bool
        True if the filename has a video extension, False otherwise.
    """
    video_extensions = (".avi", ".mp4")
    return filename.lower().endswith(video_extensions)

def is_csv(filename : str) -> bool : 
    """
    Check if a file is CSV

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    bool
        True if the filename ends with ``.csv``, False otherwise.
    """
    return filename.lower().endswith(".csv")


# ----------------------------------- function for classifiaction of file based on their name --------------------------------------


PATTERNS = {
    "rat_name": r"#\d{3}",
    "rat_type": r"(CTRL|CHR)",
    "condition": r"(Conti|NOstim|Beta)",
    "stim_location": r"(LeftHemi|RightHemi|Ipsi|ipsi|Bilateral|Contra)",
    "handedness": r"(Ambidexter|LeftHanded|RightHanded)",
    "session": r"S\d+",
    "view": r"H\d+",
    "laser_intensity": r"\d,\d*mW|\d+mW",
    "date": r"(\d{4}20\d{2}|20\d{6})",
    "clip": r"clip_(\d+)",
}

TASKS = ["onlyL1LeftHand", "onlyL2", "onlyL1", "onlyL2RightHand", "CueL2RightHand", 
        "L1", "L2", "L1L2", "L1L26040", "L1L25050", "L1-60", "L2-40",
        "NoCue", "CueL1", "CueL2"]


def parse_filename(name: str) -> dict:
    """
    Extract all metadata from filename once.
    """

    result = {key: "Unknown" for key in PATTERNS.keys()}
    result["task"] = "Unknown"

    # First pass: regex extraction
    for key, regex in PATTERNS.items():

        if result[key] == "Unknown"  :
            match = re.search(regex, name)
            if match:
                if key == "clip" : 
                    result[key] = match.group(1)
                    continue
                result[key] = match.group(0)

    # Task handling (not regex)
    for t in TASKS:
        if t in name.split("_"):
            result["task"] = t
            break

    # Second pass: derived defaults 
    if result["laser_intensity"] == "Unknown" :

        if result["condition"] == "Beta":
            result["laser_intensity"] = "1mW"
        elif result["condition"] == "Conti":
            result["laser_intensity"] = "0,5mW"
        elif result["condition"] == "NOstim":
            result["laser_intensity"] = "NOstim"

    return result



def classify_file(file_path: Path, videos: list) -> None:
    """
    Parse a video file_path and extract experimental metadata.

    The function decomposes the file_path into tokens that are then
    used to classify the file in certain categories.
    The extracted metadata is appended as a dictionary to `videos`.

    Parameters
    ----------
    file_path : pathlib.Path
        Full path or name of the video file.
    videos : list
        List to which the extracted metadata dictionary is appended.

    Returns
    -------
    None"""
    
    metadata = parse_filename(file_path.stem)

    if metadata["rat_type"] == "Unknown" : 
        metadata = parse_filename(file_path.parent.stem)

    metadata.pop("clip", None)
    metadata.pop("date", None)

    videos.append({
        "filename": str(file_path),
        **metadata
    })


    

def make_name_by_condition(name : str, laser_state: str) : 
    """
    Generate a name by extracting metadata tokens from a filename or a directory name.
    Note : This function is used when doing the analysis of multiple clips that
    come from seperate video, but have the same experimental setting. Especially
    regarding the cue 

    Parameters
    ----------
    name : str
        The filename/dir from which metadata will be extracted.

    Returns
    -------
    str
        A name composed of the extracted metadata fields.
    """
    meta = parse_filename(name)
    keys = ["rat_name", "rat_type", "condition", "stim_location", "view", "laser_intensity"]
    new_name = [meta[k] for k in keys if meta.get(k) not in ("Unknown", None)]
    new_name.append(laser_state)
    return "_".join(new_name)




def get_date(name: str):
    from datetime import datetime
    
    date_str = parse_filename(name)["date"]

    if not date_str:
        return None

    if date_str.startswith("20"):
        return datetime.strptime(date_str, "%Y%m%d")
    else:
        return datetime.strptime(date_str, "%d%m%Y")


def get_condition(name: str):
    condi = parse_filename(name)["condition"]
    return condi if condi else None


def get_clip_number(name: str):
    clip = parse_filename(name)["clip"]
    return int(clip) if clip else None


def get_session(name: str):
    return parse_filename(name)["session"]


def get_laser_intensity(name: str) : 
    return parse_filename(name)["laser_intensity"]




def is_left_view(filename : str) -> bool : 
    """
    Determine whether a video corresponds to the left camera view.
    H001 : left view
    H002 : right view

    Parameters
    ----------
    filename : str
        Name of the file to analyze.

    Returns
    -------
    bool
        True if the view identifier is ``H001`` (left view), False otherwise.
    """

    view  = parse_filename(filename)["view"]
    # print(f"view for {filename} : {view}")
    if view == "H001" : 
        return True
    return False

        
def verify_exist(path) : 
    if not path.exists() : 
        raise FileExistsError(f'This file does not exist : {path}')    



def make_database(root_dir : Path, satisfy_condition):
    """
    Build a database of video metadata from a directory tree.
    The directory is recursively scanned, and files satisfying
    a given condition are classified and stored in a DataFrame.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory to scan.
    satisfy_condition : callable
        Function that takes a filename and returns True if it
        should be included.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing extracted metadata for all matching files.
    """

    sorted_videos = []
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and satisfy_condition(file_path.name):
            classify_file(file_path, sorted_videos)
    return pd.DataFrame(sorted_videos)




if __name__ == "__main__" :

    print("no main")