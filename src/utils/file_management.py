import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from pathlib import Path

# ----------------------------------- function for classifiaction of file based on their name --------------------------------------

def open_clean_csv(csv_path : Path) -> pd.DataFrame : 
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

    view  = extract_type(filename, r"H\d+")
    print(f"view for {filename} : {view}")
    if view == "H001" : 
        return True
    return False


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

def sort_componants(components : list[str]) -> pd.DataFrame :
    """
    Components are classified into numeric, alphabetic,
    alphanumeric, or mixed categories and returned as a DataFrame.
    Note : used to check how files are named

    Parameters
    ----------
    components : list of str
        List of string components to categorize.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns corresponding to component categories.
    """
        
    categorized = {
        "number": [],
        "alpha": [],
        "alphanumeric": [],
        "special": [],
        "mixed": []
    }

    for comp in components:

        if comp.isdigit():
            categorized["number"].append(comp)

        elif comp.isalpha():
            categorized["alpha"].append(comp)

        elif comp.isalnum():
            categorized["alphanumeric"].append(comp)

        else:
            categorized["mixed"].append(comp)

    df = dict_to_df(categorized)

    return df



def decompose_filename(filename : str) -> list[str] : 
    """
    Decompose a video filename into underscore-separated components.
    The file extension is removed before splitting.

    Parameters
    ----------
    filename : str
        Video filename (with extension).

    Returns
    -------
    list of str
        List of components extracted from the filename.
    """

    # remove the format of the name (.avi, .mp4 ...)
    name, format = os.path.splitext(filename)
    componants = name.split("_")
    
    return componants


def dict_to_df(dict : dict) -> pd.DataFrame : 
    """
    Convert a dictionary of lists into a pandas DataFrame.
    Lists are padded with empty strings so that all columns
    have the same length.

    Parameters
    ----------
    dict : dict
        Dictionary mapping column names to lists of values.

    Returns
    -------
    pandas.DataFrame
        DataFrame constructed from the dictionary.
    """
        
     # Find the maximum length of the columns
    max_len = max(len(lst) for lst in dict.values())

    for key in dict:
        while len(dict[key]) < max_len:
            dict[key].append("") ## empty slots to match all columns size

    return pd.DataFrame(dict)


def extract_type(input : str, regex : str) -> str : 
    """
    Extract a substring from a string using a regular expression.

    Parameters
    ----------
    input : str
        Input string to search.
    regex : str
        Regular expression pattern.

    Returns
    -------
    str or None
        The matched substring if found, otherwise None.
    """

    match = re.search(regex, input)

    if match : 
        return match.group(0)
    
    return None


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
    
    name_comp = decompose_filename(file_path.stem)

    result = {
        "rat_name": "Unknown",
        "rat_type": "Unknown",
        "condition": "Unknown",
        "stim_location": "Unknown",
        "task": "Unknown",
        "handedness": "Unknown",
        "session": "Unknown",
        "view": "Unknown",
        "laser_intensity": "Unknown",
        "laser_on" : "Unknown"
    }

    for token in name_comp:

        if result["rat_name"] == "Unknown" :
            match = extract_type(token, r"^#(\d\d\d)")
            if match:
                result["rat_name"] = match

        if result["rat_type"] == "Unknown":
            match = extract_type(token, r"(CTRL|CHR)")
            if match:
                result["rat_type"] = match

        if result["condition"] == "Unknown":
            match = extract_type(token, r"(Conti|NOstim|Beta)")
            if match:
                result["condition"] = match

        if result["stim_location"] == "Unknown":
            match = extract_type(token, r"(LeftHemi|RightHemi|Ipsi|ipsi|Bilateral|Contra)")
            if match:
                result["stim_location"] = match

        if result["handedness"] == "Unknown":
            match = extract_type(token, r"(Ambidexter|LeftHanded|RightHanded)")
            if match:
                result["handedness"] = match

        if result["session"] == "Unknown":
            match = extract_type(token, r"S\d+")
            if match:
                result["session"] = match

        if result["view"] == "Unknown":
            match = extract_type(token, r"H\d+")
            if match:
                result["view"] = match

        if result["task"] == "Unknown" and token in [
            "onlyL1LeftHand", "onlyL2", "onlyL1",
            "L1", "L2", "L1L2", "L1L26040", "L1L25050", "L1-60", "L2-40",
            "CueL1", "CueL2" # those 2 are for renamed clip (led_detection.py) that tell exactly which cue is on
        ]:
            result["task"] = token

            # if "CueL1" in name_comp or "CueL2" in name_comp : 
            #     print(name_comp)
            #     print(result["task"], token)

        if result["laser_intensity"] == "Unknown":
            match = extract_type(token, r"\d,\d*(mW)")
            if match:
                result["laser_intensity"] = match

        # this will work only for the renamed clips (led_detection.py)
        if result["laser_on"] == "Unknown" :
            match = extract_type(token, r'(LaserOn|LaserOff)')
            if match : 
                result["laser_on"] = match

    videos.append({
        "filename": str(file_path),
        **result
    })


def display_count_per_rat_condition(df : pd.DataFrame, 
                                    fig_output_path: Path, 
                                    condition : str, 
                                    show: bool = False) -> None :
    """
    Plot the number of videos per rat type and experimental condition.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing video metadata.
    fig_output_path : pathlib.Path
        Path where the figure will be saved.
    condition : str
        Column name representing the experimental condition.
    show : bool, optional
        Whether to display the figure interactively.

    Returns
    -------
    None
    """

    counts = (
        df.groupby(["rat_type", condition])
        .size()
        .reset_index(name="count")
    )


    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=counts,
        x="rat_type",
        y="count",
        hue=condition
    )
    
    for container in ax.containers:
        ax.bar_label(container)

    plt.xlabel("Rat type")
    plt.ylabel("Number of videos")
    plt.title(f"Video count by rat type and {condition}")
    plt.tight_layout()

    if fig_output_path :
        plt.savefig(fig_output_path)

    if show:
        plt.show()


def display_images(
                images_list: list[str],
                titles_list: list[str] | None,
                fig_output_path: Path,
                figsize: tuple[float, float] = (12, 4),
                images_per_row: int = 3,
                show: bool = True
            ) -> None:
    """
    Display a list of images in a grid layout.

    Parameters
    ----------
    images_list : list of str
        Paths to image files.
    titles_list : list of str or None
        Optional list of titles for each image.
    fig_output_path : pathlib.Path
        Path where the figure will be saved.
    figsize : tuple of float, optional
        Size of the figure in inches.
    images_per_row : int, optional
        Number of images per row.
    show : bool, optional
        Whether to display the figure interactively.

    Returns
    -------
    None
    """

    import math

    n = len(images_list)
    if n == 0:
        raise ValueError("The image list is empty.")

    rows = math.ceil(n / images_per_row)

    fig, axes = plt.subplots(
        rows,
        images_per_row,
        figsize=(figsize[0], figsize[1])
    )

    # Normalize axes to a flat list
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = plt.imread(images_list[i])
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if titles_list and i < len(titles_list):
                ax.set_title(titles_list[i])
        else:
            ax.axis("off")  # Hide unused axes

    #  vertical spacing between rows
    plt.subplots_adjust(
        hspace=0.15,  
        wspace=0.05
    )

    plt.tight_layout()

    if fig_output_path :
        plt.savefig(fig_output_path)

    if show:
        plt.show()



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



def make_directory_name(name : str) : 
    """
    Generate a directory name by extracting metadata tokens from a filename or a directory name.
    Note : This function is used when doing the analysis of multiple clips that
    come from seperate video, but have the same experimental setting. Especially
    regarding the cue 

    The function parses the name using regular expressions to extract:
        - rat identifier (e.g. '#517')
        - rat type (CTRL or CHR)
        - stimulation location (LeftHemi, RightHemi, Ipsi, Bilateral, Contra)
        - laser stim type (Conti, NOstim, Beta)
        - cue type (CueL1, CueL2, NoCue)
        - view identifier ('H001', 'H002')
        - laser state (LaserOn or LaserOff)

    Extracted components are concatenated using underscores ('_').
    Any empty or missing components are omitted.

    Parameters
    ----------
    name : str
        The filename/dir from which metadata will be extracted.

    Returns
    -------
    str
        A directory name composed of the extracted metadata fields.
    """
    name_comp = name.split("_")

    result = {
        "rat_name": "Unknown",
        "rat_type": "Unknown",
        "condition": "Unknown",
        "stim_location": "Unknown",
        "cue_type": "Unknown",
        "view": "Unknown",
        "laser_on" : "Unknown",
        "laser_intensity": "Unknown"
    }

    for token in name_comp:

        if result["rat_name"] == "Unknown" :
            match = extract_type(token, r"^#(\d\d\d)")
            if match:
                result["rat_name"] = match

        if result["rat_type"] == "Unknown":
            match = extract_type(token, r"(CTRL|CHR)")
            if match:
                result["rat_type"] = match

        if result["condition"] == "Unknown":
            match = extract_type(token, r"(Conti|NOstim|Beta)")
            if match:
                result["condition"] = match

        if result["stim_location"] == "Unknown":
            match = extract_type(token, r"(LeftHemi|RightHemi|Ipsi|ipsi|Bilateral|Contra)")
            if match:
                result["stim_location"] = match

        if result["view"] == "Unknown":
            match = extract_type(token, r"H\d+")
            if match:
                result["view"] = match

        if result["cue_type"] == "Unknown" :
            match = extract_type(token, r"(CueL1|CueL2)")
            if match:
                result["cue_type"] = match

        if result["laser_intensity"] == "Unknown":
            match = extract_type(token, r"\d,\d*(mW)")
            if match:
                result["laser_intensity"] = match

        # this will work only for the renamed clips (led_detection.py)
        if result["laser_on"] == "Unknown" :
            match = extract_type(token, r'(LaserOn|LaserOff)')
            if match : 
                result["laser_on"] = match

    new_name = [val for val in result.values() if val != "Unknown"]
    print(f"old name : {name}")
    print(f"new name : {new_name}")

    return "_".join(new_name)

    
        
def verify_exist(path) : 
    if not path.exists() : 
        raise FileExistsError(f'This file does not exist : {path}')



if __name__ == "__main__" :

    # ---------------------------------------------- setup path -------------------------------------------------

    GENERATED_DATA_DIR = Path("../exploration/data") # root
    RAW_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")

    DATABASE_DIR = GENERATED_DATA_DIR / "database" 
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------- show all componants -------------------------------------------------


    all_componants = set()

    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        for name in files : 
            if is_video(name) :
                comp = decompose_filename(name)
                all_componants.update(comp)


    sorted_componants = sort_componants(all_componants)
    print(sorted_componants)


    # ----------------------------------------------  classify Video by condition -------------------------------------------------

    df = make_database(RAW_VIDEO_DIR, is_video)

    print(f"Number of video in database : {len(df)}")
    print(f"Database : \n{df}")

    # filtration of the KO rat
    no_KO_rats_df = df[df["rat_type"] != "Unknown"]
    no_KO_rats_df.to_csv(DATABASE_DIR / "no_KO_video_list.csv")

    print(f"Number of video after KO filtration : {len(no_KO_rats_df)}")

    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR /"video_count_per_rats.png",
                                    "rat_name")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_condition.png",
                                    "condition")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_task.png",
                                     "task")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_stim_location.png",
                                    "stim_location")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_handedness.png",
                                    "handedness")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_session.png",
                                    "session")
    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_view.png",
                                    "view")


    img_list = [
        DATABASE_DIR / "video_count_per_condition.png",
        DATABASE_DIR / "video_count_per_task.png", 
        DATABASE_DIR / "video_count_per_stim_location.png", 
        DATABASE_DIR / "video_count_per_handedness.png", 
        DATABASE_DIR / "video_count_per_session.png",  
        DATABASE_DIR / "video_count_per_rats.png" 
    ]


    display_images(images_list= img_list, 
                   titles_list=None,
                   fig_output_path= DATABASE_DIR / "Overall_database.png",
                   figsize=(15, 8), show=False)

