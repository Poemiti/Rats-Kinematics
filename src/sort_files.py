import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from pathlib import Path


def is_video(filename : str) -> bool : 
    video_extensions = (".avi", ".mp4")
    return filename.lower().endswith(video_extensions)

def is_csv(filename : str) -> bool : 
    return filename.lower().endswith(".csv")

def sort_componants(components : list) -> pd.DataFrame :
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



def decompose_video_name(filename : str) : 

    # remove the format of the name (.avi, .mp4 ...)
    name, format = os.path.splitext(filename)
    componants = name.split("_")
    
    return componants


def dict_to_df(dict : dict) : 
    
     # Find the maximum length of the columns
    max_len = max(len(lst) for lst in dict.values())

    for key in dict:
        while len(dict[key]) < max_len:
            dict[key].append("") ## empty slots to match all columns size

    return pd.DataFrame(dict)


def extract_type(input : str, regex : str) -> str : 
    match = re.search(regex, input)

    if match : 
        return match.group(0)
    
    return None


def classify_file(filename: str, videos: list) -> None:
    
    name_comp = decompose_video_name(filename)

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
            "L1", "L2", "L1L2", "L1L26040", "L1L25050", "L1-60", "L2-40" 
        ]:
            result["task"] = token

        if result["laser_intensity"] == "Unknown":
            match = extract_type(token, r"\d,\d*(mW)")
            if match:
                result["laser_intensity"] = match

    videos.append({
        "filename": filename,
        **result
    })


def display_count_per_rat_condition(df : pd.DataFrame, 
                                    fig_output_path: Path, 
                                    condition : str, 
                                    show: bool = False) -> None :

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
    Display images in a grid using matplotlib.
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



def make_database(root_dir, satisfy_condition):
    sorted_videos = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if satisfy_condition(name) : 
                classify_file(os.path.join(root, name), sorted_videos)
    return pd.DataFrame(sorted_videos)




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
                comp = decompose_video_name(name)
                all_componants.update(comp)


    sorted_componants = sort_componants(all_componants)
    print(sorted_componants)


    # ----------------------------------------------  classify Video by condition -------------------------------------------------

    ct_video = 0
    sorted_videos = []
    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        for name in files:
            if is_video(name):
                classify_file(os.path.join(root, name), sorted_videos)
                ct_video += 1


    df = pd.DataFrame(sorted_videos)
    print(f"Nombre total de vidéo {ct_video}")
    print(f"Nombre de video collecté : {len(df)}")
    print(f"Final dataframe : \n{df}")


    # filtration of the KO rat
    no_KO_rats_df = df[df["rat_type"] != "Unknown"]
    no_KO_rats_df.to_csv(DATABASE_DIR / "no_KO_video_list.csv")

    display_count_per_rat_condition(no_KO_rats_df, 
                                    DATABASE_DIR / "video_count_per_rats.png",
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

