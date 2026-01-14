from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# def plot_bodyparts_trajectories(csv_path, bodyparts : list[str] = None, invert_y=True) -> None:
#     """
#     Plot x,y trajectories of body parts across frames.
#     """

#     # Load CSV with multi-index header
#     df = pd.read_csv(csv_path, header=[0, 1])

#     # Infer body parts automatically if not provided
#     if bodyparts is None:
#         bodyparts = df.iloc[0].unique()

#     print(f"\nbody part list : {bodyparts}\n")

#     plt.figure()

#     for bp in bodyparts:
#         # if bp in df.iloc[0]:
#         #     continue

#         xy = df[bp][["x", "y"]]
#         print(f"bodypart : {bp}, coor : {xy}")
#         plt.plot(xy["x"], xy["y"], marker="o", label=bp)

#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("Body part trajectories across frames")
#     plt.legend()

#     if invert_y:
#         plt.gca().invert_yaxis()

#     plt.show()

def plot_bodyparts_trajectories(csv_path, bodyparts : list[str] = None, invert_y: bool=True, threshold: int = 0.5) -> None:
# DLC CSV has 3 header rows
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Drop scorer level → keep (bodypart, coord)
    df.columns = df.columns.droplevel(0)

    # Get body parts automatically
    all_bodyparts = df.columns.get_level_values(0).unique()
    print(list(all_bodyparts))

    if bodyparts is None:
        bodyparts = list(all_bodyparts)
        bodyparts.remove("bodyparts")

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
            linestyle="",
            label=bp
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Body part trajectories across frames")
    plt.legend(ncol=2, fontsize=8)

    if invert_y:
        plt.gca().invert_yaxis()

    plt.show()



if __name__ == "__main__":
    # ---------------------------------------------- setup path -------------------------------------------------

    DATABASE_PATH = "../exploration/no_KO_video_list.csv"
    DATABASE = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(DATABASE.iloc[0]["filename"])

    OUTPUT_DATA_DIR = Path("../data")
    BODYPART_POINTS_PATH = OUTPUT_DATA_DIR / f"csv_results/{VIDEO_EXEMPLE.stem}/pred_results_clip_00.csv"

    # ---------------------------------------------- setup constant -------------------------------------------------

    bodyparts_point = pd.read_csv(BODYPART_POINTS_PATH)
    print(bodyparts_point)
    
    bodyparts = ['elbow_l', 'elbow_r', 'finger_l_1', 
                 'finger_l_2', 'finger_l_3', 'finger_r_1', 'finger_r_2', 
                 'finger_r_3', 'left_hand', 'left_wrist', 'muzzle', 
                 'right_hand', 'right_wrist', 'shoulder_l', 'shoulder_r', 
                 'soft_pad_l', 'soft_pad_r']


    plot_bodyparts_trajectories(BODYPART_POINTS_PATH, ["left_hand"], invert_y=True)
    
