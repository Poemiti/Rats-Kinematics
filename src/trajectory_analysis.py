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

def plot_bodyparts_trajectories(csv_path, bodyparts : list[str] = None, invert_y=True) -> None:
# DLC CSV has 3 header rows
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Drop scorer level → keep (bodypart, coord)
    df.columns = df.columns.droplevel(0)

    # Get body parts automatically
    all_bodyparts = df.columns.get_level_values(0).unique()

    if bodyparts is None:
        bodyparts = list(all_bodyparts)
        bodyparts.remove("bodyparts")

    print(f"\nbodypart : {bodyparts}")

    plt.figure()

    for bp in bodyparts:
        if bp not in bodyparts:
            continue

    
        # Select x,y only (ignore likelihood)
        xy = df[bp].loc[:, ["x", "y"]]
        print(f"\nbodypart : {bp}, coord : \n{xy}")

        plt.plot(
            xy["x"],
            xy["y"],
            marker="o",
            linewidth=1,
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

    OUTPUT_DATA_DIR = Path("../data")
    DATABASE_PATH = Path(OUTPUT_DATA_DIR / "csv_results/result_predict_csv_3.csv")

    # ---------------------------------------------- setup constant -------------------------------------------------

    # plot_bodyparts_trajectories(DATABASE_PATH, invert_y=True)

    # plot_bodyparts_trajectories(DATABASE_PATH, invert_y=False)
    
    bodyparts = ['épaule', 'coude', 'poignet_droit', 'main_droite', 'doigtd_1', 'doigtd_2', 'doigtd_3', 'coussinetd', 'main_gauche', 'coussinetg', 'doigtg_1']

    plot_bodyparts_trajectories(DATABASE_PATH, ["doigtd_1"], invert_y=False)
    plot_bodyparts_trajectories(DATABASE_PATH, ["doigtd_1"], invert_y=True)
    
