#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import plot_bodyparts_trajectories, plot_stacked_trajectories, plot_average_trajectories, compute_metric, get_distance, get_velocity
from utils.video_annotation import annotate_single_bodypart

THRESHOLD = 0.5
BODYPART = ['left_hand']
SINGLE_TRAJ = False

# ------------------------------------ setup path ---------------------------------------

GENERATED_DATA_DIR = Path("../data")
CLIP_DIR = GENERATED_DATA_DIR / "clips"
PREDICTION_DIR = GENERATED_DATA_DIR / "csv_results"
DATABASE_DIR = GENERATED_DATA_DIR / "database" 

# ------------------------------------ make database ---------------------------------------


raw_database = make_database(PREDICTION_DIR, is_csv)

model = Model(raw_database, DATABASE_DIR)
view = View()
controller = Controller(model, view)
view.mainloop()

DATABASE = controller.filtered_dataset.reset_index(drop=True)
print(DATABASE)

# save database
if controller.dataset_name.get() :
    dataset_name = f"{controller.dataset_name.get().strip()}.csv"
    DATABASE.to_csv(DATABASE_DIR / dataset_name)
    print(f"\nFiltered dataset saved as : {DATABASE_DIR / dataset_name}")

print(f"\nNumber of files in database : {len(DATABASE)}")

RAT_NAME = DATABASE['rat_name'][0]

OUTPUT_TRAJECTORY_PATH = GENERATED_DATA_DIR / "analysis_results" / RAT_NAME
OUTPUT_TRAJECTORY_PATH.mkdir(parents=True, exist_ok=True)

# # ------------------------------------ plot stacked trajectory ---------------------------------------

# output_fig_dir = OUTPUT_TRAJECTORY_PATH / Path(make_directory_name(Path(DATABASE["filename"][0]).stem))
# output_fig_dir.mkdir(parents=True, exist_ok=True)

# print(f"File will be stored in {output_fig_dir}")

# csv_list = [Path(path) for path in DATABASE["filename"]]
# csv_list = csv_list

# for path in csv_list : 
#     verify_exist(path)

# print(f"\nAnalysing trajectory of video with these setting: ")
# print([val for val in str(output_fig_dir.name).split("_")])

# output_stacked_traj = output_fig_dir / f"trajectory_stacked.png"
# output_mean_traj = output_fig_dir / f"trajectory_average.png"

# plot_stacked_trajectories(csv_list= csv_list, 
#                         output_fig_path= output_stacked_traj,
#                         bodyparts= BODYPART, 
#                         invert_y=True,
#                         show=False,
#                         threshold=THRESHOLD)



# metrics : list[dict] = []
    
# for i, csv_path in enumerate(csv_list) : 
    
#     verify_exist(csv_path)

#     print(f"[{i}/{len(csv_list)}]")
#     print(f"\nWorking on : {csv_path}")
    

#     if SINGLE_TRAJ : 
#         # ------------------------------------ plot single trajectory + video annotation  ---------------------------------------
        
#         print("Making single figures ...")

#         output_traj_dir = output_fig_dir / "trajectory_per_clip" 
#         output_traj_path = output_fig_dir / "trajectory_per_clip" / f"trajectory_{csv_path.name}.png"
#         output_traj_dir.mkdir(parents=True, exist_ok=True)

#         ax = plot_bodyparts_trajectories(csv_path= csv_path, 
#                                         bodyparts= BODYPART, 
#                                         invert_y=True,
#                                         threshold=THRESHOLD)
        
#         ax.set_title(f"Trajectories of \n{csv_path.name}")
#         ax.legend()

#         fig = ax.figure
#         fig.savefig(output_traj_path)

#         plt.close(fig)

#         # --------------------------------------- single bodypart trajectory annotation ----------------------------------------------

#         print(f"Making video annotation ...")
#         output_annotated_clip_dir = output_fig_dir / "annotated_clips"
#         output_annotated_clip_dir.mkdir(parents=True, exist_ok=True)

#         input_clip_path = CLIP_DIR / csv_path.parent.stem / f"{csv_path.stem.replace('_pred_results', '')}.mp4"
#         verify_exist(input_clip_path)
        
#         annotate_single_bodypart(video_path=input_clip_path,
#                                 csv_path=csv_path,
#                                 output_path=output_annotated_clip_dir / f"annotated_{csv_path.name}.mp4",
#                                 bodypart_name=BODYPART[0],
#                                 radius=5,
#                                 likelihood_threshold= THRESHOLD)
        


#     # -------------------------------------------- metric measurments ----------------------------------------------

#     print("Metrics measurement ...")
#     # distance
#     distance = compute_metric(csv_path=csv_path,
#                             bodyparts="left_hand",
#                             metric=get_distance)
    
#     # velocity
#     velocity = compute_metric(csv_path=csv_path,
#                             bodyparts="left_hand",
#                             metric=get_velocity)

#     metrics.append({"file" : csv_path,
#                     "distance" : distance,
#                     "velocity" : velocity
#     })

# # save computed metrics
# metrics_df = pd.DataFrame(metrics)
# metrics_df.to_csv(output_fig_dir / "metrics_per_clips.csv", index=False)

print("\nDone !")


