#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import open_clean_csv
from utils.trajectory_metrics import Trajectory, plot_metric_time, plot_stacked_metric
from utils.led_detection import get_time_led_on, get_time_led_off

# set parameters
THRESHOLD = 0.5
BODYPART = 'left_hand'  # or "finger_r_1"
view = 'left'           # or 'right'
LEVER_POSITION = 215    # pixels
LASER_ON_TIME = 0.325   # sec

# choose which function to use
SHOW = True
STACKED_TRAJ = True
STACKED_METRICS = True

# define m per pixel
if view == 'left' : 
    frame_width_m = 8.7 # cm
else : 
    frame_width_m = 8.3 # cm
frame_width_px = 512 # pixel
CM_PER_PIXEL = frame_width_m / frame_width_px

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

# ------------------------------------ output file directory preparation ---------------------------------------


output_fig_dir = OUTPUT_TRAJECTORY_PATH / Path(make_directory_name(Path(DATABASE["filename"][0]).stem))
output_fig_dir.mkdir(parents=True, exist_ok=True)

print(f"File will be stored in {output_fig_dir}")

csv_list = [Path(path) for path in DATABASE["filename"]]
csv_list = csv_list

for path in csv_list : 
    verify_exist(path)

print(f"\nAnalysing trajectory of video with these setting: ")
print([val for val in str(output_fig_dir.name).split("_")])

        
#############################################################################################

output_stacked_traj = output_fig_dir / f"trajectory_stacked.png"
output_stacked_velo = output_fig_dir / f"velocity_stacked.png"
fig, axs = plt.subplots(figsize=(9, 7))

failed_trials : list[dict] = []

for i, csv_path in enumerate(csv_list) :

    all_velocity = []
    all_laser_on = [] 
    all_pad_off = []
    
    verify_exist(csv_path)

    print(f"\n[{i+1}/{len(csv_list)}]")
    print(f"Working on : {csv_path}")

# ------------------------------------ OPENING AND GET SOME VARIABLE  ---------------------------------------

    # get time when pad is ON or OFF
    luminosity_path = GENERATED_DATA_DIR / "luminosity" / RAT_NAME / csv_path.parent.stem / f"luminosity_{csv_path.stem.replace('_pred_results', '')}.csv"
    verify_exist(luminosity_path)

    time_pad_off = get_time_led_off(luminosity_path, "LED_3", in_sec=True) # in sec
    time_laser_on = get_time_led_on(luminosity_path, "LED_4", in_sec=True) # in sec
    all_laser_on.append(time_laser_on)
    all_pad_off.append(time_pad_off)

    # get coords + filtration
    coords = open_clean_csv(csv_path)
    xy = coords[BODYPART].copy()
    xy = xy.assign(t=np.arange(len(xy)) / 125)
    xy_filtered = xy.loc[xy["likelihood"] >= THRESHOLD, ["x", "y", "t"]]

    # verification
    if time_pad_off is None or time_pad_off+LASER_ON_TIME > len(xy_filtered)-1 : # in sec
        print(f"  ! Failed trial on, Pad off at {time_pad_off}")
        failed_trials.append({
            "path": csv_path.as_posix(),
            "reason": f"Failed trial",
            "pad_off": time_pad_off,
        })
        continue
    
    #  pad off -> laser off coords 
    xy_filtered = xy.loc[
        (xy["t"] >= time_pad_off) &
        (xy["t"] <= time_pad_off + LASER_ON_TIME)
    ].reset_index(drop=True)

    print(f"pad off={time_pad_off}, laser on={time_laser_on}")

    # laser on -> laser off coords
    if time_laser_on: 
        xy_laserOn = xy.loc[
            (xy["t"] >= time_laser_on) &
            (xy["t"] <= time_laser_on + 0.3)
        ].reset_index(drop=True)
    else : 
        xy_laserOn = xy_filtered

    # verification
    if len(xy_filtered) == 0 : 
        print(f"  ! Empty reaching coords, Pad off at {time_pad_off}")
        failed_trials.append({
            "path": csv_path.as_posix(),
            "reason": f"Empty reaching coords",
            "pad_off": time_pad_off,
        })
        continue

#     # ------------------------------------ plotting trajectories ---------------------------------------


#     plot_single_bodypart_trajectories(
#         coords=xy_filtered,
#         ax=axs,
#         invert_y=True,
#         color="red",
#         transparancy=0.2
#     )

# avg_coords = (pd.concat(all_coords, axis=1)
#                 .T
#                 .groupby(level=0)
#                 .mean()
#                 .T)

# plot_single_bodypart_trajectories(
#         coords=avg_coords,
#         ax=axs,
#         invert_y=True,
#         color="blue",
#         transparancy=1
#     )

# title = (
#     "Trajectories across trials with settings:\n"
#     f"{output_fig_dir.stem}\n"
#     f"Number of trials: {len(csv_list)}"
# )

# print("\nFailed trial : ")
# print(str(path) for path in failed_trial)
# print(f"Number of failed trial : {len(failed_trial)}")

# axs.set_title(title, fontsize=12)
# fig.savefig(output_stacked_traj)

# if SHOW:
#     plt.show()
# plt.close(fig)



    # ------------------------------------ plotting metrics ---------------------------------------

    # creat trajectory object for metric calculation
    Traj = Trajectory(coords=xy,
                    reaching_coords=xy_filtered, 
                    laserOn_coords=xy_laserOn,
                    fps=125, 
                    cm_per_pixel=CM_PER_PIXEL)

    instant_velo : pd.DataFrame = Traj.instant_velocity()
    all_velocity.append(instant_velo.set_index("t")["velocity"])

    # plotting
    plot_stacked_metric(data=instant_velo["velocity"],
                    time=instant_velo["t"],
                    laser_on=time_laser_on,
                    ax = axs, 
                    color="green",
                    transparancy=0.3)


# plotting average velocity over time
avg_velocity = pd.concat(all_velocity, axis=1).mean(axis=1)
avg_pad_off = np.array(all_pad_off).mean()

if all_laser_on[0] is not None : 
    avg_laser_on = np.array(all_laser_on).mean()
else : 
    avg_laser_on = None

plot_stacked_metric(data=avg_velocity,
                time = avg_velocity.index,
                laser_on=avg_laser_on,
                ax = axs, 
                show_pad_off=True,
                color="blue",
                transparancy=1)

title = (
    "Velocities across trials with settings:\n"
    f"{output_fig_dir.name}\n"
    f"Number of trials: {len(csv_list)}"
    )

axs.set_title(title, color="blue")
axs.set_xlabel("Time (seconds)")
axs.set_ylabel("Velocity (cm.s$^{-1}$)")

print(f"\nNumber of failed trial : {len(failed_trials)}")

# save failed trial
pd.DataFrame(failed_trials).to_csv(output_fig_dir / "failed_trial.csv", index=False)
fig.savefig(output_stacked_velo)

if SHOW:
    plt.show()
plt.close(fig)
