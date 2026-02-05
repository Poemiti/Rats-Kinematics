import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from utils.database_filter import Model, View, Controller
from utils.file_management import is_video, make_database, make_directory_name, is_csv, verify_exist
from utils.trajectory_ploting import open_clean_csv, plot_single_bodypart_trajectories
from utils.trajectory_metrics import Trajectory, plot_metric_time, plot_stacked_metric
from utils.led_detection import get_time_led_on, get_time_led_off

# --- CONFIGURATION ---
CONFIG = {
    "THRESHOLD": 0.5,
    "BODYPART": 'left_hand',
    "FPS": 125,
    "LASER_ON_TIME": 0.325,
    "MAX_LOST_COORDS": 10,
    "VIEW": 'left',
    "FRAME_WIDTH_PX": 512,
}

# Calculate derived constants
frame_m = 8.7 if CONFIG["VIEW"] == 'left' else 8.3
CM_PER_PIXEL = frame_m / CONFIG["FRAME_WIDTH_PX"]

PATHS = {
    "DATA": Path("../data"),
    "PREDICTIONS": Path("../data/csv_results"),
    "DATABASE": Path("../data/database"),
    "LUMINOSITY": Path("../data/luminosity")
}





def process_single_trial(csv_path, rat_name):
    """Extracts and cleans data for one trial. Returns a dict or None if failed."""
    
    # 1. Get Timing Data
    stem = csv_path.stem.replace('pred_results_', '')
    lum_path = PATHS["LUMINOSITY"] / rat_name / csv_path.parent.stem / f"luminosity_{stem}.csv"
    
    if not lum_path.exists():
        return {"status": "fail", "reason": "Missing luminosity file"}

    time_pad_off = get_time_led_off(lum_path, "LED_3", in_sec=True)
    time_laser_on = get_time_led_on(lum_path, "LED_4", in_sec=True)

    # 2. Get Coords & Filter
    coords = open_clean_csv(csv_path)
    xy = coords[CONFIG["BODYPART"]].copy()
    xy = xy.assign(t=np.arange(len(xy)) / CONFIG["FPS"])
    
    # 3. Validation Logic
    xy_valid = xy.loc[xy["likelihood"] >= CONFIG["THRESHOLD"], ["x", "y", "t"]]
    n_lost = len(coords) - len(xy_valid)

    if n_lost > CONFIG["MAX_LOST_COORDS"]:
        return {"status": "fail", "reason": "Too many lost coords", "lost": n_lost}
    
    if time_pad_off is None or (time_pad_off + CONFIG["LASER_ON_TIME"]) > xy["t"].max():
        return {"status": "fail", "reason": "Invalid pad-off timing", "lost": n_lost}

    # 4. Slice Data
    reaching_coords = xy_valid.loc[
        (xy_valid["t"] >= time_pad_off) & 
        (xy_valid["t"] <= time_pad_off + CONFIG["LASER_ON_TIME"])
    ].reset_index(drop=True)

    if reaching_coords.empty:
        return {"status": "fail", "reason": "Empty reaching slice", "lost": n_lost}

    return {
        "status": "success",
        "coords": reaching_coords,
        "full_xy": xy_valid,
        "time_laser_on": time_laser_on,
        "time_pad_off": time_pad_off,
        "lost_count": n_lost
    }




def plot_velocity_analysis(ax, all_results, output_path):
    all_velos = []
    for res in all_results:
        # Trajectory metric calculation
        traj_obj = Trajectory(coords=res['full_xy'], reaching_coords=res['coords'], 
                              fps=CONFIG["FPS"], cm_per_pixel=CM_PER_PIXEL)
        v = traj_obj.instant_velocity()
        all_velos.append(v)
        
        plot_stacked_metric(data=v["velocity"], time=v["t"], ax=ax, color="green", transparency=0.3)

    # Plot Average
    avg_v = pd.concat(all_velos).groupby('t').mean()
    plot_stacked_metric(data=avg_v["velocity"], time=avg_v.index, ax=ax, color="blue", transparency=1)
    
    ax.set_title("Velocity Stacked")
    plt.savefig(output_path)




def main():
    # --- 1. Database Setup ---
    raw_db = make_database(PATHS["PREDICTIONS"], is_csv)
    controller = Controller(Model(raw_db, PATHS["DATABASE"]), View())
    controller.view.mainloop()
    
    DATABASE = controller.filtered_dataset.reset_index(drop=True)
    RAT_NAME = DATABASE['rat_name'][0]
    
    # --- 2. Process All Trials ---
    successful_trials = []
    failed_trials = []

    print(f"Processing {len(DATABASE)} files...")
    for _, row in DATABASE.iterrows():
        result = process_single_trial(Path(row['filename']), RAT_NAME)
        
        if result["status"] == "success":
            successful_trials.append(result)
        else:
            failed_trials.append({"path": row['filename'], **result})

    # --- 3. Run Analysis/Plotting ---
    if successful_trials:
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # You can toggle these easily now
        output_dir = PATHS["DATA"] / "analysis_results" / RAT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_velocity_analysis(ax, successful_trials, output_dir / "velocity.png")
        # Add other plot calls here...
        
    # --- 4. Export Stats ---
    pd.DataFrame(failed_trials).to_csv(output_dir / "failed_log.csv")
    print(f"Done! Success: {len(successful_trials)}, Failed: {len(failed_trials)}")




if __name__ == "__main__":
    main()