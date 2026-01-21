# Rat Kinematics - QUICK DOCUMENTATION

This document lists **all functions organized in tables** for quick reference. Detailed usage and examples remain in each module.

---

## `split_video.py`

| Function                 | Description                                                          | Inputs / Notes                                                               | Outputs                               |
| ------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------- |
| `extract_frames()`       | Extract frames from a video and organize them into trial-based clips | Assumes fixed trial duration; handles incorrect FPS metadata (125 vs 30 FPS) | Trial folders containing frame images |
| `get_video_properties()` | Retrieve metadata and derived statistics from a video                | Uses OpenCV/FFmpeg                                                           | Dictionary of video properties        |
| `frame_to_video()`       | Reconstruct a video from image frames                                | Frames named `frame_XXX.png`                                                 | MP4 video                             |
| `split_video()`          | Split a video into fixed-duration clips using FFmpeg                 | Two-step process: re-encode then split                                       | Multiple video clips                  |
| `run_ffmpeg()`           | Execute an FFmpeg command                                            | FFmpeg ≤ 4.2.2 required                                                      | Command execution result              |

---

## `database_filter.py`

| Component    | Role             | Description                                              |
| ------------ | ---------------- | -------------------------------------------------------- |
| `Model`      | MVC – Model      | Handles database loading, filtering, and transformations |
| `View`       | MVC – View       | Displays plots and filtered results                      |
| `Controller` | MVC – Controller | Links GUI inputs to data filtering and visualization     |

---

## `dlc_prediction.py`

| Function                   | Description                                           | Notes                                | Outputs                    |
| -------------------------- | ----------------------------------------------------- | ------------------------------------ | -------------------------- |
| `dlc_predict_Rejane()`     | Run DeepLabCut prediction following Rejane’s workflow | Requires DLC environment             | CSV of predicted points    |
| `move_outputs()`           | Move relevant DLC outputs to final destination        | Used internally                      | Organized prediction files |
| `cleanup_temp_directory()` | Remove unnecessary DLC temporary files                | Frees disk space                     | Clean temp directory       |
| `dlc_prediction_Julien()`  | Modified DLC prediction for mamserver                 | Adapted from Julien’s implementation | Xarray + optional CSV      |

---

## `file_management.py`

| Function                            | Description                                     | Notes                              |
| ----------------------------------- | ----------------------------------------------- | ---------------------------------- |
| `is_left_view()`                    | Determine camera view (left/right)              | H001 → left, H002 → right          |
| `is_video()`                        | Check if file is a video                        | Supports `.avi`, `.mp4`            |
| `is_csv()`                          | Check if file is a CSV                          | —                                  |
| `sort_componants()`                 | Sort filename components into categories        | Numbers, alpha, alphanumeric, etc. |
| `decompose_filename()`              | Split filename into underscore-separated tokens | Extension removed                  |
| `extract_type()`                    | Extract substring using regex                   | Used for filename classification   |
| `classify_file()`                   | Classify video based on filename metadata       | Adds metadata dictionary           |
| `display_count_per_rat_condition()` | Plot video counts per rat type and condition    | Visualization function             |
| `display_images()`                  | Display images in a grid                        | Uses matplotlib                    |
| `make_database()`                   | Build a metadata database from directory tree   | Returns DataFrame                  |

---

## `led_detection.py`

| Function            | Description                                | Notes                     |
| ------------------- | ------------------------------------------ | ------------------------- |
| `define_cue_type()` | Determine cue type from LED_1 luminosity   | Threshold-based detection |
| `is_led_on()`       | Detect LED activation over time            | Used mainly for LED_4     |
| `rename_clip()`     | Rename clip based on cue and laser state   | File-system operation     |
| `classify_clip()`   | Classify and rename clip using LED signals | Combines cue + laser info |

---

## `video_annotation.py`

| Function                     | Description                              | Notes                        |
| ---------------------------- | ---------------------------------------- | ---------------------------- |
| `annotate_video_xr()`        | Overlay pose points from xarray on video | Likelihood thresholding      |
| `annotate_video_csv()`       | Overlay pose points from CSV             | Same logic as xarray version |
| `annotate_single_bodypart()` | Annotate trajectory of one body part     | Useful for debugging         |

---

## `trajectory_analysis.py`

| Function                          | Description                               | Outputs           |
| --------------------------------- | ----------------------------------------- | ----------------- |
| `open_clean_csv()`                | Load and clean DLC CSV with multi-headers | Cleaned DataFrame |
| `plot_bodyparts_trajectories()`   | Plot trajectories for body parts          | Matplotlib Ax     |
| `plot_stacked_trakectories()`     | Plot multiple trial trajectories together | Combined plot     |
| `plot_average_trajectories()`     | Plot average trajectory across trials     | Mean trajectory   |
| `define_StartEnd_of_trajectory()` | Detect start/end of movement              | Index range       |
| `get_instantaneous_velocity()`    | Compute instantaneous velocity            | Velocity list     |
| `get_velocity()`                  | Compute average velocity                  | Scalar value      |
| `get_distance()`                  | Compute traveled distance                 | Scalar value      |
| `compute_metric()`                | Compute kinematic metric for a clip       | Metric value      |
