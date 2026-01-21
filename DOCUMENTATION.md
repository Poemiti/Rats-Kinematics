# RAT KINEMATICS DOCUMENTATION
The `utils` modules made, was develop in the purpose of analysing video of [https://www.bordeaux-neurocampus.fr/en/team/network-dynamics-for-procedural-learning/](IMN - Network dynamics for procedural learning research) team.  

Each of these script can be used has standalone. Each of them have a exemple provided in the main.   
All the outputs of these script alone will be stored in `exploration/data` in order to not overlap with the actual `data` that comes from the pipelines  

### Dependancies :
You will need at least 2 environment :   
- One for making prediction which use DeepLabCut (DLC). Follow the [https://deeplabcut.github.io/DeepLabCut/docs/installation.html](Official tuto) 
and do not add any other package or it will break the configuration
- a second one for all the other task : especially those using cv2, and display   

# Module documentation 



## split_video.py
Function for handling behavioral video data, including frame extraction, video splitting, and metadata inspection.
This module supports both OpenCV-based and FFmpeg-based workflows and is designed to handle experimental constraints specific to trial-based recordings.

### extract_frames()
Extract frames from a single video and organize them into trial-based clips.  

During the experimental procedure, behavioral trials are recorded sequentially. 
Each trial corresponds to a cue and optional stimulation and lasts a fixed duration.
The camera records at 125 FPS, but due to metadata inconsistencies, the saved .avi files may incorrectly declare 30 FPS.  

This function extracts frames assuming a fixed trial duration (duration), creating a folder for each trial and saving individual frames as images.  

**Important FPS note**
- To extract the correct number of frames per trial:  
- Use 3 seconds if working at 125 FPS  
- Use 12.5 seconds if relying on 30 FPS metadata  
In both cases, a single trial corresponds to 375 frames.  

Output structure : 

```
output_path/
└── <video_name>/
    └── clip_00/
        ├── frame_000.png
        ├── frame_001.png
        └── ...
```

### get_video_properties()
Returns metadata and derived statistics from a video file as a dictionnary

### frame_to_video()
Reconstruct a video from a directory of image frames.
Frames must follow the naming convention `frame_XXX.png` and are assembled in sorted order into an MP4 video file.

### split_video()
Split a video into multiple fixed-duration clips using FFmpeg.

This function performs the splitting in two steps:
1. Re-encoding, FPS normalization + compression
The input video is optionally reinterpreted at a target FPS and compressed using H.264 (CRF-controlled).
2. clip splitting
The re-encoded video is split into multiple clips of equal duration without further re-encoding.

Output clips follow the naming convention: `<output_path_stem>_clip_XX.mp4`

### run_ffmpeg()
Execute an FFmpeg command using subprocess  
It is used by split_video()
**Note : ffmpeg must be installed (4.2.2 or less)**





## database_filter.py
Implements an MVC architecture for filtering and querying experimental databases.
This GUI uses buttons to set filter, and an entry to write the name of the newly filtered databse.  
*Recommended name for a database* : folow the order of the filter used and seperate them with underscores. And no need to specify the `.csv` extention.    
Exemple : `Rat_517_CTRL_Conti_S0001_1.5mw`

### class Model
Handles data loading, filtering, and transformation.

### class View
Responsible for plotting and visualizing filtered results.

### class Controller
Connects user inputs to database queries and visual outputs.






## dlc_prediction.py
Function to launch a DLC prediction using one of the previously trained model
**Note : When this module is used, you must be in an environement that has deeplabcut installed**

### dlc_predict_Rejane()
Prediction like Rejane used did. It uses move_outputs and cleanup_temp_directory.
Returns : csv file with all the point prediction made

### move_outputs()
move the right output of dlc_predict_Rejane in their final destination

### cleanup_temp_directory()
DLC produce a lot a unessessary files, and those are stored in a temporary folder which this function clean

### dlc_prediction_Julien()
Modified version of the function of Julien implemented for the mamserver.  
Returns :   
- xarray file with the prediction points
- save a csv version of the same prediction (if a output path is given)




## file_management.py

### is_left_view()
Determine whether a video corresponds to the left camera view.
- H001 : left view  -> return True
- H002 : right view -> return False

### is_video()
Check if a file is a video file.
Supported video formats are ``.avi`` and ``.mp4``.

### is_csv()
Check if a file is CSV

### sort_componants()
Video files have very long names that are manually written. This function is used to 
sort the componant of such name, in 5 category (those quategory are not meanningful) :   
numbers, alpha, alphanumeric, special, mixed

### decompose_filename() 
Decompose a video filename into underscore-separated components.
The file extension is removed before splitting.

### extract_type()
Extract a substring from a string using a regular expression.  
It is used to extract the componant of a filename to then classify the filename (r.f classify_file())

### classify_file()
The function decomposes the filename into tokens that are then
used to classify the file in certain categories.
The extracted metadata is appended as a dictionary to `videos`.   
Categories are :   
- "rat_name"  
- "rat_type"  
- "condition"  
- "stim_location"  
- "task"  
- "handedness"  
- "session"  
- "view"  
- "laser_intensity"  
- "laser_on"  

### display_count_per_rat_condition()
Plot the number of videos per rat type (CHR or CTRL) and experiment condition
experimental condition are the categories in which each files are classified

### display_images()
Display a list of images in a grid layout.

### make_database()
Build a database of video metadata from a directory tree.
Returns : Dataframe that has all the files and their categorisation




## led_detection.py

### define_cue_type()
Determine the cue type based on LED luminosity over time.  
**It must be applied on LED_1 only**  
he function analyzes a sequence of luminosity values and counts
the number of sustained light activations. A cue is detected when
luminosity exceeds a fixed threshold (50) for more than a given number
of consecutive frames (5 frames).
Cue classification:  
- 1 activation  -> ``CueL1`` = Activate lever with **left** paw  
- 2 activations -> ``CueL2`` = Activate lever with **right** paw  
- otherwise     -> ``NoCue``  


### is_led_on()
Determine if a LED is active during a clip. It uses the same principal
as `define_cue_type()`.  
This function is mainly used for the LED_4 that is the one that tells
if a the laser was ON

### rename_clip()
Rename a video clip based on detected cue and laser state.

### classify_clip()
Classify a video clip based on LED signals and rename it accordingly.  

The function determines:  
- Cue type from LED_1 luminosity  
- Laser activation from LED_4 luminosity  
It then renames the clip using this information.  






## video annotation.py
This module add overlay on video to be able to see how the predicted point are in 
comparison to the actual video/clip.
**Note : dlc_prediction must be compute before **

### annotate_video_xr()
Annotate a video with pose estimation data stored in an xarray DataArray.

This function overlays colored dots on each frame to represent
detected body part positions. One color is assigned per body part.
Annotations are applied only when the likelihood exceeds a given threshold.

### annotate_video_csv()
cf annotate_video_from_xr

### annotate_single_bodypart()
Annotate a video with the trajectory of a single body part.




## trajectory_analysis.py

### open_clean_csv()
Load and clean a DeepLabCut CSV file because they have 3 header (scorer, bodypart, coordinate)  
REturn : Dataframe with 2 level : bodypart and coordinate

### plot_bodyparts_trajectories()
Plot body part trajectories from a DeepLabCut CSV file.
Return an Ax, that has to be plt.show outside of the function


### plot_stacked_trakectories()
plot trejectories from multiple trial on a same figure
It calls `plot_bodypart_trajectories()`


### plot_average_trajectories()
plot the average trajectory from multiple trials


### define_StartEnd_of_trajectory()
Determine the start and end indices of a movement trajectory.

A trajectory is defined as the movement from the pad to the lever.
The start is fixed at the beginning of the sequence, and the end
is detected when the y-coordinate crosses a predefined lever position twice.

Lever position is set at 210 pixels

### get_instantaneous_velocity()
Return : list of instantaneous velocity from coordinates


### get_velocity()
Return : average velocity from coordinates

### get_distance()
Return : travelled distance of a limb from coordinates

### compute_metric()
Compute a kinematic metric from DeepLabCut predictions for one clip.
The metrics computed will be one specified as parameter (e.g see exemple in script)

Coordinates are filtered by likelihood and restricted to the
active movement segment before computing the metric.