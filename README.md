
## Pipeline : 

 - extract videos  
 - cut them evry 3 s (time of the trial)  
 - from the small video make points prediction using rejane model using the function made by Julien  
 - get_luminosity from the small video  
 - match trajectory of videos with led/cues/laser   


### led signification

- LED 1 : CUE, activation du stimulus sonore (1 ou 2 impulsions),  
- LED 2 : PAD DROITE (allumée = patte posée),
- LED 3 : PAD GAUCHE (allumée = patte posée),  
- LED 4 : OPTO, activation du laser de stimulation (si applicable),  
- LED 5 : REWARD, validation de la tâche correcte, entraînant une récompense.  


## Perfomance prediction

**some numbers :**  
Total number of raw video (Raphael ones, without KO rats) : 644  
Average number of clips/raw video                         : 43  
Expected Total number of clips                            :  27692  


| task              | time (only 1)           | total time (all raw video) | python function         |
| ----              | ----                    | ----                       | ----                    |
| splitting         | 37.32 sec (1 raw video) | 6.68 h                     | split_video             |
| point prediction  | 30.74 sec (1 clip)      | 236.47 h                   | dlc_predict_Rejane      |
| video annotation  | 0.79 sec  (1 clip)      | 6.05 h                     | annotate_video_from_csv |


## Function available
in src/
`PIPELINE_prediction.py` : MAIN PIPELINE for prediction (conda env = DEEPLABCUT)  
`PIPELINE_analysis.py`   : MAIN PIPELINE for analysis of predictions (conda env = kinematics)    
  
`split_video_by_trial.py ` : function to split video into clips of a certain duration  
  
| Function name          | Output                         |
| ---------------------- | ------------------------------ |
| `extract_frames`       | Collection of `frame.png`      |
| `frames_to_video`      | One `clip.mp4`                 |
| `split_video`          | Collection of `clip.mp4`       |
| `get_video_properties` | Dictionary of video properties |
  
`split_video_by_trial.py` : function to split video into clips of a certain duration  
  
| Function name             | Output                      |
| ------------------------- | --------------------------- |
| `dlc_predict_Rejane`      | `results.h5`, `results.csv` |
| `dlc_predict_Julien`      | `results.h5`, `results.csv` |
| `annotate_video`          | `annotated_clip.mp4`        |
| `annotate_video_from_csv` | `annotated_clip.mp4`        |
  
`trajectory_analysis.py` : analysis function  
  
| Function name                 | Output                            |
| ----------------------------- | --------------------------------- |
| `plot_bodyparts_trajectories` | `plt.show()`                      |
| `get_luminosity`              | `xarray` of luminosity properties |


## Folder sytem

```
-- data/                            # all the FINAL generated data from scripts of src/
    |
    | -- csv_results/               # results from the dlc model prediction (csv and h5)
    | -- dlc_results/
    |
    | -- frames/                    # generated frames for each clip
    | -- clips/                     # generated videos (clips) from the raw video
    |
    | -- temporary/                 # temporary dlc prediction (always empty because it erase the temporary files generated)
    | -- video_annotation/          # clip + point annotation on top
    | -- luminosity_figures/        # track led lightening : csv + image html
    | -- trajectory_figures/        # .png of the trajectory of video (single clip, stacked clips, average clip)


-- exploration/                     # jupyter notebook to explore / test things
    |
|-- data/                           # generated data from src/ python files (on their own, it s for test only)
        |
        | -- csv_results/           # results from the dlc model prediction (csv and h5)
        | -- dlc_results/
        |
        | -- frames/                # generated frames for each clip
        | -- frame2clips/           # generated videos (clips) from the frames
        | -- direct_clips/          # generated videos (clips) from the raw video
        |
        | -- temporary/             # temporary dlc prediction (always empty because it erase the temporary files generated)
        | -- video_annotation/      # clip + point annotation on top
        | -- luminosity_figures/    # track led lightening : csv + image html
        | -- trajectory_figures/    # .png of the trajectory of video (single clip, stacked clips, average clip)
    |
    | -- figures/                   # generated figures from the notebooks
    | -- clip_video_test/           # generated videos from the src/split_video_by_trial.py
    |
    | led_recon.ipynb               # test on led recognition (automatic)
    | video_sorting.ipynb           # exploration of the videos and the expermimental condition
    |
    | video_list_sorted.csv         # output of video_sorting.py
    | no_KO_video_list.csv

-- src/                             # "official" scripts
    |
    | -- clips/                     # generated videos from split_by_trial.py
    |
    | dlc_prediction.py             # function to run a dlc prediction
    | split_video_by_trial.py       # function to split video into clips of a certain duration 
    | trajectory_analysis           # function to analyse trajectory, and sync it with leds
    |
    | PIPELINE_prediction.py        # MAIN PIPELINE for prediction (conda env = DEEPLABCUT)
    | PIPELINE_analysis.py          # MAIN PIPELINE for analysis of predictions (conda env = kinematics)

-- 
```
