
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

### Label studio annotation signification

| Rat name                                                                  | H001        | H002       |
| --------                                                                  | -------     | -------    |
| Rat_#517Ambidexter_20240523_ContiMT300_LeftHemi_L1L2_C001H001             | 1812        | 1811       |
| Rat_#517Ambidexter_20240527_BetaMT300_RightHemiCHR_L1L25050_C001H001      | 1813        | ....       |
| Rat_#517Ambidexter_20240624_BetaMT300_RightHemiCHR_L1L25050_C001H001      | 1814        | ....       |



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
        |
        | -- #517/
            |
            |-- Rat_#512_....video_name.../
                |
                | -- clip_annotation/    # trajectory on clip video
                | -- trajectory_per_clip # graphic of the trajectory
                | 
                | trajectory_stacked.png # stacked trajectory 
                | trajectory_average.png # average trajectory 


-- exploration/                     # jupyter notebook to explore / test things
    |
    |-- data/                           # generated data of utils files on their own, it s for test only
            |
            | ...

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
    |
    | tkinter_analysis.py           # tkinter app for filtering file and make a database out of it           

-- 
```
