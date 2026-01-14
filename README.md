## Folder sytem

```
-- data/
    |
    | -- csv_results/               # results from the dlc model prediction
    | -- dlc_results/
    |
    | -- frames/                    # generated frames for each clip
    | -- frame2clips/               # generated videos (clips) from the frames
    | -- direct_clips/              # generated videos (clips) from the raw video
    |
    | -- temporary/                 # temporary dlc prediction (always empty because it erase the temporary files generated)
    | -- video_annotation/          # clip + point annotation on top


-- exploration/
    |
    | -- figures/                   # generated figures from the notebooks
    | -- clip_video_test/           # generated videos from the src/split_video_by_trial.py
    |
    | led_recon.ipynb               # test on led recognition (automatic)
    | video_sorting.ipynb           # exploration of the videos and the expermimental condition
    |
    | video_list_sorted.csv         # output of video_sorting.png
    | no_KO_video_list.csv

-- src/
    |
    | -- clips/                     # generated videos from split_by_trial.py
    |
    | dlc_prediction.py             # function to run a dlc prediction
    | split_video_by_trial.py       # function to split video into clips of a certain duration 
    |
    | main.py                       # MAIN PIPELINE that uses the previous functions

-- 
```

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


## Perfomence prediction

**some numbers :**
Total number of raw video (Raphael ones, without KO rats) : 644  
Average number of clips/raw video                         : 43  
Expected Total number of clips                            :  27692  


| task              | time (only 1)           | total time (all raw video) | python function         |
| ----              | ----                    | ----                       | ----                    |
| splitting         | 37.32 sec (1 raw video) | 6.68 h                     | split_video             |
| point prediction  | 30.74 sec (1 clip)      | 236.47 h                   | dlc_predict_Rejane      |
| video annotation  | 0.79 sec  (1 clip)      | 6.05 h                     | annotate_video_from_csv |

