## Folder sytem

```
-- data/
    |
    | -- csv_results/               # results from the dlc model prediction
    | -- dlc_results/
    |
    | -- frames/                    # generated frames for each clip
    | -- clips/                     # generated videos (clips) from the frames

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
 - from the small video make points prediction using rejane model using the function made bu Julien  
 - get_luminosity from the small video  
 - match trajectory of videos with led/cues/laser   


### led signification

- LED 1 : activation du stimulus sonore (1 ou 2 impulsions),  
- LED 2 et LED 3 : présence des pattes droite et gauche sur les pads (allumée = patte posée),  
- LED 4 : activation du laser de stimulation (si applicable),  
- LED 5 : validation de la tâche correcte, entraînant une récompense.  


