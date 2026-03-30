# Rat kinematics
This repository contains a pipeline to anaylse videos of rat doing a reaching task.  
The reaching movement is tracked using [DeepLabCut](https://github.com/DeepLabCut) package for *animal pose estimation*.

## Table of Content

1. [Pipeline](##Pipeline)
2. [Led signification](##Led-signification)
3. [Annotation signification](##Label-studio-annotation-signification)
4. [Folder system](##Folder-sytem)

## Pipeline 

### 1. `make_prediction`

Makes the trajectory prediction using a pretrained model :  
- select videos from the *raw_video* folder
- cut those video into clips to separate each trial (3sec each)
- run deeplabcut prediction using a pretrained model (see [Reaching-DLC-model](https://github.com/Poemiti/Reaching-DLC-model))
output : 
- mp4 clips in *data/clips* and   
- csv trajectory prediction in *data/csv*


### 2. `rename_files`    
Rename clips (and associated CSV files) if naming is inconsistent or incorrect   


### 3. `read_led`  

Analyse the luminosity of the leds on the trials clips :  
- select the clips you want to analyse from the *data/clips* folder
- depending on the view of the clip (see [Annotation signification](##Label-studio-annotation-signification))
- for each frame of the clip measure the luminosity intensity of each Led
- define which cue was on and if the optogenetic stimulation was on (see [Led signification](##Led-signification))
- depending on which cue was on, and the opto stimulation, it will rename the mp4 clip and the corresponding csv trajectory file.   
exemple: 
*Rat_#517Ambidexter_20240523_ContiMT300_LeftHemi_L1L2_C001H001*   
if it has a cue for the left lever (LED 1), and the opto laser was ON (LED 4), it will be renamed :   *Rat_#517Ambidexter_20240523_ContiMT300_LeftHemi_CueL1_C001H001_LaserOn*   


### 4. `preprocessing`  

Preprocess the trajectory :   
- select the trajectory you want to preprocess  
- make a joblib file with the metadata of this trial and the bodypart setup in the config
- plot the raw trajectory and the interpolated one (will be used for the validation)   

output:   
- trajectory figures in *data/figures_results*,   
- metadata in *data/metrics_results* : 
    - filename_coords : path of the csv with the trajectory points
    - filename_luminosity : path of the csv with the luminosity of each led
    - filename_clips : path of the mp4 clips
    - date : data of the trial
    - condition : opto stimulation type of the trial (beta, conti) 
    - nb_clip : clip number
    - laser_intensity : intensity of the opto stimulation
    - pad_off : time at which the rat lift its paw
    - laser_on : time at which the opto laser is on
    - reward : time at which the reward is given to the rat
    - bodypart :
        - xy_before : raw trajectory point of the bodypart
        - xy_after : interpolated trajectory points 
        - trial_success : success of the trial for this particular bodypart


### 5. `preprocessing_validation`  

In order to see if the trial is successful, a validator has been made. It open a window showing the trajetory figures made previously and let the user choose :   
- to keep the raw trajectory points,  
- to keep the interpolated trajectory 
- to rejected both. In this case this trial success will be set to false and will not be use in the future analysis
output :   
It update the metadata.joblib file with the trial_success  


### 6. `preprocessing_figure`   

This script is here to analyse the interpolation process, and make distribution figures



### 7. `compute_metrics`  

Computes kinematics metrics using the predicted points : 
- open the metadata files
- verify if the validation has been made, if not it will tell the user and stop
- then it will compute several kinematics metrics that will be stored in the metadata file of the specific bodyparts. Only the successful trials will be used    

output metadata : 
- average_velocity : computed from the pad off to the end of the laser
- peak_velocity : computed from the pad off to the end of the laser
- tortuosity : computed from the pad off to the end of the laser
- instant_velocity : velocity across time (for the whole trial)
- acceleration : acceleration across time (for the whole trial)
- xy_pad_off : points from the pad off to the end of the laser
- xy_laser_off : points from the beginning of the laser to it's end (can be None if no Laser)
- xy_reward : points from the beginning of the trial to the reward time (can be None if no Reward observed)


### 8. `trial_report`

This script generates summary figures showing the proportion of :   
- successful trial
- rejected trial, and inside those rejected trial see the proportion for each reason :   
    - No reward
    - No pad off
    - Rejected by user
All for each experimental condition



### 9. `make_comparative_figures` 

Make figures in order to compare each condition
The availables plotting function can in found in `rat_kinematics_utils/plot_comparative`


### 9. `make_inter_rats_figures` 

Make figure in order to compare rats
The availables plotting function can in found in `rat_kinematics_utils/plot_comparative`


### 9. `make_single_figures` 

Make figures of each trials  
The availables plotting function can in found in `rat_kinematics_utils/plot`


### 10. `trajectory_clustering` 

Script that experiment on clustering the trajectories based on distance calculations. This calcul is based on the shape and the spacio-temporal properties of the trajectory


## Led signification

- LED 1 : Cue type, auditive stimulus (1 pulse = Left lever, 2 pulses = Right lever),  
- LED 2 : Right Pad (lightned = paw is on the pad),
- LED 3 : Left Pad (Lightned = paw is on the pad),  
- LED 4 : Activation of the optogenetic laser 
- LED 5 : Reward given to the rat  

## Label studio annotation signification

| Rat name                                                                  | H001 - Left       | H002  - Right     |
| --------                                                                  | -------     | -------    |
| Rat_#517Ambidexter_20240523_ContiMT300_LeftHemi_L1L2_C001H001             | 1812        | 1811       |
| Rat_#517Ambidexter_20240527_BetaMT300_RightHemiCHR_L1L25050_C001H001      | 1813        | ....       |
| Rat_#517Ambidexter_20240624_BetaMT300_RightHemiCHR_L1L25050_C001H001      | 1814        | ....       |



## Folder sytem

```bash
-- data/                            # all the FINAL generated data from scripts of src/
    |
    | -- csv_results/               # results from the dlc model prediction (csv and h5)
    | -- dlc_results/
    |
    | -- frames/                    # generated frames for each clip
    | -- clips/                     # generated videos (clips) from the raw video
    |
    | -- temporary/                 # temporary dlc prediction (always empty because it erase the temporary files generated)
    | -- figures_results/           # .png of the trajectory of video (single clip, stacked clips, average clip)
        |
        | -- rat#517/
            |
            |-- Rat_#512_....video_name.../
                |
                | -- clip_annotation/    # trajectory on clip video
                | -- trajectory_per_clip # graphic of the trajectory
                | -- .../
                | 
                | trajectory_stacked.png # stacked trajectory 
                | trajectory_average.png # average trajectory 
                | ...
    
    | -- metrics_results/           # .joblib of all the metadata and computed metrics of each condition
        |                           # necessary to make any kind of figures
        | -- rat#517/
            |
            | rat#525_CHR_Conti_RightHemi_H001_LaserOn_0,5mW.joblib
            | ...


-- exploration/                     # jupyter notebook to explore / test things
    |
    |-- data/                           # generated data of utils files on their own, it s for test only
            |
            | ...

-- src/                             # "official" scripts
    |
    | -- rats_kinematics_utils/     # all utility function used by the main pipeline
    |
    | 1.1.make_prediction.py
    | 1.2.rename_files.py
    |
    | 3.0.preprocessing.py
    | 3.1.preprocess_validation.py
    | 3.2.preprocess_figures.py
    | 3.4.compute_metrics.py
    |  
    | 4.trial_report.py
    |
    | 5.make_comparative_figures.py
    | 5.make_inter_rats_figures.py
    | 5.make_single_figures.py
    | 
    | 6.trajectory_clustering.py
-- 
```
