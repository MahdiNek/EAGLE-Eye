
# EAGLE-Eye

This repository contains the code and the dataset accompanying the paper [*"EAGLE-Eye: Extreme-pose Action Grader using detaiL bird’s-Eye view
", WACV 2021*](https://openaccess.thecvf.com/content/WACV2021/html/Nekoui_EAGLE-Eye_Extreme-Pose_Action_Grader_Using_Detail_Birds-Eye_View_WACV_2021_paper.html).

<p align="center">
    <img src="https://github.com/MahdiNek/EAGLE-Eye/blob/main/Pipeline.gif">
    <br>
</p>

### Dataset of Extreme Poses
Visit these shared drive links to download the [*ExPose*](https://drive.google.com/drive/folders/1HQDMIbbwHWerr8AXfPf08K1cwR-G1z7Y?usp=sharing) and the [*G-ExPose*](https://drive.google.com/drive/folders/1sStYPEtPnggp0mI5VrCwzyg5qtk2c39u?usp=sharing) datasets.

### Dependencies
This code requires the following:
* python 2.7.15+ or python 3.5+
* PyTorch v1.0+

### Instructions
First, download the [*AQA-7*](http://rtis.oit.unlv.edu/datasets.html) dataset samples and put the raw videos under `AQA_dataset/SportName_raw`. Then run the `AQA_dataset/video2img.py` to extract the frames of the videos. By default, the appearance dynamics assessment stream of the network uses the I3D features of these frames.\  EAGLE-Eye further requires the pose features of a routine to assess it. We entangled the [*DiMP*](https://github.com/visionml/pytracking) visual object tracker with [*HRNet*](https://github.com/HRNet/HRNet-Human-Pose-Estimation) pose estimator to get the pose sequence of the videos. The resulted pose heatmaps should be located under `AQA_dataset/SportName_heatmaps`. Finally, run the `AQA_head/experiments/train_model_AQA.py` to train and test the network on the *AQA-7* dataset. You may adjust the network parameters on `AQA_head/configs/config_py.py`. 

### Citation

```bibtex
@InProceedings{Nekoui_2021_WACV,
    author    = {Nekoui, Mahdiar and Cruz, Fidel Omar Tito and Cheng, Li},
    title     = {EAGLE-Eye: Extreme-Pose Action Grader Using Detail Bird's-Eye View},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {394-402}
}
```
