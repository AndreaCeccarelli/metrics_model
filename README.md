# Object criticality model

### Repository of the paper "Evaluating the Consequences of Object (mis)Detection from a Safety and Reliability Perspective: Discussion and Measures" (submitted) https://arxiv.org/abs/2203.02205

(see also the submitted pdf and the supplemental material in this repo) 

## Abstract and scope of the repo

We argue that object detectors in the safety critical domain should prioritize detection of objects that are most likely to interfere with the actions of the actor, especially when they can impact task safety and reliability.

In the context of autonomous driving, we propose new object detection measures that reward the correct identification of objects that are most likely to interact with the subject vehicle (i.e., the actor), and that can interfere on its driving decision. 

To achieve this, we build a criticality model to reward the detection of the objects based on proximity, orientation, and relative velocity with respect to the target vehicle. 

Then, we apply our model on the recent autonomous driving dataset nuScenes, and we compare eighth different object detectors. Results show that, in several settings, object detectors that perform best according to the nuScenes ranking are not the preferable ones when the focus is shifted on safety and reliability.

This repository contains the source code to apply such model on nuScenes, and reproduce our entire set of results.

## Contents of the repository
- the modified nuScenes library
- notebooks to run our model and collect results

## Instructions to reproduce our setting
The easiest approach is the following:

1- install nuScenes data from https://www.nuscenes.org/ and libraries nuscenes-dev https://github.com/nutonomy/nuscenes-devkit . Our repo has been tested up to Devkit v1.1.7.
2- install mmdetection3d https://mmdetection3d.readthedocs.io/en/latest/ . Latest version should be fine, without problems.
3- download the models weights for nuScenes object detection from the model zoo of mmdetection3d https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html

At this point, if everything has been installed correctly, you should be able to run the notebook MMDetection3D.ipynb (you should just need to adjust paths): you should be able to visualize results of the object detector of your choice. Such notebook allows collecting the data that needs to be processed by nuScenes, so it is important that it runs smoothly. The notebook contains instructions for its execution.

Then:
- in the folder "eval", there are some files that need to be replaced to your nuScene-dev installation. For example, if you installed nuScene libraries with conda, the path is something like: $HOME/anaconda3/envs/mmm/lib/python3.7/site-packages/nuscenes/eval. You can just overwrite files with the ones we provide.

- run the notebook compute_APCRIT.ipynb.  You should just need to adjust paths, then it should run smootly. The notebook contains instructions for its execution.

At this point, if everything is correct, a "results" folders with some files will be created. These files includes the novel metrics discussed in our work. The files are CSV and JSON, and their content is straightforward, but if you have any kind of trouble just ask us for clarification.

If you arrived here, it means you are fully able to obtain the results we presented in our work. You can further play with the notebook, for example to test different parameters and different models. It has the settings we used to produce the data in our paper, but beware as it will take quite some time to complete.

To explore the data, we present the notebook Analysis_of_data.ipynb. It allows extracting the data we present in the paper plus much more.


## Cite this work as:

Andrea Ceccarelli, Leonardo Montecchi, Evaluating the Consequences of Object (mis)Detection from a Safety and Reliability Perspective: Discussion and Measures, https://arxiv.org/abs/2203.02205

