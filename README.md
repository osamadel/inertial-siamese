# Inertial Gait-based Person Authentication Using Siamese Networks

This repository is the official implementation of "Inertial Gait-based Person Authentication Using Siamese Networks" (IJCNN 2021). In this paper, we propose a Siamese Network-based framework for inertial gait-based person authentication. Our proposed framework allows for learning a model on a set of subjects and being tested on a totally different set of subject (authenticate new subjects) without retraining the model from scratch.

## Files Structure
The file structure is quite straightforward. We do experiments on three publicly available datasets: OU-ISIR, MMUISD and EJUST-GINR1 and hence, we made an experiment notebook for each dataset in the `src` directory. Two more experiments are available, training a different classifier on top of the feature extractor (instead of the fully connected layer in the Siamese Network) and transfer learning. You will find a separate notebook for the first experiment for each dataset (again) and a single experiment notebook for the transfer learning.

Moreover, three utility files exist. 
1. `analyze.py`: has some code to evaluate and analyze a model in terms of the False Rejection Rate (FRR) and False Acceptance Rate (FAR), and also return these values for different thresholds to plot a ROC curve. 
2. `datasets.py` have the bulk of the code of loading the datasets, pre-processing them and generating the pairwise inputs to the Siamese Network.
3. `models.py` has a number of function that implement different variations of our model.

## Datasets

Three datasets are used in this work:

1. OU-ISIR: Can be found from [the this link.](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html)
2. MMUISD: Can be found from [the this link.](https://github.com/ceciljc/MMUISD)
3. EJUST-GINR1: Can be found from [the this link.](https://sites.google.com/ejust.edu.eg/walid-gomaa/projects/robust-wearable-activity-recognition-system-based-on-imu-signals/data/ejust-ginr-1?authuser=0)

## Cite

```
@misc{siamese ,
author = {Adel, Osama and Soliman, Mostafa and Gomaa, Walid},
title = {Inertial Gait-based Person Authentication Using Siamese Networks},
booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
note = {(in press)}
}
```
