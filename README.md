# Singulating Layers of Cloth with a ReSkin Sensor: Classification Code

This is for our IROS 2022 paper on tactile sensing for cloth manipulation
**([arXiv link][3], [project website][4])**. We used this code for the kNN
classifier experiments reported in the paper. If you have questions about the
code, please use the GitHub issues tracker.

Contents:

- [Installation](#installation)
- [The Data](#the-data)
- [Training Models](#training-models)
- [Loading Models](#loading-models)
- [Citation](#citation)

## Installation

```
conda create --name cloth-tactile python=3.8 -y
conda activate cloth-tactile
conda install ipython -y
pip install scikit-learn
pip install matplotlib
pip install moviepy
pip install opencv-python
```

## The Data

For the IROS 2022 paper, the data we used is named and structured as follows:

```
seita@takeshi:/data/seita/tactile_cloth$ ls -lh finetune_noisymanual_clean
drwxrwxr-x 29 seita seita 4.0K Feb 27 23:00 0cloth_norub_auto_robot
drwxrwxr-x 14 seita seita 4.0K Feb 27 23:00 1cloth-openloop
drwxrwxr-x 15 seita seita 4.0K Feb 27 23:01 2cloth-openloop
drwxrwxr-x 16 seita seita 4.0K Feb 27 23:00 3cloth-openloop
```

For the ReSkin classification experiments, **you can find the data [here as a
tar.gz file][2] (about 5G).** Download and un-tar it to the directory of your
choice. This should be specified near the top of the
`scripts/reskin_classify.py`. Change the `HEAD` (and possibly `RESKIN_EXP`) as
needed.

The four subdirectories correspond to data from the four classes. Classes
{0,1,2,3} indicate that the gripper is closed and grasping {0,1,2,3} layers of
cloth, respectively. Each directory within those subdirectories is a single
training episode. For example, this directory:

```
seita@takeshi:/data/seita/tactile_cloth$ ls -lh finetune_noisymanual_clean/2cloth-openloop/2022-02-26-17-37-51_attempt0/
-rw-rw-r-- 1 seita seita  47K Feb 27 23:01 classifier_pred.png
-rw-rw-r-- 1 seita seita 2.6K Feb 27 23:01 config.yaml
-rw-rw-r-- 1 seita seita 161M Feb 27 23:01 data.bag
-rw-rw-r-- 1 seita seita  76K Feb 27 23:01 reskin_data_cmd.csv
-rw-rw-r-- 1 seita seita 916K Feb 27 23:01 reskin_data.csv
drwxrwxr-x 2 seita seita 4.0K Feb 27 23:01 videos
```

contains information about one training episode which involves the robot
grasping two cloth layers. A GIF of the webcam video is in
`videos/magnetometers.gif`.

## Training Models

We support kNN, random forests, logistic regression, or SVMs, though in the
paper we only report kNN. The general command structure is as follows:

```
python scripts/reskin_classify.py --method knn
python scripts/reskin_classify.py --method rf
python scripts/reskin_classify.py --method lr
python scripts/reskin_classify.py --method svm
```

To better understand a classifier's performance, use the `--n_folds` argument.
For example, running:

```
python scripts/reskin_classify.py --method knn --n_folds 100
```

means we get the output:

```
===============================================================================================
Done over 100 different folds!
Elapsed Time (seconds): 42.543
===============================================================================================
Avg Balanced Acc: 0.8357 +/- 0.06
Avg confusion matrix:
[[2643.01    0.      0.      0.  ]
 [   0.04  533.19    0.01    0.69]
 [  16.26    1.81  462.04   53.4 ]
 [  68.42  136.63   73.8   255.6 ]]
Class 0 avg acc: 264301 / 264301 = 1.000
Class 1 avg acc: 53319 / 53393 = 0.999
Class 2 avg acc: 46204 / 53351 = 0.866
Class 3 avg acc: 25560 / 53445 = 0.478
```

The value of 0.8357 +/- 0.06 is what we report in the paper (as 0.84 +/- 0.06)
in the results (Section VI-A) when we ran 100 folds.

If we want one model trained on all the data, which we can use for test-tine
deployment, add `--train_all`:

```
python scripts/reskin_classify.py --method knn --train_all
```

For experiments in the paper, we used a kNN model trained on 95% of the data (as
mentioned in Section V-A).


## Loading Models

To load a model such as the kNN, make sure we originally ran training with
`--train_all` as described above. This will save `.joblib` files in the
directory we assign for the results (by default, `results/`):

```
clf_knn_alldata_size_18838_nclasses_4.joblib
scaler_knn_alldata_size_18838_nclasses_4.joblib
```

To load them, [see this documentation][1] for more information. We need the
scalers as well, for normalization.


## Citation

If you find the code helpful, consider citing the following paper:

```
@INPROCEEDINGS{tirumala2022reskin,
  author={Tirumala, Sashank and Weng, Thomas and Seita, Daniel and Kroemer, Oliver and Temel, Zeynep and Held, David},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Learning to Singulate Layers of Cloth using Tactile Feedback}, 
  year={2022},
  volume={},
  number={},
  pages={7773-7780},
  doi={10.1109/IROS47612.2022.9981341}}
```


[1]:https://scikit-learn.org/stable/modules/model_persistence.html
[2]:https://drive.google.com/file/d/1UEbsFz4v04cDbgAH9a4J8-sZPPXos02z/view?usp=sharing
[3]:https://arxiv.org/abs/2207.11196
[4]:https://sites.google.com/view/reskin-cloth/
