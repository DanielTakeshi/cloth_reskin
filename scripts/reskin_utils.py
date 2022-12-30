"""Put utility files here to support `reskin_classify.py`."""
import os
from os.path import join
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
from sklearn import preprocessing
import matplotlib.pyplot as plt


def print_class_counts(y, data_type):
    y_unique = np.sort(np.unique(y))
    print(f'Unique class labels for y: {y_unique} in {data_type}')
    for k in y_unique:
        print(f'  count {int(k)}: {np.sum(y==k)}')


def get_data_XY_new(dir_name, get_images=False):
    """Extract data.

    Extending this to support video data. We use video capture to read the .avi
    files, which will give us frames for debugging:
        frame = frame[:480, :, :]  # for an image-based classifier.
        frame = frame[480:, :, :]  # magnetometers (keep as future reference).
    The frame count will be lower than the ReSkin counts, so we sub-sample.
    Assumes the frames are spread out 'uniformly' throughout the training. That's
    embedded in the `np.linspace()` which assumes equally spaced intervals.

    For quick and dirty testing:
        import moviepy
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(list(frames), fps=20)
        clip.write_gif('test.gif', fps=20)
    """
    file_name = join(dir_name, 'reskin_data_cmd.csv')  # for feb26+ data
    data = np.loadtxt(file_name, delimiter = ",")
    X = data[:,:-2]
    Y = np.reshape(data[:,-2],(-1,1))

    # Hack to subtract labels for this dataset.
    Y = Y - 1

    if get_images:
        video_name = join(dir_name, 'videos/magnetometers.avi')
        if not os.path.exists(video_name):
            print(f'Warning, ignoring this as video file does not exist:\n{video_name}')
            return (None,None)
        vcap = cv2.VideoCapture(video_name)
        ret = True
        frames = []
        while ret:
            ret, frame = vcap.read()
            if frame is not None:
                frames.append(frame)
        X = np.array(frames)  # (n_frames, 960, 640, 3)
        subsamp = np.linspace(start=0, stop=len(Y)-1, num=len(frames))
        subsamp = subsamp.astype(np.int32)
        Y = Y[subsamp, :]   # Y.shape --> (n_frames, 1)
    return X,Y


def get_cloth_dir_names(classes):
    cloth0_dir_names = []
    cloth1_dir_names = []
    cloth2_dir_names = []
    cloth3_dir_names = []

    if 0 in classes:
        p0 = classes[0]
        for dir in sorted(os.listdir(p0)):
            cloth0_dir_names.append(p0+"/"+dir)
    if 1 in classes:
        p1 = classes[1]
        for dir in sorted(os.listdir(p1)):
            cloth1_dir_names.append(p1+"/"+dir)
    if 2 in classes:
        p2 = classes[2]
        for dir in sorted(os.listdir(p2)):
            cloth2_dir_names.append(p2+"/"+dir)
    if 3 in classes:
        p3 = classes[3]
        for dir in sorted(os.listdir(p3)):
            cloth3_dir_names.append(p3+"/"+dir)

    return (cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, cloth3_dir_names)


def get_dataset_dir_names(ntrain, cloth0, cloth1, cloth2, cloth3=[]):
    """Get the directory names; it considers that we might shuffle directories.

    Note: adjusting ntrain is now a dict. Should work even if cloth3 is empty.
    We're using the first ntrain for train, but that's because we assume we
    already shuffled the `clothX` lists beforehand.
    """
    train_dir_names = cloth0[ :ntrain[0] ] + \
                      cloth1[ :ntrain[1] ] + \
                      cloth2[ :ntrain[2] ] + \
                      cloth3[ :ntrain[3] ]
    test_dir_names = cloth0[ ntrain[0]: ] + \
                     cloth1[ ntrain[1]: ] + \
                     cloth2[ ntrain[2]: ] + \
                     cloth3[ ntrain[3]: ]
    return train_dir_names, test_dir_names


def get_raw_data(train_dir_names, test_dir_names, get_images):
    """Forms the raw (X,Y,x,y) data (train upper, test lower).

    Extending this to support a `get_images` method for video data, by supplying it
    as an argument to `get_data_XY_new()`. We still use the same 'vstack'-ing, etc.
    """
    X = []
    Y = []
    print('get_raw_data(), collecting train data...')
    for dir_name in train_dir_names:
        # Each of these episodes has 2 labels, (a) grasping nothing, and (b) the
        # label of {0,1,2} cloth, where 0 here means ReSkin touches the other tip.
        x,y = get_data_XY_new(dir_name, get_images=get_images)
        if (x is None) or (y is None):
            continue
        X.append(x)
        Y.append(y)
    if len(X) > 0:
        X = np.vstack(X)
        Y = np.vstack(Y)

    x_test = []
    y_test = []
    info = []
    print('get_raw_data(), collecting valid data...')
    for dir_name in test_dir_names:
        x,y = get_data_XY_new(dir_name, get_images=get_images)
        if (x is None) or (y is None):
            continue
        x_test.append(x)
        y_test.append(y)
        data_info = (len(y), dir_name)
        info.append(data_info)
    if len(x_test) > 0:
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
    return X, Y, x_test, y_test


def normalize_data(X, x_test=None, return_scaler=False):
    """Fit only the train data (important!).

    Adjusting to allow for no test (in case we train on all) as well as returning
    the scaler so we can save and load it later. See scikit docs for more info.
    """
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    if x_test is not None:
        x_test = scaler.transform(x_test)
    if return_scaler:
        return (X, x_test, scaler)
    else:
        return (X, x_test)


def get_dataset_from_dir_names(cloth0_dir_names=[], cloth1_dir_names=[],
        cloth2_dir_names=[], cloth3_dir_names=[], train_frac=0.8, get_images=False,
        test_drift=0):
    """Get the data for machine learning.

    Provides some partial support for testing drift, which we used for checking if
    the data changed over time. But by default this is turned off, `test_drift=0`.
    """
    nepis = {
        0: len(cloth0_dir_names),
        1: len(cloth1_dir_names),
        2: len(cloth2_dir_names),
        3: len(cloth3_dir_names),
    }
    ntrain = {
        0: int(train_frac * len(cloth0_dir_names)),
        1: int(train_frac * len(cloth1_dir_names)),
        2: int(train_frac * len(cloth2_dir_names)),
        3: int(train_frac * len(cloth3_dir_names)),
    }

    # Normally we WANT to shuffle the episodic data (i.e., `test_drift = 0`).
    if test_drift == 0:
        ss_c0 = np.random.permutation(nepis[0])
        ss_c1 = np.random.permutation(nepis[1])
        ss_c2 = np.random.permutation(nepis[2])
        ss_c3 = np.random.permutation(nepis[3])
    else:
        ss_c0 = np.arange(nepis[0])
        ss_c1 = np.arange(nepis[1])
        ss_c2 = np.arange(nepis[2])
        ss_c3 = np.arange(nepis[3])
    tr_c0, te_c0 = ss_c0[ :ntrain[0] ], ss_c0[ ntrain[0]: ]
    tr_c1, te_c1 = ss_c1[ :ntrain[1] ], ss_c1[ ntrain[1]: ]
    tr_c2, te_c2 = ss_c2[ :ntrain[2] ], ss_c2[ ntrain[2]: ]
    tr_c3, te_c3 = ss_c3[ :ntrain[3] ], ss_c3[ ntrain[3]: ]
    print(f'Train/Test epis (c0):\n  {tr_c0}\n  {te_c0}\n  .. tot {len(tr_c0)+len(te_c0)}')
    print(f'Train/Test epis (c1):\n  {tr_c1}\n  {te_c1}\n  .. tot {len(tr_c1)+len(te_c1)}')
    print(f'Train/Test epis (c2):\n  {tr_c2}\n  {te_c2}\n  .. tot {len(tr_c2)+len(te_c2)}')
    print(f'Train/Test epis (c3):\n  {tr_c3}\n  {te_c3}\n  .. tot {len(tr_c3)+len(te_c3)}')
    cloth0_shuffled = []
    cloth1_shuffled = []
    cloth2_shuffled = []
    cloth3_shuffled = []
    for idx in range(nepis[0]):
        cloth0_shuffled.append( cloth0_dir_names[ss_c0[idx]] )
    for idx in range(nepis[1]):
        cloth1_shuffled.append( cloth1_dir_names[ss_c1[idx]] )
    for idx in range(nepis[2]):
        cloth2_shuffled.append( cloth2_dir_names[ss_c2[idx]] )
    for idx in range(nepis[3]):
        cloth3_shuffled.append( cloth3_dir_names[ss_c3[idx]] )

    # Since we shuffled these, we can just take first ntrain for training.
    train_dir_names, test_dir_names = get_dataset_dir_names(ntrain,
            cloth0_shuffled, cloth1_shuffled, cloth2_shuffled, cloth3_shuffled)
    print(f'len(train_dir_names): {len(train_dir_names)}')
    print(f'len(test_dir_names):  {len(test_dir_names)}')
    print(f'total:  {len(train_dir_names)+len(test_dir_names)}')

    # Now we can get the raw data. DO NOT NORMALIZE, do that later!
    X, Y, x, y = get_raw_data(train_dir_names, test_dir_names, get_images)

    # Return train, test, info.
    return (X, Y, x, y)


# Debugging methods -- not used for learning but could be convenient.

def plot_magnetometer(data):
    """Shows all magnetometer readings as a function of time (x-axis)."""
    fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
    names = {0: "Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    for i in range(5):
        axes[i].plot(data[:,i*3],   'r', label="Bx")
        axes[i].plot(data[:,i*3+1], 'g', label="By")
        axes[i].plot(data[:,i*3+2], 'b', label="Bz")
        axes[i].set_title(names[i])
    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    fig.set_size_inches(20, 11.3)
    plt.show()


def plot_diff(data):
    fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
    names = {0: "Center", 1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
    dx = 0.001
    for i in range(5):
        axes[i].plot(np.diff(data[:,i*3])/dx,   'r', label="Bx")
        axes[i].plot(np.diff(data[:,i*3+1])/dx, 'g', label="By")
        axes[i].plot(np.diff(data[:,i*3+2])/dx, 'b', label="Bz")
        axes[i].set_title(names[i])
    lines, labels = fig.axes[-2].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize =12.0)
    fig.tight_layout(pad=0.5)
    fig.text(0.04, 0.5, 'Magnetometer Data', va='center', rotation='vertical', fontsize = 14.0)
    fig.set_size_inches(20, 11.3)
    plt.show()


def plot_norm(data):
    """Squares the data, then plots _differences_ across consecutive time steps."""
    fig = plt.figure()
    sqdata = np.square(data)
    norm_data = np.diff(np.mean(sqdata, axis = 1))
    plt.plot(norm_data, 'r', label="norm")
    fig.set_size_inches(20, 11.3)
    plt.show()
