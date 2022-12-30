"""ReSkin classification with scipy."""
import cv2
import os
import sys
from os.path import join
import shutil
import argparse
import time
import numpy as np
from joblib import dump
from collections import defaultdict
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
np.set_printoptions(suppress=True, precision=4, linewidth=120, edgeitems=100)
import reskin_utils as U

# ----------------------------------------------------------------------- #
HEAD =  '/data/seita/tactile_cloth'
RESULTDIR = 'results'
RESKIN_EXP = 'finetune_noisymanual_clean'
CLASSES = {
    0: join(HEAD, RESKIN_EXP, '0cloth_norub_auto_robot'),
    1: join(HEAD, RESKIN_EXP, '1cloth-openloop'),
    2: join(HEAD, RESKIN_EXP, '2cloth-openloop'),
    3: join(HEAD, RESKIN_EXP, '3cloth-openloop'),
}
# ----------------------------------------------------------------------- #


def classify_reskin(cloth0_dir_names, cloth1_dir_names, cloth2_dir_names,
        cloth3_dir_names=[], n_folds=1, debug_print=True):
    """Classify ReSkin data from physical data collection.

    This iterates over the number of folds, where each involves some assignment of
    training versus validation episodes.
    """
    stats = defaultdict(list)
    stats['cmat'] = np.zeros( (args.NUM_CLASSES, args.NUM_CLASSES) )
    start_t = time.time()

    for i in range(n_folds):
        print(f'\n|------------------------- Fold {i+1} of {n_folds} --------------------------|')
        # Randomize train / test for this fold. CAREFUL! Do not override X, Y, x, y.
        # NOTE(daniel): data is NOT YET normalized!
        X_train_orig, y_train_orig, X_test_orig, y_test_orig = U.get_dataset_from_dir_names(
                cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, cloth3_dir_names,
                train_frac=args.train_frac)
        if args.train_all:
            X_test_orig = None
            y_test_orig = None
        if debug_print:
            print(f'Train (X,Y): {X_train_orig.shape}, {y_train_orig.shape}')
            if not args.train_all:
                print(f'Test  (x,y): {X_test_orig.shape}, {y_test_orig.shape}')
                print(f'Total train/test: {len(y_train_orig) + len(y_test_orig)}')
            U.print_class_counts(y_train_orig, data_type='train')
            if not args.train_all:
                U.print_class_counts(y_test_orig, data_type='valid')

        if args.method == 'rf':
            clf = ensemble.RandomForestClassifier(max_depth=None, random_state=0)
        elif args.method == 'knn':
            clf = neighbors.KNeighborsClassifier(10, weights="distance")
        elif args.method == 'svm':
            clf = svm.SVC(kernel='rbf')
        elif args.method == 'lr':
            clf = linear_model.LogisticRegression(penalty='l2', max_iter=250)

        # Scipy / sklearn uses a consistent API, fortunately. Train on FULL data.
        X_train_all, X_test_all, scaler = U.normalize_data(
            X_train_orig, X_test_orig, return_scaler=True)
        clf.fit(X_train_all, y_train_orig.ravel())

        # Save the joblib of the classifier AND scaler, in case we load it later.
        if args.train_all:
            N = X_train_orig.shape[0]
            clf_fname = join(RESULTDIR,
                f'clf_{args.method}_alldata_size_{N}_nclasses_{args.NUM_CLASSES}.joblib')
            dump(clf, clf_fname, protocol=2)
            s_fname = join(RESULTDIR,
                f'scaler_{args.method}_alldata_size_{N}_nclasses_{args.NUM_CLASSES}.joblib')
            dump(scaler, s_fname, protocol=2)
            print(f'Look at: {clf_fname} and {s_fname}')
            sys.exit()

        # Now predict!
        y_pred_all = clf.predict(X_test_all)
        y_test_orig = np.squeeze(y_test_orig)
        score = balanced_accuracy_score(y_test_orig, y_pred_all)
        best_cf = confusion_matrix(y_test_orig, y_pred_all)
        print(f'\n[Full data]\nBalanced accuracy score: {score:0.4f}')
        print(best_cf)
        stats['score'].append(score)
        stats['cmat'] += best_cf  # keep ADDING so we can divide

    elapsed_t = time.time() - start_t
    print(f'\n')
    print(f'='*100)
    print(f'Done over {n_folds} different folds!')
    print(f'Elapsed Time (seconds): {elapsed_t:0.3f}')
    print(f'='*100)
    print('Avg Balanced Acc: {:0.4f} +/- {:0.2f}'.format(
            np.mean(stats['score']), np.std(stats['score'])))
    print('Avg confusion matrix:\n{}'.format(stats['cmat'] / n_folds))

    # For each class, # correct / # in real data.
    C = stats['cmat']   # The sum of these confusion matrices.
    for k in range(args.NUM_CLASSES):
        ctot = int(np.sum(C[k,:]))
        print(f'Class {k} avg acc: {int(C[k,k])} / {ctot} = {(C[k,k]/ctot):0.3f}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, default='knn')
    p.add_argument('--n_folds', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--train_all', action='store_true', default=False,
        help='Only use this if we want to train one model on all data (no test).')
    args = p.parse_args()
    np.random.seed(args.seed)

    # Handle data directory stuff.
    if RESKIN_EXP == 'finetune_noisymanual_clean':
        args.NUM_CLASSES = len(CLASSES.keys())
    else:
        raise NotImplementedError()
    for key in sorted(CLASSES.keys()):
        print(f'Cloth dirs {key}: {CLASSES[key]}')

    # Load all ReSkin data paths. This is train and test for all classes.
    cloth0_dir_names, cloth1_dir_names, cloth2_dir_names, cloth3_dir_names = \
            U.get_cloth_dir_names(CLASSES)
    print(f'  len cloth0_dir_names: {len(cloth0_dir_names)}')
    print(f'  len cloth1_dir_names: {len(cloth1_dir_names)}')
    print(f'  len cloth2_dir_names: {len(cloth2_dir_names)}')
    print(f'  len cloth3_dir_names: {len(cloth3_dir_names)}')
    args.n_epis = n_epis = len(cloth0_dir_names)

    # Adjust depending on desired level of training data.
    if args.train_all:
        assert args.n_folds == 1
        args.train_frac = 1.0
    else:
        args.train_frac = 0.8

    print(f'About to classify ReSkin, train_frac: {args.train_frac}')
    print(f'Method: {args.method} with {args.NUM_CLASSES} classes')

    if args.method in ['knn', 'rf', 'svm', 'lr']:
        classify_reskin(cloth0_dir_names,
                        cloth1_dir_names,
                        cloth2_dir_names,
                        cloth3_dir_names,
                        n_folds=args.n_folds)
    else:
        raise ValueError(args.method)
