# Medical Diagnosis and Digit Classification in MATLAB

Four MATLAB experiments that each build a classifier or regressor from its cost function and gradient, then train it with a classical optimizer: least squares, Newton's method, gradient descent and BFGS. No ML toolboxes are used.

The two main results:

- **Breast cancer diagnosis:** L2-regularized logistic regression on the Wisconsin Diagnostic Breast Cancer data (569 patients, 30 features), trained with gradient descent. **98.59%** test accuracy, with 4 errors in 284 patients.
- **Handwritten digits:** 10-class softmax regression on MNIST trained with BFGS, comparing raw 784-pixel inputs against HOG features. **91.84% → 98.04%** test accuracy on 10,000 digits when switching to HOG.

## Table of Contents

- [File Structure](#file-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Running the Code](#running-the-code)
- [External Dependencies](#external-dependencies)

## File Structure

```
ml-diagnosis-digit-classification/
├── README.md
├── 01-linear-regression-mpg/
│   ├── mpg_linear_regression.m  # Least-squares fuel-economy regression (pinv), train/test RMSE
│   └── D_mpg.mat                # Auto MPG: 6 features x 392 cars
├── 02-iris-logistic-newton/
│   ├── iris_newton_one_vs_all.m # One-vs-all logistic regression with Newton's method, K = 1, 3, 5
│   └── X_iris.mat               # Iris: 4 features x 150 flowers, 3 classes
├── 03-breast-cancer-logistic/
│   ├── wdbc_logistic_gd.m       # Driver: normalize, train 4 (mu, K) settings, confusion matrices
│   ├── f_wdbc.m                 # Regularized logistic cost
│   ├── g_wdbc.m                 # Its gradient
│   ├── grad_desc_mod.m          # Gradient descent for exactly K iterations, with backtracking
│   ├── D_wdbc.mat               # WDBC: 30 features + label x 569 patients
│   └── wdbc_output.txt          # Logged run: weights, costs, test results
└── 04-mnist-softmax-hog/
    ├── mnist_softmax_raw_vs_hog.m        # Raw-pixel vs HOG softmax regression, accuracy + throughput
    ├── mnist_softmax_raw_vs_hog_short.m  # Condensed variant of the same comparison
    └── mnist_output.txt                  # Logged run: loss per BFGS iteration, confusion matrices
```

## Experiments

### 01: Linear regression (Auto MPG)
Augments the 6 features with a bias row, solves the least-squares weights with the pseudo-inverse on 314 training cars, and reports RMSE on the remaining 78. It also plots predicted against true MPG for the test set.

### 02: Iris with Newton's method
Trains three one-vs-all logistic classifiers (35 training flowers per class) with Newton's method for K = 1, 3 and 5 iterations. Each weight vector is normalized by ‖w‖ so the three scores are comparable, and each flower goes to the class with the highest score. It reports confusion matrices and accuracy on both training and test data.

### 03: Breast cancer diagnosis
Splits the data into 285 training and 284 test patients. Each feature is z-scored using the training set's mean and standard deviation only. Then it minimizes

```
f(w) = (1/P) Σ log(1 + exp(-y_p · ŵᵀx̂_p)) + (μ/2)‖ŵ‖²
```

with gradient descent and a backtracking line search, for a fixed K iterations. It compares μ ∈ {0, 0.1} at K = 10 and μ ∈ {0, 0.075} at K = 30 to show how regularization affects test error.

### 04: MNIST softmax regression, raw pixels vs HOG
Trains on 16,000 digits (1,600 per class) and tests on the full 10,000-digit MNIST test set. Each 28×28 image is either used as raw pixels or turned into a HOG descriptor (`hog20` with d = 7 and B = 9 orientation bins). A softmax regression model is trained on each version with BFGS: μ = 0.002 for 62 iterations on raw pixels, μ = 0.001 for 57 iterations on HOG. The script reports a 10×10 confusion matrix, accuracy, training time and classification throughput.

## Results

**Breast cancer (WDBC) test set, 284 patients**

| μ | K | Test accuracy | Misclassified | False positives | False negatives |
|---|---|---|---|---|---|
| 0 | 10 | 97.54% | 7 | 4 | 3 |
| 0.1 | 10 | 98.24% | 5 | 1 | 4 |
| 0 | 30 | 97.54% | 7 | 2 | 5 |
| **0.075** | **30** | **98.59%** | **4** | **0** | **4** |

Regularization helped in both cases. Without it, running more iterations lowered the training cost (0.091 → 0.063) but didn't improve test accuracy.

**MNIST test set, 10,000 digits**

| Features | Test accuracy | Final loss | Training time |
|---|---|---|---|
| Raw pixels (784) | 91.84% | 0.384 | 84.4 s |
| HOG | **98.04%** | 0.200 | 57.9 s |

HOG features gave about 6 points more accuracy and a lower final loss, and trained faster.

## Running the Code

Open MATLAB in an experiment folder and run its script, for example `wdbc_logistic_gd`. Each script loads its `.mat` file from the current folder.

## External Dependencies

These helper functions and data files were written by others and aren't redistributed here. Put them on the MATLAB path to run experiments 02–04:

- `bt_lsearch2019.m`: backtracking line search (02–04)
- `LRBC_newton.m`: Newton's method for logistic regression (02)
- `SRMCC_bfgsML.m`, `f_SRMCC.m`, `g_SRMCC.m`: softmax regression with BFGS (04)
- `hog20.m`: HOG descriptor from Ludwig et al., *Trainable Classifier-Fusion Schemes: An Application to Pedestrian Detection*, ITSC 2009 (04)
- `X1600.mat`, `Te28.mat`, `Lte28.mat`: the MNIST training subset and test set as MATLAB matrices (about 11 MB). They're built from the public [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The other three datasets are public UCI datasets: [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), [Iris](https://archive.ics.uci.edu/dataset/53/iris) and [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
