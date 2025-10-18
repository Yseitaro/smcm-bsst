# Adaptive EMG Pattern Classification via Probabilistic Knowledge Transfer With Scale Mixture-Based Bayesian Sequential Learning

---

This is the implementation of our paper: [Adaptive EMG Pattern Classification via Probabilistic Knowledge Transfer With Scale Mixture-Based Bayesian Sequential Learningk](https://ieeexplore.ieee.org/abstract/document/11079723).

## Overview
 <img src=img/overview_ver7-1.png><br>
 **Fig. 1.** Overview of the proposed adaptive EMG pattern classification method<br>
 <img src=img/table.png><br>
## folder structure
```
.
├── config
│   ├── settings_furui_lab.yaml
│   ├── settings_kanoga_long.yaml
│   └── settings_khushaba1.yaml
├── data
├── experiments
│   ├── accuracy_by_motion.py
│   ├── accuracy_by_threshold.py
│   ├── accuracy_nu_opt.py
│   ├── accuracy.py
│   ├── ece_by_motion.py
│   ├── ece_by_trial_subject.py
│   ├── ece.py
│   ├── false_label_rate.py
│   ├── memory_size.py
│   ├── rejection_rate.py
│   └── svm_tuning.py
├── img
│   ├── overview_ver7.pdf
│   └── table.png
├── notebooks
├── README.md
├── reports
├── results
└── src
    ├── __init__.py
    ├── data
    │   ├── data_utils.py
    │   └── process_emg.py
    ├── models
    │   ├── adaptive_pc.py
    │   ├── alda.py
    │   ├── asvm.py
    │   ├── bayes_gmm.py
    │   ├── bayesian_fmsmm.py
    │   ├── cnn.py
    │   ├── mlp.py
    │   ├── optimization_nu.py
    │   └── sklearn_qda.py
    └── utils
```
## Getting Started
### Install Requriements
Create a python 3.10.0 environment, e.g.:<br>
```
python3.10.0 -m venv venv
. venv/bin/activate
```
Install requirements with pip.<br>
```
pip install -r requirements.txt
```

### How to run
```
# Compute classification accuracy
python experiments/accuracy.py
# Compute ECE
python experiments/ece.py
```
