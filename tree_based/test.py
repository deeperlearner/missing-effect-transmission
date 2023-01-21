import os
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from lifelines.utils import concordance_index as ci_lifelines
import imblearn
import xgboost
import matplotlib.pyplot as plt
from tqdm import tqdm

from logger import get_logger
from mains import Cross_Valid
from parse_config import ConfigParser
from tree_based.bootstrap import bootstrapping
from utils import msg_box, consuming_time


# fix random seeds for reproducibility
SEED = 123
np.random.seed(SEED)

logger = get_logger("test")


def test(config):
    # use last N entries of dataset
    last_N = config["datasets"]["last_N"]
    # datasets
    testset = config.init_obj(["datasets", "train", "data"])

    # metrics
    metrics = config["metrics"]["per_epoch"]
    metrics_pred = {"TPR": recall_score, "PPV": precision_score}
    metrics_prob = {"AUROC": roc_auc_score, "AUPRC": average_precision_score}

    repeat_time = config["cross_validation"]["repeat_time"]
    k_fold = config["cross_validation"]["k_fold"]

    results = pd.DataFrame()
    Cross_Valid.create_CV(repeat_time, k_fold)
    start = time.time()
    for k in range(k_fold):
        # data_loaders
        test_loaders = config.init_obj(
            ["data_loaders", "test", "data"], testset
        )
        test_loader = test_loaders.test_loader

        # models
        if k_fold > 1:
            fold_prefix = f"fold_{k}_"
            dirname = os.path.dirname(config.resume)
            basename = os.path.basename(config.resume)
            resume = os.path.join(dirname, fold_prefix + basename)
        else:
            dirname = os.path.dirname(config.resume)
            resume = config.resume
        # load
        with open(resume, "rb") as f:
            model = pickle.load(f)
        # max_seq_len = testset.max_seq_len

        # testing
        X_test = []
        event_times = []
        y_test = []
        for (data, target) in test_loader:
            # data: (x_num, x_cat, x_num_mask, x_cat_mask)
            # target: (day_delta, group)
            x_num, x_cat, x_num_mask, x_cat_mask = [x.cpu().detach().numpy() for x in data]
            X = np.concatenate([x_num, x_cat, x_num_mask, x_cat_mask], axis=-1)
            day_delta, y = target

            X_test.append(X)
            event_times.append(day_delta)
            y_test.append(y)
        X_test = np.concatenate(X_test)
        event_times = np.concatenate(event_times)
        y_test = np.concatenate(y_test)

#         # pad X_test to consist with X_train
#         seq_pad_len = max_seq_len - X_test.shape[1]
#         pad_width = ((0, 0), (seq_pad_len, 0), (0, 0))
#         X_test = np.pad(X_test, pad_width, constant_values=0)

        if last_N:
            X_test = X_test[:, -last_N:, :]
        N_test = X_test.shape[0]
        X_test = X_test.reshape(N_test, -1)
        y_test = np.squeeze(y_test)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics_result = pd.DataFrame(index=metrics)
        for key, value in metrics_pred.items():
            metrics_result.at[key, ""] = value(y_test, y_pred)
        for key, value in metrics_prob.items():
            metrics_result.at[key, ""] = value(y_test, y_prob)
        # compute c_index
        metrics_result.at["c_index", ""] = ci_lifelines(event_times, -y_prob, y_test)

        test_log = metrics_result[""].rename(k)
        results = pd.concat((results, test_log), axis=1)
        logger.info(test_log)

        if k_fold > 1:
            Cross_Valid.next_fold()

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    result = pd.DataFrame()
    result["mean"] = results.mean(axis=1)
    result["std"] = results.std(axis=1)
    msg += f"\n{result}"

    logger.info(msg)

    # bootstrap
    if config.test_args.bootstrapping:
        assert k_fold == 1, "k-fold ensemble and bootstrap is mutually exclusive."
        N = config.test_args.bootstrap_times
        bootstrapping(event_times, y_test, y_pred, y_prob, metrics_pred, metrics_prob, repeat=N)
