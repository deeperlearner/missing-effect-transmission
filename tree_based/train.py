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

from logger import get_logger
from mains import Cross_Valid
from utils import msg_box, consuming_time


def train(config):
    config.set_log()
    logger = get_logger("train")

    # use last N entries of dataset
    last_N = config["datasets"]["last_N"]
    # datasets
    dataset = config.init_obj(["datasets", "train", "data"])
    if config["datasets"].get("imbalanced", False):
        target = dataset.y_train["group"]  # TODO

    # metrics
    metrics = config["metrics"]["per_epoch"]
    metrics_pred = {"TPR": recall_score, "PPV": precision_score}
    metrics_prob = {"AUROC": roc_auc_score, "AUPRC": average_precision_score}

    repeat_time = config["cross_validation"]["repeat_time"]
    k_fold = config["cross_validation"]["k_fold"]

    results = pd.DataFrame()
    Cross_Valid.create_CV(repeat_time, k_fold)
    start = time.time()
    for t in range(repeat_time):
        if k_fold > 1:  # cross validation enabled
            dataset.split_cv_indexes(k_fold)
        for k in range(k_fold):
            # data_loaders
            kwargs = {}
            # stratify_by_labels
            kwargs.update(stratify_by_labels=target)
            loaders = config.init_obj(
                ["data_loaders", "train", "data"], dataset, **kwargs
            )
            train_loader = loaders.train_loader
            valid_loader = loaders.valid_loader

            # training data
            X_train = []
            y_train = []
            for (data, target) in train_loader:
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                x_num, x_cat, x_num_mask, x_cat_mask = [x.cpu().detach().numpy() for x in data]
                X = np.concatenate([x_num, x_cat, x_num_mask, x_cat_mask], axis=-1)
                day_delta, y = target

                X_train.append(X)
                y_train.append(y)
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)

            if last_N:
                X_train = X_train[:, -last_N:, :]
            N_train = X_train.shape[0]
            X_train = X_train.reshape(N_train, -1)
            y_train = np.squeeze(y_train)

            # models
            TREE_TYPE = config["name"]
            model = config.init_obj(["models", TREE_TYPE])
            model.fit(X_train, y_train)
            # save
            dirpath = config.save_dir["model"]
            fold_prefix = f"fold_{k}_" if k_fold > 1 else ""
            filename = f"{fold_prefix}model_best.pth"
            with open(os.path.join(dirpath, filename), "wb") as f:
                pickle.dump(model, f)

            # validation data
            X_valid = []
            event_times = []
            y_valid = []
            for (data, target) in valid_loader:
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                x_num, x_cat, x_num_mask, x_cat_mask = [x.cpu().detach().numpy() for x in data]
                X = np.concatenate([x_num, x_cat, x_num_mask, x_cat_mask], axis=-1)
                day_delta, y = target

                X_valid.append(X)
                event_times.append(day_delta)
                y_valid.append(y)
            X_valid = np.concatenate(X_valid)
            event_times = np.concatenate(event_times)
            y_valid = np.concatenate(y_valid)

            if last_N:
                X_valid = X_valid[:, -last_N:, :]
            N_valid = X_valid.shape[0]
            X_valid = X_valid.reshape(N_valid, -1)
            y_valid = np.squeeze(y_valid)

            y_pred = model.predict(X_valid)
            y_prob = model.predict_proba(X_valid)[:, 1]
            metrics_result = pd.DataFrame(index=metrics)
            for key, value in metrics_pred.items():
                metrics_result.at[key, ""] = value(y_valid, y_pred)
            for key, value in metrics_prob.items():
                metrics_result.at[key, ""] = value(y_valid, y_prob)
            # compute c_index
            metrics_result.at["c_index", ""] = ci_lifelines(event_times, -y_prob, y_valid)

            valid_log = metrics_result[""].rename(k)
            results = pd.concat((results, metrics_result), axis=1)
            logger.info(valid_log)

            if k_fold > 1:
                Cross_Valid.next_fold()

        if repeat_time > 1:
            Cross_Valid.next_time()

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    result = pd.DataFrame()
    result["mean"] = results.mean(axis=1)
    result["std"] = results.std(axis=1)
    msg += f"\n{result}"

    logger.info(msg)

    return result
