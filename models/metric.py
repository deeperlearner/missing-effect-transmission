import logging
from math import sqrt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from lifelines.utils import concordance_index as ci_lifelines
from lifelines import NelsonAalenFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

smooth = 1e-6


class MetricTracker:
    def __init__(self, keys_iter: list, keys_epoch: list, writer=None):
        self.writer = writer
        self.iter_record = pd.DataFrame(
            index=keys_iter,
            columns=[
                "current",
                "sum",
                "square_sum",
                "counts",
                "mean",
                "square_avg",
                "std",
            ],
            dtype=np.float64,
        )
        self.epoch_record = pd.DataFrame(
            index=keys_epoch, columns=["mean"], dtype=np.float64
        )
        self.reset()

    def reset(self):
        for col in self.iter_record.columns:
            self.iter_record[col].values[:] = 0

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.iter_record.at[key, "current"] = value
        self.iter_record.at[key, "sum"] += value * n
        self.iter_record.at[key, "square_sum"] += value * value * n
        self.iter_record.at[key, "counts"] += n

    def epoch_update(self, key, value):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.epoch_record.at[key, "mean"] = value

    def current(self):
        return dict(self.iter_record["current"])

    def avg(self):
        for key, row in self.iter_record.iterrows():
            self.iter_record.at[key, "mean"] = row["sum"] / row["counts"]
            self.iter_record.at[key, "square_avg"] = row["square_sum"] / row["counts"]

    def std(self):
        for key, row in self.iter_record.iterrows():
            self.iter_record.at[key, "std"] = sqrt(
                row["square_avg"] - row["mean"] ** 2 + smooth
            )

    def result(self):
        self.avg()
        self.std()
        iter_result = self.iter_record[["mean", "std"]]
        epoch_result = self.epoch_record
        return pd.concat([iter_result, epoch_result])


###################
# pick thresholds #
###################
THRESHOLD = 0.5


def Youden_J(target, output, beta=1.0, plot=False):
    global THRESHOLD
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        if plot:
            # ROC plotting
            auroc = roc_auc_score(y_true, y_score)
            plt.plot(
                fpr,
                tpr,
                label="ROC curve (area = %0.2f)" % auroc
            )
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="random baseline")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right")
            plt.savefig("fig/ROC.png")
            plt.clf()
            # PRC
            probas_pred = y_score
            ppv, tpr, thresholds = precision_recall_curve(y_true, probas_pred)
            auprc = average_precision_score(y_true, y_score)
            plt.plot(
                tpr,
                ppv,
                label="PR curve (area = %0.2f)" % auprc
            )
            rate = 0.04
            plt.plot([0, 1], [rate, rate], color="navy", linestyle="--", label="random baseline")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("True Positive Rate")
            plt.ylabel("Precision")
            plt.title("Precision recall curve example")
            plt.legend(loc="upper right")
            plt.savefig("fig/PRC.png")
            plt.clf()
            ###############
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
        THRESHOLD = thresholds[np.argmax(beta * tpr - fpr)]
    return THRESHOLD


def F_beta(target, output, beta=1.0):
    global THRESHOLD
    with torch.no_grad():
        y_true = target.cpu().numpy()
        probas_pred = output.cpu().numpy()
        ppv, tpr, thresholds = precision_recall_curve(y_true, probas_pred)
        THRESHOLD = thresholds[np.argmax(beta * tpr - ppv)]
    return THRESHOLD


#############################
# for binary classification #
#############################
# metrics_iter
def binary_accuracy(target, output):
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        correct = 0
        correct += torch.sum(predict == target).item()
    return correct / len(target)


# metrics_epoch
def TPR(event_times, target, output):  # recall
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        y_true = target.cpu().numpy()
        y_pred = predict.cpu().numpy()
        value = recall_score(y_true, y_pred)
    return value


def PPV(event_times, target, output):  # precision
    with torch.no_grad():
        predict = (output > THRESHOLD).type(torch.uint8)
        y_true = target.cpu().numpy()
        y_pred = predict.cpu().numpy()
        value = precision_score(y_true, y_pred)
    return value


def F_beta_score(target, output, beta=1.0):
    recall = TPR(target, output)
    precision = PPV(target, output)
    score = (precision * recall) / (beta ** 2 * precision + recall)
    return score


# AUC
def AUROC(event_times, target, output):
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        value = roc_auc_score(y_true, y_score)
    return value


def AUPRC(event_times, target, output):
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_score = output.cpu().numpy()
        value = average_precision_score(y_true, y_score)
    return value


# concordance index
def c_index(event_times, target, output):
    with torch.no_grad():
        event_times = event_times.cpu().numpy()
        predicted_scores = -output.cpu().numpy()
        event_observed = target.cpu().numpy()
        ci = ci_lifelines(event_times, predicted_scores, event_observed)
    return ci


# Nelson-Aalen cumulative hazard
def NA_curve(event_times, target, predicted_scores, threshold,
             clf=True, model=None):
    logging.getLogger('matplotlib.font_manager').disabled = True
    with torch.no_grad():
        event_times = event_times.cpu().numpy() / 365.25
        target = target.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        high_risk_idx = np.squeeze(predicted_scores >= threshold)
        low_risk_idx = ~high_risk_idx
        model_str = "" if model is None else f" ({model})"

        # high risk
        h_durations = event_times[high_risk_idx]
        h_durations = np.clip(h_durations, 0, 6)
        h_event_observed = target[high_risk_idx]
        naf = NelsonAalenFitter()
        naf.fit(h_durations, h_event_observed, label="high risk" + model_str)
        ax = naf.plot_cumulative_hazard()

        # low risk
        l_durations = event_times[low_risk_idx]
        l_durations = np.clip(l_durations, 0, 6)
        l_event_observed = target[low_risk_idx]
        naf.fit(l_durations, l_event_observed, label="low risk" + model_str)
        naf.plot_cumulative_hazard(ax=ax)

        plt.xlabel("Years since first diagnosis")
        plt.ylabel("Cumulative probability of HCC")
        plt.title("Cumulative hazard function of different HCC risk")

        # number at risk
        # print(h_durations)
        # plt.xticks([])
        bins = np.arange(0, 14, 2)

        high_digi = np.digitize(h_durations, bins)
        values, counts = np.unique(high_digi, return_counts=True)
        high_num = counts.sum() - np.cumsum(counts)
        # print(high_digi)
        # print(counts)
        # print(high_num)

        low_digi = np.digitize(l_durations, bins)
        values, counts = np.unique(low_digi, return_counts=True)
        counts = counts[:-1]
        low_num = counts.sum() - np.cumsum(counts)
        # print(bins)
        # print(values)
        # print(counts)
        # print(low_num)
        # os._exit(0)

        high_num = high_num[:-1]
        print(high_num.shape)
        print(low_num.shape)
        cell_text = np.stack((high_num, low_num))
        the_table = ax.table(
            cellText=cell_text,
            rowLabels=["High risk", "Low risk"],
            colLabels=[" " for i in range(len(high_num))],
            cellLoc='center',
            # bbox=[0,-0.65,1,0.65],
            edges='open',
        )
        the_table.scale(1, 2.5)
        ax.xaxis.labelpad = 60
        # print(bins)
        # print(high_num)

        plt.tight_layout()
        plt.savefig("fig/NA_plot.png")
        if clf:
            plt.clf()


# Kaplan-Meier Estimate
def KM_curve(event_times, target, predicted_scores, threshold,
             clf=True, model=None):
    logging.getLogger('matplotlib.font_manager').disabled = True
    with torch.no_grad():
        event_times = event_times.cpu().numpy() / 365.25
        target = target.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        high_risk_idx = np.squeeze(predicted_scores >= threshold)
        low_risk_idx = ~high_risk_idx
        model_str = "" if model is None else f" ({model})"

        # high risk
        h_durations = event_times[high_risk_idx]
        h_durations = np.clip(h_durations, 0, 6)
        h_event_observed = target[high_risk_idx]
        h_kmf = KaplanMeierFitter()
        ax = h_kmf.fit(h_durations, h_event_observed, label="high risk" + model_str).plot_survival_function()

        # low risk
        l_durations = event_times[low_risk_idx]
        l_durations = np.clip(l_durations, 0, 6)
        l_event_observed = target[low_risk_idx]
        l_kmf = KaplanMeierFitter()
        ax = l_kmf.fit(l_durations, l_event_observed, label="low risk" + model_str).plot_survival_function(ax=ax)

        plt.title("Survival function of since first diagnosis")

        add_at_risk_counts(l_kmf, h_kmf, ax=ax)

        plt.tight_layout()
        plt.savefig("fig/KM_plot.png")
        if clf:
            plt.clf()


#################################
# for multiclass classification #
#################################
def accuracy(target, output):
    with torch.no_grad():
        predict = torch.argmax(output, dim=1)
        assert predict.shape[0] == len(target)
        correct = 0
        correct += torch.sum(predict == target).item()
    return correct / len(target)


def top_k_acc(target, output, k=3):
    with torch.no_grad():
        predict = torch.topk(output, k, dim=1)[1]
        assert predict.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(predict[:, i] == target).item()
    return correct / len(target)


def mean_iou_score(target, output):
    """
    Compute mean IoU score over 6 classes
    """
    with torch.no_grad():
        predict = torch.argmax(output, dim=1)
        predict = predict.cpu().numpy()
        target = target.cpu().numpy()
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(predict == i)
            tp_fn = np.sum(target == i)
            tp = np.sum((predict == i) * (target == i))
            iou = (tp + smooth) / (tp_fp + tp_fn - tp + smooth)
            mean_iou += iou / 6
    return mean_iou
