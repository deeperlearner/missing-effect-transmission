import sys
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(sys.path[0], ".."))
from logger import get_logger
from mains import Cross_Valid


class EHRDataset:
    logger = get_logger("dataset")
    # Biomarkers
    NUM_COLS = [
        "ALT",
        "HBVDNA_gp",
        "PLT",
    ]
    NUM_COLS = ["final_" + col for col in NUM_COLS]
    NUM_COLS += [
        "age",
    ]
    CAT_COLS = ["HBeAg"]
    CAT_COLS = ["final_" + col for col in CAT_COLS]
    CAT_COLS += ["sex"]

    def __init__(
        self,
        impute_method="mean",
        compute_method="by_entry",
        do_locf=False,
        data_dir="./data",
        mode="train",
    ):
        """
        impute_method:
            zero: 0/0
            mean: mean/mode
            median: median/mode
            gaussian
            mean_minus_3_std
        compute_method:
            by_entry
            by_patient
        do_locf:
            If true, do forward filling
        """
        self.num_feature = (len(self.NUM_COLS), len(self.CAT_COLS))
        self.impute_method = impute_method
        self.compute_method = compute_method
        self.do_locf = do_locf
        self.mode = mode

        train_path = {
            "x_s": os.path.join(data_dir, "train/x_s.csv"),
            "y": os.path.join(data_dir, "train/y.csv"),
        }
        test_path = {
            "x_s": os.path.join(data_dir, "test/x_s.csv"),
            "y": os.path.join(data_dir, "test/y.csv"),
        }

        train_data = {
            "x_s": pd.read_csv(train_path["x_s"]),
            "y": pd.read_csv(train_path["y"], index_col="pid"),
        }
        test_data = {
            "x_s": pd.read_csv(test_path["x_s"]),
            "y": pd.read_csv(test_path["y"], index_col="pid"),
        }

        # x_seq
        x_num_train = train_data["x_s"][["pid"] + self.NUM_COLS]
        x_cat_train = train_data["x_s"][["pid"] + self.CAT_COLS]
        x_num_test = test_data["x_s"][["pid"] + self.NUM_COLS]
        x_cat_test = test_data["x_s"][["pid"] + self.CAT_COLS]

        self.x_train_df = pd.concat(
            [x_num_train, x_cat_train.drop(columns="pid")], axis=1
        )
        self.y_train = train_data["y"]
        self.y_train["date"] = pd.to_datetime(self.y_train["date"])
        self.y_train["index_date"] = pd.to_datetime(self.y_train["index_date"])
        if mode == "test":
            self.x_test_df = pd.concat(
                [x_num_test, x_cat_test.drop(columns="pid")], axis=1
            )
            self.y_test = test_data["y"]
            self.y_test["date"] = pd.to_datetime(self.y_test["date"])
            self.y_test["index_date"] = pd.to_datetime(self.y_test["index_date"])

    def transform(self, split_idx=None):
        # self.compute_info()
        self.impute()

    def compute_info(self):
        data = self.x_train_df.drop(columns="pid")
        num_dim, cat_dim = self.num_feature
        x_num = data.iloc[:, :num_dim]
        x_cat = data.iloc[:, num_dim:]

        method = self.compute_method
        if method == "by_entry":
            self.num_mean = x_num.mean()
            self.cat_mode = x_cat.mode().loc[0]

        # print(self.num_mean)
        # print(self.cat_mode)
        # os._exit(0)

    def impute(self):
        # Last Observation Carried Forward
        # print(self.x_train_df.head(50))
        if self.mode == "train":
            x_pid = self.x_train_df["pid"]
            self.x_train = self.x_train_df.groupby(by="pid").ffill()
        elif self.mode == "test":
            x_pid = self.x_test_df["pid"]
            self.x_test = self.x_test_df.groupby(by="pid").ffill()
        # print(self.x_test.head(50))
        # os._exit(0)

        # # filling num_fill, cat_fill for remainder
        # method = self.impute_method
        # if method == "zero":
        #     num_fill, cat_fill = 0, 0
        # elif method == "mean":
        #     num_fill, cat_fill = self.num_mean, self.cat_mode
        # elif method == "median":
        #     num_fill, cat_fill = self.num_median, self.cat_mode
        # elif method == "gaussian":
        #     epsilon = np.random.normal()
        #     num_fill, cat_fill = self.num_mean + epsilon*self.num_std, self.cat_mode
        # elif method == "mean_minus_3_std":
        #     num_fill, cat_fill = self.num_mean - 3*self.num_std, self.cat_mode

        num_dim, cat_dim = self.num_feature
        if self.mode == "train":
            self.x_train = self.x_train.dropna()
            x_num = self.x_train.iloc[:, :num_dim]
            x_cat = self.x_train.iloc[:, num_dim:]
        elif self.mode == "test":
            self.x_test = self.x_test.dropna()
            x_num = self.x_test.iloc[:, :num_dim]
            x_cat = self.x_test.iloc[:, num_dim:]

        self.x_data = pd.concat(
            [x_pid, x_num, x_cat], axis=1
        )

    def get_last_entry(self):
        x = self.x_data.groupby(by="pid").tail(1)
        x = x.set_index("pid")
        return x

    def get_label(self):
        y_data = self.y_train
        if self.mode == "test":
            y_data = self.y_test

        target = y_data["group"]
        event_time = (y_data["index_date"] - y_data["date"]).dt.days

        return target, event_time
