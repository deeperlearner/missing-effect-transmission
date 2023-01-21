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
from utils import ensure_dir


class EHRDataset(Dataset):
    logger = get_logger("dataset")
    # Biomarkers
    Time_series = [
        "time",
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
        "Weight",
    ]
    Static = [
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "Weight",
    ]
    Outcome = [
        "SAPS-I",
        "SOFA",
        "Length_of_stay",
        "Survival",
        "In-hospital_death",
    ]

    def __init__(
        self,
        impute_method="mean",
        cat_type="binary",
        compute_method="by_entry",
        do_locf=False,
        inverse_padding=True,
        truncated_len=None,
        add_emb=False,
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
        cat_type:
            binary: 0/1  # default data
            pos_neg: -1/+1
            one_hot: (1, 0)/(0, 1)
            norm_emb: (x, y) s.t. x^2 + y^2 = 1
        compute_method:
            by_entry
            by_patient
        do_locf:
            If true, do forward filling
        """
        EHRDataset.num_feature = (len(self.Time_series), len(self.Static))
        EHRDataset.impute_method = impute_method
        EHRDataset.cat_type = cat_type
        self.compute_method = compute_method
        self.do_locf = do_locf
        EHRDataset.inverse_padding = inverse_padding
        EHRDataset.truncated_len = truncated_len
        self.add_emb = add_emb
        self.mode = mode

        percent = "0"
        static_path = {
            "set_a": os.path.join(data_dir, f"set-a_static_{percent}.csv"),
            "set_b": os.path.join(data_dir, f"set-b_static_{percent}.csv"),
        }
        time_path = {
            "set_a": os.path.join(data_dir, f"set-a_time_{percent}.csv"),
            "set_b": os.path.join(data_dir, f"set-b_time_{percent}.csv"),
        }
        y_path = {
            "set_a": os.path.join(data_dir, f"set-a_y_{percent}.csv"),
            "set_b": os.path.join(data_dir, f"set-b_y_{percent}.csv"),
        }

        static_data = {
            "train": pd.read_csv(static_path["set_a"]),
            "test": pd.read_csv(static_path["set_b"])
        }
        time_data = {
            "train": pd.read_csv(time_path["set_a"]),
            "test": pd.read_csv(time_path["set_b"])
        }
        y_data = {
            "train": pd.read_csv(y_path["set_a"]),
            "test": pd.read_csv(y_path["set_b"])
        }

        # # change categorical from 0/1 to -1/+1
        # if cat_type == "pos_neg":
        #     x_cat_train.replace(to_replace=0, value=-1, inplace=True)
        #     x_cat_test.replace(to_replace=0, value=-1, inplace=True)

        # train
        self.x_train_df = time_data["train"]
        train_patient_gp = self.x_train_df.groupby(by="recordid")
        self.train_pid_keys = np.array([*train_patient_gp.groups.keys()])
        num_entries_train = train_patient_gp.size()
        train_max_seq_len = num_entries_train.max()

        # test
        self.x_test_df = time_data["test"]
        test_patient_gp = self.x_test_df.groupby(by="recordid")
        self.test_pid_keys = np.array([*test_patient_gp.groups.keys()])
        num_entries_test = test_patient_gp.size()
        test_max_seq_len = num_entries_test.max()

        EHRDataset.max_seq_len = max(train_max_seq_len, test_max_seq_len)

        # change into 3D array: (N, seq_len, feat_dim)
        N = len(self.train_pid_keys)
        seq_len = self.max_seq_len
        time_dim, static_dim = self.num_feature
        sample_shape = [N, seq_len, time_dim]
        self.x_train, self.x_age, self.x_mask_train, self.pad_mask_train = self.df_gp_to_3D_array(
            sample_shape, train_patient_gp
        )
        self.y_train = y_data["train"].set_index("recordid")
        # self.y_train["date"] = pd.to_datetime(self.y_train["date"])
        # self.y_train["index_date"] = pd.to_datetime(self.y_train["index_date"])
        if mode == "test":
            N = len(self.test_pid_keys)
            seq_len = self.max_seq_len
            time_dim, static_dim = self.num_feature
            sample_shape = [N, seq_len, time_dim]
            self.x_test, self.x_age, self.x_mask_test, self.pad_mask_test = self.df_gp_to_3D_array(
                sample_shape, test_patient_gp
            )
            self.y_test = y_data["test"]
            self.y_test = y_data["test"].set_index("recordid")
            # self.y_test["date"] = pd.to_datetime(self.y_test["date"])
            # self.y_test["index_date"] = pd.to_datetime(self.y_test["index_date"])

        self.train_idx = np.arange(len(self.y_train))

    def df_gp_to_3D_array(self, sample_shape, patient_gp):
        self.logger.info("preparing 3D array...")
        x_all = np.empty(sample_shape)
        x_age = np.empty(sample_shape[:-1])
        x_mask_all = np.empty(sample_shape)
        pad_mask_all = np.empty(sample_shape)
        for idx, (pid, patient_df) in enumerate(patient_gp):
            patient_data = patient_df.drop(columns="recordid").values
            # age = round(patient_df["age"].mean())
            mask_data = ~np.isnan(patient_data)
            # zero padding
            num_entry = patient_data.shape[0]
            seq_pad_len = self.max_seq_len - num_entry
            if self.inverse_padding:
                pad_width = ((seq_pad_len, 0), (0, 0))
            else:
                pad_width = ((0, seq_pad_len), (0, 0))
            x_seq = np.pad(patient_data, pad_width, constant_values=(np.nan, np.nan))
            x_mask = np.pad(mask_data, pad_width, constant_values=(0, 0))
            pad_mask = np.zeros(sample_shape[1:])
            pad_mask[:seq_pad_len] = 1.

            # print(x_all)
            x_all[idx] = x_seq
            # x_age[idx] = age
            x_mask_all[idx] = x_mask
            pad_mask_all[idx] = pad_mask

        if self.truncated_len is not None:
            if self.inverse_padding:
                x_all = x_all[:, -self.truncated_len:, :]
                x_mask_all = x_mask_all[:, -self.truncated_len:, :]
                pad_mask_all = pad_mask_all[:, -self.truncated_len:, :]
            else:
                x_all = x_all[:, :self.truncated_len, :]
                x_mask_all = x_mask_all[:, :self.truncated_len, :]
                pad_mask_all = pad_mask_all[:, :self.truncated_len, :]

        return x_all, x_age, x_mask_all, pad_mask_all.astype(bool)

    def split_cv_indexes(self, N):
        SEED = Cross_Valid.repeat_idx
        kfold = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
        X, y = self.train_idx, self.y_train["In-hospital_death"].values
        self.indexes = list(kfold.split(X, y))

    def get_split_idx(self, fold_idx):
        return self.indexes[fold_idx]

    def transform(self, split_idx=None):
        self.set_train_idx(split_idx)
        self.compute_info()
        self.impute()
        if self.truncated_len is not None:
            EHRDataset.max_seq_len = min(self.max_seq_len, self.truncated_len)

    def set_train_idx(self, split_idx=None):
        # compute data info only on training data!
        if self.mode == "train" and split_idx is not None:
            train_idx, valid_idx = split_idx
            self.train_pid = self.train_idx[train_idx]
        else:
            # contain train and valid data
            self.train_pid = self.train_idx

    def compute_info(self):
        x_no_valid = self.x_train[self.train_pid, ...]
        x_mask_no_valid = self.x_mask_train[self.train_pid, ...]

        time_dim, static_dim = self.num_feature

        method = self.compute_method
        if method == "by_entry":
            self.num_mean = np.zeros(time_dim)
            self.num_median = np.zeros(time_dim)
            self.num_std = np.zeros(time_dim)
            for n in range(time_dim):
                num_value = x_no_valid[..., n]
                num_mask = x_mask_no_valid[..., n].astype(bool)
                observed_value = num_value[num_mask]
                self.num_mean[n] = np.mean(observed_value)
                self.num_mean[n] = np.median(observed_value)
                self.num_std[n] = np.std(observed_value)
        elif method == "by_patient":
            # x_num.shape = (N, seq_len, feat_dim)
            # 1. reduce on seq_len
            pid_num = np.nanmean(x_num, axis=1)
            pid_cat = stats.mode(x_cat, axis=1, nan_policy="omit")[0]
            pid_cat = np.squeeze(pid_cat)
            # pid_num.shape = (N, feat_dim)
            # 2. reduce on N
            self.num_mean = np.nanmean(pid_num, axis=0)
            self.num_median = np.nanmedian(pid_num, axis=0)
            self.num_std = np.nanstd(pid_num, axis=0)
            self.cat_mode = stats.mode(pid_cat)[0]

        # print(self.num_mean)
        # print(self.num_median)
        # print(self.num_std)

    def impute(self):
        # Last Observation Carried Forward
        # https://stackoverflow.com/a/60941040/8380054
        def np_ffill(arr, axis):
            idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
            idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
            np.maximum.accumulate(idx, axis=axis, out=idx)
            slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
                                       for dim in range(len(arr.shape))])]
                   for i, k in enumerate(arr.shape)]
            slc[axis] = idx
            return arr[tuple(slc)]
        if self.do_locf:
            self.x_train = np_ffill(self.x_train, 1)
            if self.mode == "test":
                self.x_test = np_ffill(self.x_test, 1)

        # padding 0 according to pad_mask
        self.x_train[self.pad_mask_train] = 0
        if self.mode == "test":
            self.x_test[self.pad_mask_test] = 0

        # filling num_fill, cat_fill for remainder
        method = self.impute_method
        if method == "zero":
            num_fill, cat_fill = 0, 0
        elif method == "mean":
            num_fill = self.num_mean
        elif method == "median":
            num_fill, cat_fill = self.num_median, self.cat_mode
        elif method == "gaussian":
            epsilon = np.random.normal()
            num_fill, cat_fill = self.num_mean + epsilon*self.num_std, self.cat_mode
        elif method == "mean_minus_3_std":
            num_fill, cat_fill = self.num_mean - 3*self.num_std, self.cat_mode

        time_dim, static_dim = self.num_feature
        x_num = self.x_train
        num_nan = np.isnan(x_num)
        self.x_train = np.where(num_nan, num_fill, x_num)
        if self.mode == "test":
            x_num = self.x_test
            num_nan = np.isnan(x_num)
            self.x_test = np.where(num_nan, num_fill, x_num)

    def normalize(self):
        time_dim, static_dim = self.num_feature
        # normalize on numerical data only
        x_num = self.x_train
        # mean/std are computed by training set only
        x_num_no_valid = x_num[self.train_pid]
        num_mean = np.mean(x_num_no_valid, axis=0)
        num_std = np.std(x_num_no_valid, axis=0)
        self.x_train = (x_num - num_mean) / (num_std + 1e-8)
        if self.mode == "test":
            x_num = self.x_test
            self.x_test = (x_num - num_mean) / (num_std + 1e-8)

    def __getitem__(self, idx):
        if self.mode == "train":
            pid = self.train_pid_keys[idx]
            x_seq = torch.from_numpy(self.x_train[idx]).float()
            x_mask = torch.from_numpy(self.x_mask_train[idx]).float()
        elif self.mode == "test":
            pid = self.test_pid_keys[idx]
            x_seq = torch.from_numpy(self.x_test[idx]).float()
            x_mask = torch.from_numpy(self.x_mask_test[idx]).float()

        # X
        time_dim, static_dim = self.num_feature
        x_num, x_cat = x_seq, 0
        x_num_mask, x_cat_mask = x_mask, 0

        # y
        if self.mode == "train":
            label = self.y_train.loc[pid]
        elif self.mode == "test":
            label = self.y_test.loc[pid]
        ## event_times
        day_delta = label["Survival"]
        ## group
        y = label["In-hospital_death"].astype(np.float32)
        y = np.expand_dims(y, axis=0)

        if self.add_emb:
            x_age = self.x_age[idx]
            x_age = torch.from_numpy(x_age).long()
            x_pos = torch.arange(self.max_seq_len)
            return (x_num, x_cat, x_num_mask, x_cat_mask), (day_delta, y), x_age, x_pos
        return (x_num, x_cat, x_num_mask, x_cat_mask), (day_delta, y)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_pid_keys)
        return len(self.test_pid_keys)


if __name__ == "__main__":
    data_dir = "/mnt/Data/peiying/Github/others/challenge2012/data"
    dataset = EHRDataset(data_dir=data_dir)

    N = 10
    ensure_dir("fig")
    print("plotting mask...")
    for i in range(N):
        (x_num, x_cat, x_num_mask, x_cat_mask), (day_delta, y) = dataset[i]
        x_mask = torch.cat([x_num_mask, x_cat_mask], 1)
        # print(x_num.size())
        # print(x_cat.size())
        # print(x_mask.size())
        save_image(x_mask, f"fig/mask_{i}.png")
    os._exit(0)

    from data_loaders.valid_loader import ValidDataLoader
    DataLoader_kwargs = {"batch_size": 128}
    loaders = ValidDataLoader(dataset, DataLoader_kwargs=DataLoader_kwargs)
    train_loader = loaders.train_loader

    print("computing missing rate...")
    padding_N = 0
    padding_slot = 0
    missing_N = 0
    missing_slot = 0
    for batch_idx, (*data, target) in enumerate(train_loader):
        x_num, x_cat, x_num_mask, x_cat_mask = data
        y = target
        x_mask = torch.cat([x_num_mask, x_cat_mask], 2)
        x_mask = x_mask > 0

        # padding rate
        padding = torch.sum(x_mask, 2) < 1.
        padding_N += padding.numel()
        padding_slot += padding.sum()

        # missing rate
        non_padding = ~padding
        non_padding = torch.unsqueeze(non_padding, -1)
        non_padding = non_padding.expand(-1, -1, x_mask.size(-1))
        missing = ~x_mask * non_padding
        N_non_padding = non_padding.sum()
        missing_slot += missing.sum()
        missing_N += N_non_padding
    print(f"padding rate: {padding_slot/padding_N}")  # 0.9588102102279663
    print(f"missing rate: {missing_slot/missing_N}")  # 0.7288970351219177
