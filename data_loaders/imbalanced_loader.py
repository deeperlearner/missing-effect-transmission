import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from base import BaseDataLoader
from mains import Cross_Valid


class ImbalancedDataLoader(BaseDataLoader):
    """
    This loader will balance the ratio of class for each mini-batch
    """

    def __init__(
        self,
        dataset,
        class_weight=None,
        target=None,
        validation_split=0.0,
        DataLoader_kwargs=None,
        stratify_by_labels=None,
        do_transform=True,
        do_normalize=True,
        sampling_type="under_sampling",
        RUS_rate=20.,
    ):
        self.sampling_type = sampling_type
        self.RUS_rate = RUS_rate
        super(ImbalancedDataLoader, self).__init__(
            dataset, validation_split, DataLoader_kwargs
        )

        if dataset.mode in ("train", "valid"):
            if Cross_Valid.k_fold > 1:
                split_idx = dataset.get_split_idx(Cross_Valid.fold_idx)
                train_sampler, valid_sampler = self._get_sampler(
                    *split_idx, class_weight, target
                )
            else:
                if validation_split > 0.0:
                    split_idx = self._train_valid_split(labels=stratify_by_labels)
                    train_sampler, valid_sampler = self._get_sampler(
                        *split_idx, class_weight, target
                    )
                else:
                    split_idx = None
                    train_sampler, valid_sampler = None, None

            if do_transform:
                dataset.transform(split_idx)
            if do_normalize:
                dataset.normalize()
            self.train_loader = DataLoader(
                dataset, sampler=train_sampler, **self.init_kwargs
            )
            self.valid_loader = DataLoader(
                dataset, sampler=valid_sampler, **self.init_kwargs
            )

        elif dataset.mode == "test":
            if do_transform:
                dataset.transform()
            if do_normalize:
                dataset.normalize()
            self.test_loader = DataLoader(dataset, **self.init_kwargs)

    def _get_sampler(self, train_idx, valid_idx, class_weight, target):
        train_mask = np.zeros(self.n_samples)
        train_mask[train_idx] = 1.0

        if self.sampling_type == "under_sampling":
            class_weight = np.array([1., self.RUS_rate])  # (majority, minority)
            train_weights = class_weight[target] * train_mask
            num_samples = len(train_idx)  # tuning len_epoch in trainer to use fewer samples

            train_sampler = WeightedRandomSampler(train_weights, num_samples, replacement=True)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs["shuffle"] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
