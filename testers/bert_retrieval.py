import torch


class Tester:
    """
    Tester class
    """

    def __init__(
        self, train_data_loaders, test_data_loaders,
        models, device, metrics_epoch, test_metrics,
        ensemble_times=30
    ):
        self.train_data_loaders = train_data_loaders
        self.test_data_loaders = test_data_loaders
        self.models = models
        self.device = device
        self.metrics_epoch = metrics_epoch
        self.test_metrics = test_metrics
        self.ensemble_times = ensemble_times

    def test(self):
        with torch.no_grad():
            print("testing...")
            train_loader = self.train_data_loaders["data"]
            test_loader = self.test_data_loaders["data"]
            model = self.models["model"]

            if len(self.metrics_epoch) > 0:
                outputs_results = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for t in range(self.ensemble_times):
                train_loader.set_loader(t)
                label_loader = train_loader.label_loader
                outputs = torch.FloatTensor().to(self.device)
                for X1, X2 in zip(label_loader, test_loader):
                    # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                    *data1, target1 = X1
                    data1 = [x.to(self.device) for x in data1]
                    target1 = target1.to(self.device)

                    *data2, target2 = X2
                    data2 = [x.to(self.device) for x in data2]
                    target2 = target2.to(self.device)

                    # last mini-batch inconsist
                    label_size = target1.size(0)
                    test_size = target2.size(0)
                    if label_size != test_size:
                        data1 = [x[:test_size] for x in data1]
                        target1 = target1[:test_size]

                    pair_target = (target1 != target2).long()

                    output = model(data1, data2)
                    if len(self.metrics_epoch) > 0:
                        outputs = torch.cat((outputs, output))
                        if t == 0:
                            targets = torch.cat((targets, target2))

                    #
                    # save sample images, or do something with output here
                    #
                outputs_results = torch.cat((outputs_results, outputs), dim=1)
            avg_result = torch.mean(outputs_results, 1)

            for met in self.metrics_epoch:
                self.test_metrics.epoch_update(met.__name__, met(targets, avg_result))

        return targets, outputs, self.test_metrics.result()
