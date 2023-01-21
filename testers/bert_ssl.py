import torch


class Tester:
    """
    Tester class
    """

    def __init__(self, test_data_loaders, models, device, metrics_epoch, test_metrics):
        self.test_data_loaders = test_data_loaders
        self.models = models
        self.device = device
        self.metrics_epoch = metrics_epoch
        self.test_metrics = test_metrics

    def test(self):
        with torch.no_grad():
            print("testing...")
            test_loader = self.test_data_loaders["data"]
            encoder = self.models["encoder"]
            # mask_decoder = self.models["mask_decoder"]
            # recon_decoder = self.models["recon_decoder"]
            target_decoder = self.models["target_decoder"]

            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                event_times = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(test_loader):
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                data = [x.to(self.device) for x in data]
                event_time, target = [y.to(self.device) for y in target]

                encoder_output = encoder(data)
                target_output = target_decoder(encoder_output)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, target_output))
                    event_times = torch.cat((event_times, event_time))
                    targets = torch.cat((targets, target))

                #
                # save sample images, or do something with output here
                #

            for met in self.metrics_epoch:
                self.test_metrics.epoch_update(met.__name__, met(event_times, targets, outputs))

        return event_times, targets, outputs, self.test_metrics.result()
