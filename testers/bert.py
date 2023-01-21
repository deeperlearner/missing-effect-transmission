import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms


class Tester:
    """
    Tester class
    """

    def __init__(self, test_data_loaders, models, device, metrics_epoch, test_metrics):
        self.test_data_loaders = test_data_loaders
        self.model = models["model"]
        self.device = device
        self.metrics_epoch = metrics_epoch
        self.test_metrics = test_metrics

    def test(self):
        self.model.eval()
        with torch.no_grad():
            print("testing...")
            test_loader = self.test_data_loaders["data"]

            do_vis = True
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                event_times = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(test_loader):
                # data: (x_num, x_cat, x_num_mask, x_cat_mask)
                # target: (day_delta, group)
                data = [x.to(self.device) for x in data]
                event_time, target = [y.to(self.device) for y in target]

                if batch_idx == 0 and do_vis:
                    attn_layers = {}
                    other_layers = {}
                    def get_layer(name):
                        def hook(model, input, output):
                            if type(output) == tuple:  # this is self_attn
                                attn_layers[name] = output[1].detach()
                            else:
                                other_layers[name] = output.detach()
                        return hook
                    for name, layer in self.model.named_modules():
                        print(name)
                        layer.register_forward_hook(get_layer(name))

                output = self.model(data)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    event_times = torch.cat((event_times, event_time))
                    targets = torch.cat((targets, target))

                #
                # save sample images, or do something with output here
                #

                if batch_idx == 0 and do_vis:
                    attn0 = attn_layers['transformer_encoder.layers.0.self_attn']
                    attn1 = attn_layers['transformer_encoder.layers.1.self_attn']
                    attn2 = attn_layers['transformer_encoder.layers.2.self_attn']
                    attn2 = torch.unsqueeze(attn2, 1)
                    # print(attn2.size())

                    # transform = transforms.Compose([
                    #     transforms.ToPILImage(),
                    #     transforms.Resize(size=24),
                    #     transforms.ToTensor()
                    # ])
                    # attn2 = [transform(x_) for x_ in attn2]

                    attn_mask_2 = make_grid(attn2)
                    save_image(attn_mask_2, 'fig/attn_mask_2.png')
                    # print(attn_mask_2.size())


            for met in self.metrics_epoch:
                self.test_metrics.epoch_update(met.__name__, met(event_times, targets, outputs))

        return event_times, targets, outputs, self.test_metrics.result()
