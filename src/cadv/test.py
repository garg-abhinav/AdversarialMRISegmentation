import os
from src.cadv.models import color_net
from src.cadv.util import compute_class, forward, compute_loss, get_colorization_data
from src.cadv.dataloader import im_dataset
import config.config as exp_config
import torch
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
import argparse
import json


class Options:
    def __init__(self):
        self.batch_size = 1
        self.ab_max = 110.
        self.ab_quant = 10.
        self.l_norm = 100.
        self.l_cent = 50.
        self.mask_cent = .5
        self.target = 444
        self.hint = 50
        self.lr = 1e-4
        self.targeted = True
        self.n_clusters = 8
        self.k = 4
        self.num_iter = 500


opt = Options()


def run(images, segmentation_model, criterion, target, device):
    dataset = im_dataset(images)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)

    model = color_net().to(device=device).eval()
    model.load_state_dict(torch.load(os.path.join(exp_config.log_root, 'latest_net_G.pth')))

    threshold = 1e-5 # stopping criteria

    for i, (image) in enumerate(dataset_loader):
    # print(im)
    im = image.to(device=device)
    opt.target = target
    # Prepare hints, mask, and get current classification
    data = get_colorization_data(im, opt, model, device)
    # print(data.shape, opt.target.shape)
    optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()],
                                  lr=opt.lr, betas=(0.9, 0.999))

    prev_norm = 0
    for itr in range(10):
        out_rgb, y = forward(model, segmentation_model, opt, data)
        val = compute_class(opt, y)
        loss = compute_loss(opt, y, criterion, device)
        # print(out_rgb.shape, val.shape)

        optimizer.zero_grad()
        loss.backward()
#         print(y.grad)
        optimizer.step()
        # plot_grad_flow(model.named_parameters(), itr)
#         norm = 0
        # norm = frobenius_norm(torch.squeeze(out_rgb).cpu().detach().numpy())
        print(f'[{itr+1}/{opt.num_iter}] Loss: {loss:.3f}')
#         print("%.5f"%(loss.item()))

        # if np.abs(prev_norm - norm) < threshold:
        #     break

#         prev_norm = norm

        diff = val[0, 0, :, :] - val[0, 1, :, :]
        
        if opt.targeted:
            if idx == opt.target and torch.mean(diff) > threshold and torch.mean((diff-prev_diff).abs()) < 1e-3:
                break
        else:
            if idx != opt.target and torch.mean(diff) > threshold and torch.mean((diff-prev_diff).abs()) < 1e-3:
                break
        prev_diff = diff
    print('-=-=-=-=-',out_rgb.shape)
    return out_rgb
    # file_name = file_name[0] + '.png'
    # save_image(out_rgb, os.path.join(opt.results_dir, file_name))