import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils
from model import UNet2D
import config.config as exp_config
import torch.nn.functional as F
import acdc_data
import matplotlib.pyplot as plt


def test_net(net, device):
    data = acdc_data.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False,
        split_test_train=True
    )

    images = data['images_test'][:5]
    labels = data['masks_test'][:5]

    print(np.unique(labels))
    test_data = acdc_data.BasicDataset(images, labels)
    n_test = len(images)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32,
                                                              device=device, requires_grad=False))
    net.eval()
    total_loss = 0
    total_dice = 0
    fig, ax = plt.subplots(3, n_test, figsize=(n_test * 3, 9))
    with tqdm(total=n_test, desc='Test Round', unit='batch') as pbar:
        for idx, batch in enumerate(test_loader):
            labels = batch['label']
            imgs = torch.reshape(batch['image'], [batch['label'].shape[0]] + [1] + list(exp_config.image_size))
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            with torch.no_grad():
                logits = net(imgs)
            loss, dice = utils.evaluation(logits, labels, criterion)
            total_loss += loss
            total_dice += dice

            x = imgs.clone().detach().cpu().numpy()
            y = labels.clone().detach().cpu().numpy()
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds = preds.clone().detach().cpu().numpy()

            ax[0, idx].imshow(np.squeeze(x), cmap='gray')
            ax[0, idx].set_title('x')
            ax[1, idx].imshow(np.squeeze(y))
            ax[1, idx].set_title('y')
            ax[2, idx].imshow(np.squeeze(preds))
            ax[2, idx].set_title(f'pred (dice: {dice})')

            pbar.update()
    fig.tight_layout()
    image_output_file = 'output.pdf'
    print("Writing output to ", image_output_file)
    plt.savefig(image_output_file, format="pdf")
    plt.clf()
    print(preds)
    return logits, total_loss / n_test, total_dice / n_test


if __name__ == '__main__':
    log_dir = os.path.join(exp_config.log_root, exp_config.experiment_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet2D(nchannels=1, nlabels=4)

    global_step = 0
    if os.path.exists(log_dir):
        net, global_step = utils.get_latest_checkpoint(net, log_dir, device)

    net.to(device=device)

    preds, closs, cdice = test_net(net=net, device=device)
    print(f'Avg loss: {closs}, Avg dice: {cdice}')
