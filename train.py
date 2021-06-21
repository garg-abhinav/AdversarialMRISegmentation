import logging
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from model import UNet2D
import config.config as exp_config
import acdc_data

log_dir = os.path.join(exp_config.log_root, exp_config.experiment_name)


def train_net(net, device, global_step=0):

    data = acdc_data.load_and_maybe_process_data(
        input_folder=exp_config.data_root,
        preprocessing_folder=exp_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False,
        split_test_train=True
    )

    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train']
    labels_train = data['masks_train']
    images_val = data['images_test']
    labels_val = data['masks_test']

    print(np.unique(labels_train), np.unique(labels_val))

    train_data = acdc_data.BasicDataset(images_train, labels_train)
    val_data = acdc_data.BasicDataset(images_val, labels_val)

    n_val = len(images_val)
    n_train = len(images_train)

    train_loader = DataLoader(train_data, batch_size=exp_config.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=exp_config.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment='LR_{}_BS_{}'.format(exp_config.lr, exp_config.batch_size))

    optimizer = optim.Adam(net.parameters(), lr=exp_config.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                     threshold=exp_config.lr_threshold, verbose=True)
    # criterion = utils.loss
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32,
                                                        device=device, requires_grad=False))
    max_epochs = exp_config.max_epochs - global_step//math.ceil(n_train/exp_config.batch_size)
    logging.info(f'''Starting training from step {global_step}:
            Epochs:          {max_epochs}
            Batch size:      {exp_config.batch_size}
            Learning rate:   {exp_config.lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
        ''')
    for epoch in range(max_epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='img') as pbar:
            for batch in train_loader:
                labels = batch['label']
                imgs = torch.reshape(batch['image'], [batch['label'].shape[0]] + [1] + list(exp_config.image_size))

                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                assert imgs.shape[1] == net.nchannels, \
                    f'Network has been defined with {net.nchannels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # assert len(torch.unique(labels)) == net.nlabels, \
                #     f'Network has been defined with {net.nlabels} output labels, ' \
                #     f'but loaded images have {torch.unique(labels, sorted=True)} labels. Please check that ' \
                #     'the labels are loaded correctly.'

                logits = net(imgs)
                loss = criterion(logits, labels)
                # epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss(batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        # if global_step % exp_config.val_eval_frequency == 0:
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

            val_loss, val_score = val_net(net, val_loader, device, criterion)
            scheduler.step(val_score)

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            logging.info('Validation Loss: {}'.format(val_loss))
            writer.add_scalar('Loss/test', val_loss, global_step)
            logging.info('Validation Dice Score: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, global_step)
            # writer.add_images('images/image', imgs, global_step)
            # writer.add_images('images/gt', labels, global_step)
            # writer.add_images('images/pred', torch.argmax(F.softmax(logits[0], dim=1), dim=1), global_step)

            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), os.path.join(log_dir, f'CP_step_{global_step}.pth'))
            logging.info(f'Checkpoint {global_step} saved !')

    writer.close()


def val_net(net, loader, device, criterion):
    n_val = len(loader)
    net.eval()
    total_loss = 0
    total_dice = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            labels = batch['label']
            imgs = torch.reshape(batch['image'], [batch['label'].shape[0]] + [1] + list(exp_config.image_size))
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.no_grad():
                logits = net(imgs)

            loss, dice = utils.evaluation(logits, labels, criterion)
            total_loss += loss
            total_dice += dice
            pbar.update()
    net.train()
    return total_loss/n_val, total_dice/n_val


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet2D(nchannels=1, nlabels=4)
    logging.info(f'Network:\n'
                 f'\t{net.nchannels} input channels\n'
                 f'\t{net.nlabels} output channels (classes)')

    global_step = 0
    if os.path.exists(log_dir):
        net, global_step = utils.get_latest_checkpoint(net, log_dir, device)
        logging.info(f'Model loaded from step {global_step}')

    net.to(device=device)

    train_net(net=net, device=device, global_step=global_step)
