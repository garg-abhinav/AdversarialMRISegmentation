import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from src import utils
from src.model import UNet2D
import config.config as exp_config
import os
from data import acdc_data
from src import utils

use_cuda = True
image_nc = 1
epochs = 2
BOX_MIN = 0
BOX_MAX = 1
b = 4.775

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32,
                                                          device=device, requires_grad=False))

targeted_model = UNet2D(nchannels=1, nlabels=4)
log_dir = os.path.join(exp_config.log_root, exp_config.experiment_name)
global_step = 0
if os.path.exists(log_dir):
    targeted_model, global_step = utils.get_latest_checkpoint(targeted_model, log_dir, device)

targeted_model.to(device=device)
targeted_model.eval()
model_num_labels = 4

# MNIST train dataset and dataloader declaration
# mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
# dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

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

adv_labels_train = []
for label in labels_train:
    adv_labels_train.append(utils.get_thicker_perturbation(label, 1))
adv_labels_train = np.array(adv_labels_train)
adv_labels_train = np.squeeze(adv_labels_train)
print(adv_labels_train.shape)
train_data = acdc_data.BasicDataset(images_train, adv_labels_train)
# val_data = acdc_data.BasicDataset(images_val, labels_val)

# n_val = len(images_val)
n_train = len(images_train)

train_loader = DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

advGAN = AdvGAN_Attack(device,
                       targeted_model,
                       model_num_labels,
                       image_nc,
                       BOX_MIN,
                       BOX_MAX,
                       criterion,
                       b)

advGAN.train(train_loader, epochs, n_train)
