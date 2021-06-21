import os

experiment_name = 'unet2D'

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
nlabels = 4

# Training settings
batch_size = 4
lr = 0.01

# Rarely changed settings
max_epochs = 20000
lr_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                        # be reduced
train_eval_frequency = 200
val_eval_frequency = 100

# Directory settings
project_root = '/home/abhinavgarg/adv_mri/'
data_root = '/home/abhinavgarg/data/training/'
test_data_root = '/home/abhinavgarg/data/testing/'
log_root = os.path.join(project_root, 'acdc_logdir')
preproc_folder = os.path.join(project_root, 'preproc_data')
