import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import glob


def loss(logits, labels, class_weights=[0.1, 0.3, 0.3, 0.3]):
    """
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    """
    n_class = len(class_weights)
    flat_logits = torch.reshape(logits, [-1, n_class])
    flat_labels = torch.reshape(labels, [-1, ])
    if logits.is_cuda:
        class_weights = torch.tensor(class_weights).cuda()
    else:
        class_weights = torch.tensor(class_weights)
    weight_map = F.one_hot(flat_labels, num_classes=n_class) * class_weights
    weight_map = torch.sum(weight_map, dim=1)
    loss_map = F.cross_entropy(input=flat_logits, target=flat_labels, reduction='none')
    weighted_loss = loss_map * weight_map
    return torch.mean(weighted_loss)


def dice_score_per_structure(logits, labels, epsilon=1e-10):
    """
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: tensor shaped (torch.shape(logits)[0], torch.shape(logits)[1])
    """
    hard_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    hard_pred = F.one_hot(hard_pred.to(torch.int64), num_classes=logits.shape[1])
    labels = F.one_hot(labels.to(torch.int64), num_classes=logits.shape[1])
    intersection = hard_pred * labels
    intersec_per_img_per_lab = torch.sum(intersection, dim=[1, 2])
    l = torch.sum(hard_pred, dim=[1, 2])
    r = torch.sum(labels, dim=[1, 2])
    dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)
    return dices_per_subj


def evaluation(logits, labels, criterion, target=''):
    """
    A function for evaluating the performance of the network on a minibatch. This function returns the loss and the
    current foreground Dice score.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    """
    segmentation_loss = criterion(logits, labels)
    cdice_structures = dice_score_per_structure(logits, labels)
    if target == 'blank':
        cdice = torch.mean(cdice_structures[:, 0])
    else:
        cdice_foreground = cdice_structures[:, 1:]
        cdice = torch.mean(cdice_foreground)
    return segmentation_loss, cdice


def get_latest_checkpoint(net, log_dir, device):
    files = glob.glob(log_dir + '/*.pth')
    checkpoints = []
    for i in files:
        checkpoints.append(int(i.split('/')[-1].split('_')[-1].split('.')[0]))
    latest_cp = max(checkpoints)
    file = os.path.join(log_dir, f'CP_step_{latest_cp}.pth')
    net.load_state_dict(torch.load(file, map_location=device))
    print(f'Model loaded from {file}')
    return net, latest_cp


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)
    
    
def KL(a, b):
    '''
    Function to calculate KL divergence between two distributions
    '''
    a = np.asarray(a.reshape(-1), dtype=np.float)
    b = np.asarray(b.reshape(-1), dtype=np.float)

    return np.sum(np.where((a != 0) & (b != 0), a * np.log(a / b), 0))


def get_thicker_perturbation(label, scale=0.5):
    perturbed_y = np.squeeze(label).copy()
    rows_with_3 = sorted(list(set(np.where(perturbed_y == 3)[0])))
    for row in rows_with_3:
        y = perturbed_y[row]
        all_3 = np.where(y == 3)[0]
        value, counts = np.unique(y[:all_3[0]], return_counts=True)
        try:
            y[all_3[0]: min(all_3[0] + int(counts[np.where(value == 2)][0] * scale), all_3[-1] + 1)] = 2
        except IndexError:
            print('skipped')
            pass

        all_3 = np.where(y == 3)[0]
        if len(all_3) != 0:
            value, counts = np.unique(y[all_3[-1]:], return_counts=True)
            y[max(all_3[-1] - int(counts[np.where(value == 2)][0] * scale) + 1, all_3[0]):all_3[-1] + 1] = 2
    return perturbed_y[np.newaxis, :]
