import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import utils
from src.model import UNet2D
import config.config as exp_config
import torch.nn.functional as F
from data import acdc_data
import matplotlib.pyplot as plt
from src.attacks import fgsm, ifgsm, cadv


def get_thicker_perturbation(label, scale=0.5):
    perturbed_y = np.squeeze(label).copy()
    rows_with_3 = sorted(list(set(np.where(perturbed_y == 3)[0])))
    for row in rows_with_3:
        y = perturbed_y[row]
        all_3 = np.where(y == 3)[0]
        value, counts = np.unique(y[:all_3[0]], return_counts=True)
        y[all_3[0]: min(all_3[0] + int(counts[np.where(value == 2)][0] * scale), all_3[-1] + 1)] = 2

        all_3 = np.where(y == 3)[0]
        if len(all_3) != 0:
            value, counts = np.unique(y[all_3[-1]:], return_counts=True)
            y[max(all_3[-1] - int(counts[np.where(value == 2)][0] * scale) + 1, all_3[0]):all_3[-1] + 1] = 2
    return perturbed_y[np.newaxis, :]


def attack_net(net, device, targets=[], attacks=[]):
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
    for attack in attacks:
        print(f'--------------------- Starting attack: {attack["attack"]} ---------------------------')
        baseline_loss = 0
        baseline_dice = 0
        attack_loss = [0] * len(targets)
        attack_dice = [0] * len(targets)
        for batch_idx, batch in enumerate(test_loader):
            labels = batch['label']
            imgs = torch.reshape(batch['image'], [batch['label'].shape[0]] + [1] + list(exp_config.image_size))
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            x = imgs.clone().detach().cpu().numpy()
            y = labels.clone().detach().cpu().numpy()

            with torch.no_grad():
                logits = net(imgs)
            loss, dice = utils.evaluation(logits, labels, criterion)
            baseline_loss += loss.item()
            baseline_dice += dice.item()
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds = preds.clone().detach().cpu().numpy()

            fig, ax = plt.subplots(3, len(targets) + 1, figsize=((len(targets) + 1) * 3, 9))
            ax[0, 0].imshow(np.squeeze(x), cmap='gray')
            ax[0, 0].set_title('x')
            ax[1, 0].imshow(np.squeeze(y))
            ax[1, 0].set_title('y')
            ax[2, 0].imshow(np.squeeze(preds))
            ax[2, 0].set_title(f'pred (dice: {round(dice.item(), 3)})')

            for idx, target in enumerate(targets):
                if target == 'thicker':
                    adv_labels = get_thicker_perturbation(y, 1)
                elif target == 'blank':
                    adv_labels = np.zeros_like(y)
                else:
                    raise NotImplementedError(f'Adv target {target} has not been implemented yet.')
                adv_labels = torch.tensor(adv_labels, device=device, requires_grad=False)

                if attack['attack'] == 'fgsm':
                    adv_imgs = fgsm(imgs, adv_labels, net, criterion, device, attack['params'])
                elif attack['attack'] == 'ifgsm':
                    adv_imgs = ifgsm(imgs, adv_labels, net, criterion, device, attack['params'])
                elif attack['attack'] == 'cadv':
                    adv_imgs = cadv(imgs, adv_labels, net, criterion, device, attack['params'])
                else:
                    raise NotImplementedError(f'Attack {attack} has not been implemented yet.')

                with torch.no_grad():
                    logits = net(adv_imgs)
                loss, dice = utils.evaluation(logits, adv_labels, criterion, target)
                attack_loss[idx] += loss.item()
                attack_dice[idx] += dice.item()
                adv_x = adv_imgs.clone().detach().cpu().numpy()
                adv_y = adv_labels.clone().detach().cpu().numpy()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                preds = preds.clone().detach().cpu().numpy()

                ax[0, idx + 1].imshow(np.squeeze(adv_x), cmap='gray')
                ax[0, idx + 1].set_title('adv_x')
                ax[1, idx + 1].imshow(np.squeeze(adv_y))
                ax[1, idx + 1].set_title('adv_y')
                ax[2, idx + 1].imshow(np.squeeze(preds))
                ax[2, idx + 1].set_title(f'pred (dice: {round(dice.item(), 3)})')
            fig.tight_layout()

            if not os.path.exists('attack_outputs'):
                os.mkdir('attack_outputs')
            image_output_file = f'attack_outputs/output_{attack["attack"]}_{batch_idx + 1}.jpg'
            print("Writing output to ", image_output_file)
            plt.savefig(image_output_file, format="jpg")
            plt.clf()

        print(f'Baseline - Loss: {round(baseline_loss / n_test, 3)}, Baseline Dice: {round(baseline_dice / n_test, 3)}')
        for i in range(len(targets)):
            print(f'Adv Target - {targets[i]} Loss: {round(attack_loss[i] / n_test, 3)}, '
                  f'Dice: {round(attack_dice[i] / n_test, 3)}')


if __name__ == '__main__':
    log_dir = os.path.join(exp_config.log_root, exp_config.experiment_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet2D(nchannels=1, nlabels=4)

    if os.path.exists(log_dir):
        net, _ = utils.get_latest_checkpoint(net, log_dir, device)

    net.to(device=device)
    attack_params = [
                    {
                        'attack': 'fgsm',
                        'params': {'alpha': 0.1}
                     },
                     {
                         'attack': 'ifgsm',
                         'params': {'alpha': 0.1, 'eps': 0.5, 'steps': 40}
                     },
                    {
                        'attack': 'cadv',
                        'params': {
                            'batch_size': 1,
                            'ab_max': 110.,
                            'ab_quant': 10.,
                            'l_norm': 100.,
                            'l_cent': 50.,
                            'mask_cent': 0.5,
                            'hint': 50,
                            'lr': 1e-3,
                            'target': 0,
                            'targeted': True,
                            'n_clusters': 8,
                            'k': 4,
                            'num_iter': 700
                        }
                    }]
    attack_net(net=net, device=device, targets=['thicker', 'blank'], attacks=attack_params)
