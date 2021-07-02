import os
from src.cadv.models import color_net
from src.cadv.util import compute_class, forward, compute_loss, get_colorization_data
from src.cadv.dataloader import im_dataset
import config.config as exp_config
import torch
from scipy.stats import rice


def fgsm(images, labels, model, criterion, device, attack_params=dict()):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True
    logits = model(images)
    cost = criterion(logits, labels)

    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

    adv_images = images - attack_params['alpha'] * grad.sign()
    # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images


def ifgsm(images, labels, model, criterion, device, attack_params=dict()):
    adv_images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    clip_min = images - attack_params['eps']
    clip_max = images + attack_params['eps']

    for i in range(attack_params['steps']):
        adv_images.requires_grad = True
        logits = model(adv_images)
        cost = criterion(logits, labels)

        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

        step_output = attack_params['alpha'] * grad.sign()
        adv_images = torch.clamp(adv_images - step_output, min=clip_min, max=clip_max).detach()

    return adv_images


def cadv(images, target, segmentation_model, criterion, device, attack_params=dict()):
    dataset = im_dataset(images)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=attack_params['batch_size'])

    model = color_net().to(device=device).eval()
    model.load_state_dict(torch.load(os.path.join(exp_config.log_root, 'latest_net_G.pth')))

    threshold = 1e-5  # stopping criteria
    prev_diff = 0
    adv_images = []
    for i, (image) in enumerate(dataset_loader):
        im = image.to(device=device)
        attack_params['target'] = target

        # Prepare hints, mask, and get current classification
        data = get_colorization_data(im, attack_params, model, device)
        optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()],
                                     lr=attack_params['lr'], betas=(0.9, 0.999))

        for itr in range(attack_params['num_iter']):
            out_rgb, y = forward(model, segmentation_model, attack_params, data)
            val, idx = compute_class(attack_params, y)
            loss = compute_loss(attack_params, y, criterion, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diff = val[0, 0, :, :] - val[0, 1, :, :]

            if (itr + 1) % 100 == 0 or itr == 0:
                print(f'[{itr + 1}/{attack_params["num_iter"]}] Loss: {loss:.3f}, '
                      f'{torch.mean(diff).item():.5f}, {torch.mean((diff - prev_diff).abs()).item():.5f}')

            if attack_params['targeted']:
                if torch.allclose(torch.argmax(val, dim=1), target) and torch.mean(
                        diff).item() > threshold and torch.mean((diff - prev_diff).abs()).item() < 1e-4:
                    print(f'[{itr + 1}/{attack_params["num_iter"]}] Loss: {loss:.3f}, '
                          f'{torch.mean(diff).item():.5f}, {torch.mean((diff - prev_diff).abs()).item():.5f}')
                    break
            else:
                if not torch.allclose(torch.argmax(val, dim=1), target) and torch.mean(
                        diff).item() > threshold and torch.mean((diff - prev_diff).abs()).item() < 1e-4:
                    print(f'[{itr + 1}/{attack_params["num_iter"]}] Loss: {loss:.3f}, '
                          f'{torch.mean(diff).item():.5f}, {torch.mean((diff - prev_diff).abs()).item():.5f}')
                    break
            prev_diff = diff
        adv_images.append(out_rgb[0])  # assuming batch_size of 1

    return torch.stack(adv_images)


def rice_ifgsm(images, labels, model, criterion, device, attack_params=dict()):
    adv_images = images.clone().detach().to(device) + torch.tensor(rice.rvs(attack_params['b'], size=images.shape),
                                                                   device=device)
    labels = labels.clone().detach().to(device)
    clip_min = images - attack_params['eps']
    clip_max = images + attack_params['eps']

    for i in range(attack_params['steps']):
        adv_images.requires_grad = True
        logits = model(adv_images)
        cost = criterion(logits, labels)

        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

        step_output = attack_params['alpha'] * grad.sign()
        adv_images = torch.clamp(adv_images - step_output, min=clip_min, max=clip_max).detach()

    return adv_images
