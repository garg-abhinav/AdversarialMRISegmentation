import torch


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
