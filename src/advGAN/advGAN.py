import torch.nn as nn
import torch
import config.config as exp_config
import models
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import rice
import os

log_dir = os.path.join(exp_config.log_root, exp_config.experiment_name)

torch.autograd.set_detect_anomaly(True)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 criterion,
                 b=4.775):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.b = b

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.0001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.0001)
        self.BCELoss = nn.BCELoss()
        self.model_criterion = criterion

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    def get_rician_samples(self, img_shape):
        rician_samples = torch.tensor(rice.rvs(self.b, size=img_shape),
                                      device=self.device, dtype=torch.float32, requires_grad=False)
        return rician_samples

    def train_generator(self, x, labels):
        # optimize G
        self.optimizer_G.zero_grad()

        # cal G's loss in GAN
        perturbation = self.netG(x)
        # perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_images = perturbation + x
        # pred_fake = self.netDisc(perturbation.detach())
        pred_fake = self.netDisc(adv_images.detach())
        # loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_G_fake = self.BCELoss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        # loss_G_fake.backward(retain_graph=True)

        # calculate perturbation norm
        C = 0.1
        loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
        loss_perturb = torch.mean(torch.max(loss_perturb - C, torch.zeros(loss_perturb.shape, device=self.device)))
        
        # calculate adv loss
        logits_model = self.model(adv_images.detach())
        loss_adv = self.model_criterion(logits_model, labels)

        adv_lambda = 10
        gen_lambda = 1
        pert_lambda = 0
        loss_G = adv_lambda * loss_adv + gen_lambda * loss_G_fake + pert_lambda * loss_perturb
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train_discriminator(self, x, labels):
        # optimize D
        perturbation = self.netG(x)
        # perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_images = perturbation + x

        self.optimizer_D.zero_grad()
        # rician_noise = self.get_rician_samples(perturbation.shape)
        # pred_real = self.netDisc(rician_noise)
        pred_real = self.netDisc(x)
        # loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        loss_D_real = self.BCELoss(pred_real, torch.ones_like(pred_real, device=self.device))
        # loss_D_real.backward()

        # pred_fake = self.netDisc(perturbation.detach())
        pred_fake = self.netDisc(adv_images.detach())
        # loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D_fake = self.BCELoss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        # loss_D_fake.backward()
        loss_D_GAN = loss_D_fake + loss_D_real
        loss_D_GAN.backward()
        self.optimizer_D.step()
        return loss_D_GAN.item()

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            # not sure to include this or not?
            # perturbation = torch.clamp(perturbation, -0.3, 0.3)
            #             adv_images = perturbation + x
            # adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            rician_noise = self.get_rician_samples(perturbation.shape)
            pred_real = self.netDisc(rician_noise)
            #             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real = self.BCELoss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(perturbation.detach())
            #             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake = self.BCELoss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = (loss_D_fake + loss_D_real) / 2
            self.optimizer_D.step()
        # optimize G
        for i in range(3):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            perturbation = self.netG(x)
            adv_images = perturbation + x
            pred_fake = self.netDisc(perturbation.detach())
            #             loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake = self.BCELoss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            # loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            # loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))
            loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            loss_perturb = torch.mean(
                torch.max(loss_perturb - C, torch.zeros(loss_perturb.shape, device=self.device)))

            # cal adv loss
            #             eps = 0.5
            #             alpha = 0.1
            #             pgd_images = adv_images.clone().detach().to(self.device)
            #             pgd_labels = labels.clone().detach().to(self.device)
            #             clip_min = adv_images - eps
            #             clip_max = adv_images + eps

            #             for i in range(1):
            #                 pgd_images.requires_grad = True
            #                 logits = self.model(pgd_images)
            #                 cost = self.model_criterion(logits, pgd_labels)

            #                 grad = torch.autograd.grad(cost, pgd_images, retain_graph=False, create_graph=False)[0]

            #                 step_output = alpha * grad.sign()
            #                 pgd_images = torch.clamp(pgd_images + step_output, min=clip_min, max=clip_max).detach()

            logits_model = self.model(adv_images.detach())
            # probs_model = F.softmax(logits_model, dim=1)
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            loss_adv = self.model_criterion(logits_model, labels)

            # C&W loss function
            #             probs_model = F.softmax(logits_model, dim=1)
            #             onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            #             onehot_labels = onehot_labels.reshape(onehot_labels.shape[0], onehot_labels.shape[3],
            #                                                  onehot_labels.shape[1], onehot_labels.shape[2])
            # #             print(labels.shape, onehot_labels.shape, probs_model.shape)
            #             real = torch.sum(onehot_labels * probs_model, dim=1)
            #             other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            #             zeros = torch.zeros_like(other)
            #             loss_adv = torch.max(real - other, zeros)
            #             loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = - F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 0.5
            pert_lambda = 1
            loss_G = loss_adv + adv_lambda * loss_G_fake  # + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs, n_train):
        writer = SummaryWriter(comment="advGAN")
        print('tensorboard')
        global_step = 0
        for epoch in range(1, epochs + 1):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                if epoch == 50:
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                        lr=0.00001)
                    self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                        lr=0.00001)
                if epoch == 80:
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                        lr=0.000001)
                    self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                        lr=0.000001)
                loss_D_sum = 0
                loss_G_fake_sum = 0
                loss_perturb_sum = 0
                loss_adv_sum = 0
                loss_D_batch = 0
                for i, data in enumerate(train_dataloader):
                    labels = data['label']
                    imgs = torch.reshape(data['image'], [data['label'].shape[0]] + [1] + list(exp_config.image_size))

                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.long)

                    # loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    #     self.train_batch(imgs, labels)
                    loss_D_batch = self.train_discriminator(imgs, labels)
                    loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_generator(imgs, labels)
                    
                    loss_D_sum += loss_D_batch
                    loss_G_fake_sum += loss_G_fake_batch
                    loss_perturb_sum += loss_perturb_batch
                    loss_adv_sum += loss_adv_batch

                    pbar.set_postfix(**{"loss_D": loss_D_batch,
                                        "loss_G_fake": loss_G_fake_batch,
                                        "loss_perturb": loss_perturb_batch,
                                        "loss_adv": loss_adv_batch})
                    pbar.update(imgs.shape[0])
                    writer.add_scalar('loss_D/train', loss_D_batch, global_step)
                    writer.add_scalar('loss_G_fake/train', loss_G_fake_batch, global_step)
                    writer.add_scalar('loss_perturb/train', loss_perturb_batch, global_step)
                    writer.add_scalar('loss_adv/train', loss_adv_batch, global_step)
                    global_step += 1
                writer.add_scalar('learning_rate', self.optimizer_G.param_groups[0]['lr'], global_step)

                # print statistics

                # print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
                #  \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                #       (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                #        loss_perturb_sum / num_batch, loss_adv_sum / num_batch))

                # save generator
                if epoch % 10 == 0:
                    netG_file_name = os.path.join(log_dir, 'netG_base_epoch_' + str(epoch) + '.pth')
                    torch.save(self.netG.state_dict(), netG_file_name)
        netG_file_name = os.path.join(log_dir, 'netG_base_epoch_' + str(epoch) + '.pth')
        torch.save(self.netG.state_dict(), netG_file_name)
        writer.close()

