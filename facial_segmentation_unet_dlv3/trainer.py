          

import os
import time
import torch
import datetime
import wandb 
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
from data_loader import Data_Loader_Split, Data_Loader
from unet import unet
from utils import *
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/training')
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.detection import maskrcnn_resnet50_fpn



def dice_coefficient(pred, target, class_index=1, epsilon=1e-6):
    """
    Compute the Dice coefficient for a specific class in batched data.
    
    Parameters:
    - pred: Predicted labels (B, H, W), where B is the batch size.
    - target: True labels (B, H, W), with the same shape as pred.
    - class_index: The class to compute the Dice score for.
    - epsilon: A small value to avoid division by zero.
    
    Returns:
    - dice_score: The average Dice score for the specified class across all batches.
    """
 
    pred_cls = (pred == class_index).float()
    target_cls = (target == class_index).float()
    
    intersection = (pred_cls * target_cls).sum(dim=[1, 2])
    union = pred_cls.sum(dim=[1, 2]) + target_cls.sum(dim=[1, 2])
    
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    return dice_score.mean()

class Trainer(object):
    def __init__(self, data_loader, config, hp_tune = None,device='cpu'):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.use_tensorboard = config.use_tensorboard
        self.img_path = config.img_path
        self.label_path = config.img_path
        self.train_switch = config.train
    
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.train_limit = config.train_limit
        print(hp_tune)
        self.w_b_config = hp_tune
        self.split_face = config.split_face
        
        self.dlv3 = config.dlv3
        self.device = device
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):
        
        if self.w_b_config:
            wandb.init(project="DLV3_celeb", 
                       name=f"dl_v3_celeb_3_class_hp_search_GRID"
            )
            # print(f"PRINTING SWEEP CONFIG: {sweep_config}")
            self.g_lr = wandb.config.learning_rate
            self.batch_size = wandb.config.batch_size
            self.beta1 = wandb.config.beta1
            self.beta2 = wandb.config.beta1
            
            if self.split_face:
                loader = Data_Loader_Split(self.img_path, self.label_path, self.imsize,
                                self.batch_size, self.train_switch, self.train_limit)    
                self.data_loader = loader.loader()
        
            else:
                loader = Data_Loader(self.img_path, self.label_path, self.imsize,
                                self.batch_size, self.train_switch, self.train_limit)
                self.data_loader = loader.loader()


        else:
            # configure weights and biases here!!
            wandb.init(project="unet_celeb", name=f"hp_tune_3_trainlimit={self.train_limit}", config={
                "learning_rate": self.g_lr,
                "batch_size": self.batch_size,
                "image_size": self.imsize,
                "steps":self.total_step,
                "training samples": self.train_limit,
                "classes":['sclera', 'brows'],
                "network": "unet"
            })
            pass
        
        

        print(f"Using device: {self.device}")
        
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        
        print(f"steps per epoch : {step_per_epoch}")
        # model_save_step = int(self.model_save_step * step_per_epoch)
        model_save_step = int(self.model_save_step)

        
        
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time`
        start_time = time.time()
        for step in range(start, self.total_step):
            self.G.train()
            try:
                imgs, labels = next(data_iter)
                
            except:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            size = labels.size()

            imgs = imgs.to(self.device)

            print('PREDICTING')
            # ================== Train G =================== #
            if not self.dlv3:
                labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
                labels_real_plain = labels[:, 0, :, :].to(self.device)
                
                labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
                oneHot_size = (size[0], 3, size[2], size[3])
                labels_real = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(self.device)
                labels_real = labels_real.scatter_(1, labels.data.long().to(self.device), 1.0)
                labels_predict = self.G(imgs)
                c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                labels_predict_dice = labels_predict.argmax(dim=1)

            elif self.dlv3:
                labels_predict = self.G(imgs)['out']
                labels_real_plain = labels[:, 0, :, :].to(self.device)
            
                # Create labels_real for DeepLabV3
                oneHot_size = (size[0], 3, size[2], size[3])
                labels_real = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(self.device)
                labels_real = labels_real.scatter_(1, labels.data.long().to(self.device), 1.0)
                
                c_loss = cross_entropy2d(labels_predict, labels_real_plain.long())
                labels_predict_dice = labels_predict.argmax(dim=1)

            # Calculate cross entropy loss
            print(f'CEL: {c_loss}')

            ####### MAKE DICE SCORE COEFFICIENT MAKE SENSE FOR BATCHES HERE
            dice_score_class_1 = dice_coefficient(labels_predict_dice, labels_real_plain.long(), class_index=1)
            print(f"DICE SCORE FOR CLASS 1 : {dice_score_class_1.item()}")
            
            # num_plots = min(5, self.batch_size)
            # fig, axs = plt.subplots(num_plots, 2, figsize=(10, 5*num_plots))

            # for i in range(num_plots):
            #     # Plot predicted labels
            #     axs[i, 0].imshow(labels_predict_dice[i].cpu().numpy(), cmap='viridis')
            #     axs[i, 0].set_title(f'Predicted Labels (Batch {i+1})')
            #     axs[i, 0].axis('off')
                
            #     # Plot ground truth labels
            #     axs[i, 1].imshow(labels_real_plain[i].cpu().numpy(), cmap='viridis')
            #     axs[i, 1].set_title(f'Ground Truth Labels (Batch {i+1})')
            #     axs[i, 1].axis('off')

            # plt.tight_layout()            
            # plt.savefig(f'{step}_iris_log.jpg')
            
 
            self.reset_grad()
            c_loss.backward()
            self.g_optimizer.step()
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], Cross_entrophy_loss: {:.4f}".
                      format(elapsed, step + 1, self.total_step, c_loss.data))
                # wandb.log({"Cross_entropy_loss": c_loss.data, "Dice_score": dice, "step": step + 1})
                wandb.log({"Cross_entropy_loss": c_loss.item(), "Dice_score_class_1": dice_score_class_1.item(), "step": step + 1})



            label_batch_predict = generate_label(labels_predict, self.imsize)
            label_batch_real = generate_label(labels_real, self.imsize)

 
            # image infor on tensorboardX
            img_combine = imgs[0]
            real_combine = label_batch_real[0]
            predict_combine = label_batch_predict[0]
            for i in range(1, len(imgs)):
                img_combine = torch.cat([img_combine, imgs[i]], 2)
                real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)


            # Sample images
            if (step + 1) % self.sample_step == 0:
                
                if self.dlv3:
                    labels_sample = self.G(imgs)['out']
                else:
                    labels_sample = self.G(imgs)

                print(len(imgs))
                print(i)
                labels_sample = generate_label(labels_sample, self.imsize)
                #might need to coment below line out 
                # labels_sample = torch.from_numpy(labels_sample)
                # save_image(denorm(labels_sample.data),
                            # os.path.join(self.sample_path, '{}_predict.png'.format(step + 1)))
                img_log = wandb.Image((img_combine.data + 1) / 2.0, caption="Images")
                real_log = wandb.Image(real_combine, caption="Real Labels")
                predict_log = wandb.Image(predict_combine, caption="Predictions")
                                
                wandb.log({"Sample Images": img_log, "Real Labels": real_log, "Predictions": predict_log}, step=step + 1)
                
            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                            os.path.join(self.model_save_path, f'{step + 1}_OPTIMAL_HP_train_limit_{self.train_limit}.pth'))
        wandb.finish()



    def build_model(self):
        if not self.dlv3:
            self.G = unet().to(self.device)
        elif self.dlv3:
            self.G = deeplabv3_resnet101(num_classes=3).to(self.device)
 
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])


    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
