"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
from argparse import Namespace
import os
import pdb
import random
from collections import OrderedDict
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.loggers import TensorBoardLogger

class ImageClassifier(LightningModule):
    def __init__(self, opt: Namespace):
        """
        TODO: add docstring here
        """
        super().__init__()
        self.opt = opt
        self.model = define_model(opt)
        self.loss_fun = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model.forward(x)


    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss_train = self.loss_fun(output, target)
        acc1 = self.__accuracy(output, target, topk=(1,))

        tqdm_dict = {'train_loss': loss_train}
        output = OrderedDict({
            'loss': loss_train,
            'acc1': acc1,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output


    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss_val = self.loss_fun(output, target)
        acc1 = self.__accuracy(output, target, topk=(1,))

        output = OrderedDict({
            'y_pred': out,
            'y_label': target,
            'val_loss': loss_val,
            'val_acc1': acc1,
        })

        return output


    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        y_pred_all = torch.stack([output['y_pred'] for output in outputs]).detach().cpu().numpy()
        y_label_all = torch.stack([output['y_label'] for output in outputs]).detach().cpu().numpy()
        y_pred_all = y_pred_all[:,:,:self.opt.num_class]
        y_pred_all = np.reshape(y_pred_all, (-1, self.opt.num_class))
        y_label_all = np.reshape(y_label_all, (-1))
        y_label_all_oh = LabelBinarizer().fit_transform(y_label_all)
        tqdm_dict['auc'] = roc_auc_score(y_label_all_oh, y_pred_all, "micro")

        for metric_name in ["val_loss", "val_acc1"]:
            tqdm_dict[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        result = {'progress_bar': tqdm_dict, 
                  'log': tqdm_dict, 
                  'val_loss': tqdm_dict["val_loss"], 
                  'val_acc1': tqdm_dict['val_acc1']
                  'auc': tqdm_dict['auc']}
        return result


    def test_step(self, batch, batch_idx):
        images, target = batch
        out = self.forward(images)
        loss_val = self.loss_fun(out, target)
        acc1 = self.__accuracy(out, target, topk=(1,))

        output = OrderedDict({
            'y_pred': out,
            'y_label': target,
            'test_loss': loss_val,
            'test_acc1': acc1,
        })

        return output


    def test_epoch_end(self, outputs):
        tqdm_dict = {}
        y_pred_all = torch.stack([output['y_pred'] for output in outputs]).detach().cpu().numpy()
        y_label_all = torch.stack([output['y_label'] for output in outputs]).detach().cpu().numpy()
        y_pred_all = y_pred_all[:,:,:self.opt.num_class]
        y_pred_all = np.reshape(y_pred_all, (-1, self.opt.num_class))
        y_label_all = np.reshape(y_label_all, (-1))
        y_label_all_oh = LabelBinarizer().fit_transform(y_label_all)
        tqdm_dict['auc'] = roc_auc_score(y_label_all_oh, y_pred_all, "micro")

        for metric_name in ["test_loss", "test_acc1"]:
            tqdm_dict[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        result = {'progress_bar': tqdm_dict, 
                  'log': tqdm_dict, 
                  'test_loss': tqdm_dict["test_loss"],
                  'val_acc1': tqdm_dict['val_acc1'],
                  'auc': tqdm_dict['auc']}
        return result


    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

            if len(res) == 1: return res[0]
            return res


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)
        return [optimizer]


    def train_dataloader(self):
        train_dataloader = define_dataloader(split='train', opt=self.opt)
        return train_dataloader


    def val_dataloader(self):
        val_dataloader = define_dataloader(split='val', opt=self.opt)
        return val_dataloader


    def test_dataloader(self):
        test_dataloader = define_dataloader(split='test', opt=self.opt)
        return test_dataloader



### Utils
def define_model(opt: Namespace):
    if opt.net_arch in models.__dict__.keys() and callable(models.__dict__[opt.net_arch]):
        return models.__dict__[opt.net_arch](pretrained=opt.pretrained)
    elif opt.net_arch == 'mnist_toy':
        return MNIST_ToyNet()
    elif opt.net_arch == 'cifar_toy':
        return CIFAR_ToyNet()
    else:
        raise ValueError('Undefined model type')


def define_dataloader(opt: Namespace, split: str='train'):
    transform = define_transform(opt=opt, split=split)

    if opt.dataset == 'mnist':
        print("Loading MNIST...")
        dataset = MNIST(root=opt.data_path, 
                        train=True if split == 'train' else False, 
                        transform=transform, download=False)
    elif opt.dataset == 'cifar10':
        print("Loading CIFAR10...")
        dataset = CIFAR10(root=opt.data_path, 
                          train=True if split == 'train' else False, 
                          transform=transform, download=False)
    elif opt.dataset == 'imagefolder':
        dataset = ImageFolder(root=opt.train_path if split == 'train' else opt.val_path, transform=transform)
    else:
        raise ValueError('Undefined dataset type')

    data_loader = DataLoader(dataset=dataset,
                             num_workers=opt.num_workers,
                             batch_size=opt.batch_size, 
                             shuffle=True if split == 'train' else False, 
                             drop_last=True)
    return data_loader


def define_transform(opt: Namespace, split: str='train'):
    if opt.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    elif opt.dataset == 'cifar10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        crop = transforms.RandomCrop if split == 'train' else transforms.CenterCrop
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((288, 288)),
                                        crop((opt.load_size, opt.load_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform



##### Custom Networks
class View(nn.Module):
    # https://github.com/pytorch/vision/issues/720
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


class MNIST_ToyNet(nn.Module):
    def __init__(self, hidden_dim=28):
        super(MNIST_ToyNet, self).__init__()

        self.net = nn.Sequential(
            View((-1, 784)),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class CIFAR_ToyNet(nn.Module):
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(CIFAR_ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
