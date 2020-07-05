"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
from argparse import Namespace

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from networks import define_net
from datasets import define_dataloader

import pdb

class ImageClassifier(LightningModule):

	def __init__(self, opt: Namespace):
		super(ImageClassifier, self).__init__()
		self.opt = opt
		self.net = define_net(opt)
		self.loss_fun = nn.CrossEntropyLoss()

	def forward(self, x):
		return self.net.forward(x)

	def training_step(self, batch, batch_idx):
		data, target = batch
		output = self.net.forward(data)
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
		data, target = batch
		output = self.net.forward(data)
		loss_val = self.loss_fun(output, target)
		acc1 = self.__accuracy(output, target, topk=(1,))

		output = OrderedDict({
			'y_pred': output,
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
				  'val_acc1': tqdm_dict['val_acc1'],
				  'auc': tqdm_dict['auc']}
		return result

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)
		return [optimizer]

	def train_dataloader(self):
		train_dataloader = define_dataloader(split='train', opt=self.opt)
		return train_dataloader

	def val_dataloader(self):
		val_dataloader = define_dataloader(split='val', opt=self.opt)
		return val_dataloader

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

