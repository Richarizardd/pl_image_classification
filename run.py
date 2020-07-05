from argparse import ArgumentParser, Namespace
import os, shutil
import yaml

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from ImageClassifier import *

### Helper Functions
def printConfig(config: dict):
    print('------------------- Options -----------------')
    for key1 in config.keys():
        print(key1+':')

        for val1 in config[key1]:
            if isinstance(config[key1][val1], dict):
                print('\t%s: ' % (val1))
                for val2 in config[key1][val1]:
                    print('\t\t%s: ' % val2 + str(config[key1][val1][val2]))
            else:
                print('\t%s: ' % val1 + str(config[key1][val1]))
        print()
    print('------------------- End -------------------')

### Loading Config YAML
parser = ArgumentParser(description='Generic Runner for Image Classification')
parser.add_argument('--project', '-p', default='mnist', help='Path to project folder')
parser.add_argument('--config', '-c', default='mnist_fc.yaml', help='path to the config file')#, default='./configs/mnist.yaml')
args = parser.parse_args()
config_src = os.path.join('./configs', args.project, args.config)
config = yaml.safe_load(open(config_src, 'r'))
printConfig(config)


for k in config:
    exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(Namespace(**config[k]))))
 
### Initialize Logs + Runs
if logging_opt.name not in os.listdir(logging_opt.save_dir):
    os.makedirs(os.path.join(logging_opt.save_dir, logging_opt.name))
versions = os.listdir(os.path.join(logging_opt.save_dir, logging_opt.name))
if '.DS_Store' in versions: versions.remove('.DS_Store')
if len(versions) == 0:
    next_version = -1
else:
    next_version = int(sorted(versions)[-1][:3])
logging_opt.version = '{:03d}-'.format(next_version+1) + logging_opt.version
config_dest = os.path.join(logging_opt.save_dir, logging_opt.name, logging_opt.version)
os.makedirs(config_dest)
shutil.copyfile(config_src, os.path.join(config_dest, args.config))

### For reproducibility
seed_everything(opt.seed)

### Logging
logger = TensorBoardLogger(**vars(logging_opt))

### Checkpointing
ckpt_callback = ModelCheckpoint(filepath=os.path.join(logger.save_dir, logger.name, logger.version, '{epoch:02d}-{loss:.2f}'), **vars(ckpt_opt))

### Setting Up Logger, Image Classification Model, PL Trainer
model = ImageClassifier(opt)
trainer = Trainer(logger=logger,
				  checkpoint_callback=ckpt_callback,
                  **vars(trainer_opt))

print(f"======= Training {logging_opt.name} =======")
trainer.fit(model)
trainer.save_checkpoint(os.path.join(logger.save_dir, logger.name, logger.version, 'latest.ckpt'))