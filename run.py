import argparse
import yaml
import random

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
import torch.backends.cudnn as cudnn

from ImageClassifier import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



### Loading Config YAML
parser = argparse.ArgumentParser(description='Generic Runner for Image Classification')
parser.add_argument('--project', '-p', dest="folder", help='Path to project folder')
parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='path to the config file', default='./configs/mnist.yaml')
args = parser.parse_args()
config = yaml.safe_load(open(os.path.join('./configs', args.folder, args.filename), 'r'))
for k in config: exec('{KEY} = {VALUE}'.format(KEY = k, VALUE = repr(Namespace(**config[k]))))
   
### For reproducibility
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
cudnn.deterministic = True

### Logging
logger = TensorBoardLogger(**vars(logging_opt))

### Checkpointing
ckpt_callback = ModelCheckpoint(filepath=os.path.join(logger.save_dir, logger.name, logger.version, '{epoch:02d}-{val_loss:.2f}'), **vars(ckpt_opt))

### Setting Up Logger, Image Classification Model, PL Trainer
model = ImageClassifier(opt)
trainer = pl.Trainer(default_save_path=f"{logger.save_dir}",
                     logger=logger,
					 checkpoint_callback=ckpt_callback,
                     **vars(trainer_opt))

print(f"======= Training {logging_opt.name} =======")
trainer.fit(model)