
# README

Basic scaffold code for image classification using Pytorch Lightning. Largely adapted from:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py

but with out-the-block MNIST, CIFAR10, and ImageFolder datasets in torchvision for posterity. You may need to change the data path in the configs to point to where you have MNIST / CIFAR10 downloaded (else, set **download=True**).

### Dependencies
- pytorch (1.5.0)
- pytorch_lightning (0.8.4)
- tensorboard (2.1.1)

### GitHub Code Organization
```bash
./
├── configs
      └── mnist
            ├── mnist_fc.yaml
            ├── mnist_cnn.yaml
            ├── mnist_gpu1.yaml
            ├── ...
      └── cifar10
            ├── ...
└── logs
      └── mnist
            ├── 000-mnist_fc
                ├── ...
            ├── 001-mnist_cnn
            	├── ...
            ├── ...
```
**configs** directory contains your list of experiment folders. Under each experiment, is a list of config yaml files (contains all of your arguments). Results from your experiments should be in **logs**. 

To run:
```bash
python run.py -p EXPERIMENT NAME -c CONFIG YAML NAME
```

To start Tensorboard:
```bash
tensorboard --logdir ./
```

### MNIST Image Classification
FC layers.
```bash
python run.py -p mnist -c mnist_fc
```

CNN layers
```bash
python run.py -p mnist -c mnist_fc
```
