
# README

Basic scaffold code for image classification using Pytorch Lightning. Largely adapted from:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py

but with out-the-block MNIST, CIFAR10, and ImageFolder datasets in torchvision for posterity. You may need to change the data path in the configs to point to where you have MNIST / CIFAR10 downloaded (else, set download=True in the DataLoader).

### Dependencies
- pytorch (1.5.0)
- pytorch_lightning (0.7.3)

### GitHub Code Organization
```bash
./
├── configs
      └── examples
            ├── mnist_cpu.yaml
            ├── mnist_gpu1.yaml
            ├── mnist_gpu4.yaml
            ├── ...
      └── EXPERIMENT NAME
            ├── ...
└── logs
      └── examples
            ├── version_mnist_cpu
                ├── ...
            ├── ...
```
**configs** directory contains your list of experiment folders. Under each experiment, is a list of config yaml files (contains all of your arguments). Results from your experiments should be in **logs**. 

To run:
```bash
python run.py -p EXPERIMENT NAME -c CONFIG YAML NAME
```

### MNIST Image Classification
CPU spport.
```bash
python run.py -p examples -c mnist_cpu
```

GPU (single GPU) support.
```bash
python run.py -p examples -c mnist_gpu1
```

GPU (multi-GPU) support (using DDP)
```bash
python run.py -p examples -c mnist_gpu4
```

### CIFAR10 Image Classification
CPU spport.
```bash
python run.py -p examples -c cifar10_cpu
```

GPU (single GPU) support.
```bash
python run.py -p examples -c cifar10_gpu1
```

GPU (multi-GPU) support (using DDP)
```bash
python run.py -p examples -c cifar10_gpu4
```
