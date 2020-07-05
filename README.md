
# README

Basic scaffold code for image classification using Pytorch Lightning.

### Dependencies
- pytorch (1.5.0)
- pytorch_lightning (0.8.4)

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

### MNIST Image Classification
FC layers.
```bash
python run.py -p mnist -c mnist_fc
```

CNN layers
```bash
python run.py -p mnist -c mnist_fc
```