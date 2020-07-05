from argparse import Namespace
from tqdm import tqdm

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms


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

    do_shuffle = True if split != 'test' else False
    batch_size = opt.batch_size if split != 'test' else 1
    data_loader = DataLoader(dataset=dataset,
                             num_workers=opt.num_workers,
                             batch_size=batch_size, 
                             shuffle=do_shuffle, 
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