from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import random


def get_train_test_dataset(name, augmented):
    if name == 'mnist':
        print('Wow such mnist')
        p = 0.7 if augmented else 0
        train = MNIST(
            '../data/mnist',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomApply(
                    # [transforms.RandomAffine(degrees=10, translate=(0, 0.1))],
                    [transforms.RandomRotation(10)],
                    p
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))

            ]),
        )
        test = MNIST(
            '../data/mnist', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    elif name == 'fashion-mnist':
        print('Wow such fashion')
        p = 0.8 if augmented else 0
        train = FashionMNIST(
            '../data/fashion-mnist',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomRotation(10)
                    # , transforms.RandomHorizontalFlip(0.5)
                ], p),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        test = FashionMNIST(
            '../data/fashion-mnist', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    elif name == 'cifar10':
        p = 0.9 if augmented else 0
        print('Wow such cifar10')
        train = CIFAR10(
            '../data/cifar10', train=True, download=True, transform=transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomAffine(degrees=5),
                    transforms.RandomHorizontalFlip()
                ], p),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784)
                )
            ])
        )
        test = CIFAR10(
            '../data/cifar10', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784)
                )
            ])
        )
    else:
        raise NotImplemented

    return train, test
