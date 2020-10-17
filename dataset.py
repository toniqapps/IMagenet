from . import tinyimagenet

import torchvision
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

def tiny_imagenet_albumentations():
    AUGMENTATIONS_TRAIN = A.Compose([
                                A.RandomCrop(32, 32),
                                A.HorizontalFlip(),
                                A.Cutout (num_holes=1, max_h_size=8, max_w_size=8, fill_value=0.4733, always_apply=False, p=0.5),
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    AUGMENTATIONS_TEST = A.Compose([
                                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                ToTensorV2()
                                ])

    train_set = torchvision.datasets.ImageFolder(root='IMagenet/tiny-imagenet-200/new_train',
			transform=AUGMENTATIONS_TRAIN)
    test_set = torchvision.datasets.ImageFolder(root='IMagenet/tiny-imagenet-200/new_test',
        transform=AUGMENTATIONS_TEST)
		

    return train_set, test_set
