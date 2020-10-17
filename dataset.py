from . import tinyimagenet

import torchvision
from torchvision import datasets

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def albumentations_transforms(is_train=False):
  # Mean and standard deviation of train dataset
  mean = np.array([0.4914, 0.4822, 0.4465])
  std = np.array([0.2023, 0.1994, 0.2010])
  transforms_list = []

  if is_train:
    transforms_list.extend([
                            A.RandomCrop(32, 32),
                            A.HorizontalFlip(),
                            A.Cutout (num_holes=1, max_h_size=8, max_w_size=8, fill_value=0.4733, always_apply=False, p=0.5)  
                            ])
  
  
  transforms_list.extend([
                          A.Normalize(mean=mean, std=std),
                          ToTensorV2()
                          ])
  data_transforms = A.Compose(transforms_list)
  return lambda img: data_transforms(image=np.array(img))["image"]

def tiny_imagenet_albumentations(root='IMagenet/tiny-imagenet-200/', root_train='new_train', root_test='new_test'):
  # tinyimagenet.main()
  train_transform = albumentations_transforms(is_train=True)
  test_transform = albumentations_transforms()
  
  train_set = torchvision.datasets.ImageFolder(root=root + root_train,
                                               transform=train_transform)
  test_set = torchvision.datasets.ImageFolder(root=root + root_test,
                                              transform=test_transform)
  
  return train_set, test_set
