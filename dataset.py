from . import tiny-imagenet-200

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

    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    img_size = 32
    num_channels = 3
    num_classes = 10

    cifar10DataMgr = datasetmgr.DatasetManager(data_url, "cifar-10-data", "cifar-10-batches-py", img_size, num_channels, num_classes)

    # Download and extract CIFAR-10 data
    cifar10DataMgr.maybe_download_and_extract()

    # training data
    x_train, y_train = cifar10DataMgr.load_training_data()
    train = CIFAR10Sequence(x_train, y_train, transform=AUGMENTATIONS_TRAIN)

    # Validation data
    x_val, y_val = cifar10DataMgr.load_validation_data(5000)
    test = CIFAR10Sequence(x_val, y_val, transform=AUGMENTATIONS_TEST)

    return train, test
