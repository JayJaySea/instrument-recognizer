import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
from torch import nn

def init_transforms():
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ],
    )

def init_train_dataset(trans):
    train_data_path = "../dataset/images/train"

    return torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=trans,
        is_valid_file=check_image
    )

def init_val_dataset(trans):
    val_data_path = "../dataset/images/val"

    return torchvision.datasets.ImageFolder(
        root=val_data_path,
        transform=trans,
        is_valid_file=check_image
    )

def init_test_dataset(trans):
    test_data_path = "../dataset/images/test"

    return torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=trans,
        is_valid_file=check_image
    )

def check_image(path):
    try:
        Image.open(path)
        return True
    except:
        return False

def init_resnet():
    resnet = models.resnet50()

    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 5)
    )

    return resnet
