import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets, utils


def visualize(batch, title, max_images=64):
    plt.figure(figsize=(12, 12))
    plt.title(title)
    plt.imshow(np.transpose(utils.make_grid(batch[:max_images], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.axis('off')
    plt.show()


def get_dataloader(dataroot, image_size, batch_size, shuffle=True):
    initial_dataset = datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.Resize(image_size),
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

    mirror_dataset = datasets.ImageFolder(root=dataroot,
                                          transform=transforms.Compose([
                                              transforms.Resize(image_size),
                                              transforms.CenterCrop(image_size),
                                              transforms.RandomHorizontalFlip(p=1.0),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))

    dataset = ConcatDataset([initial_dataset, mirror_dataset])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
