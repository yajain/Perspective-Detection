import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PerspectiveDataset(Dataset):
    def __init__(self, imgs_path, label_file):
        self.img_dim = (512, 512)
        self.data = []

        # Make a list of all the labels from label_file and store in labels

        for index, image in enumerate(os.listdir(imgs_path)):
            self.data.append([image, labels[index]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        target = np.asarray(label)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        # In this line convert target to a torch tensor
        return img_tensor, label_tensor
