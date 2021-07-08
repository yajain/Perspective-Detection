import os
import cv2
import numpy as np
import torch
import pandas as pd

class PerspectiveDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, label_file):
        self.img_dim = (512, 512)
        self.data = []
        df = pd.read_csv(label_file, header = None)
        with open (img_paths, 'r') as file:
            for index, image in enumerate(file):
                image = image[0:-1]
                self.data.append([image, list(df.iloc[index])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        
        # forming image tensors
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # normalizing image tensors to be between 0 and 1
        img_tensor = (img_tensor - img_tensor.min())/img_tensor.max()
        
        # forming label tensors
        label_tensor = torch.tensor(target)
        
        # adjusting label coords according to default image dimension
        for i in range(8):
            if i%2 == 0:
                label_tensor[i] = label_tensor[i]*self.img_dim[0]/width
            else:
                label_tensor[i] = label_tensor[i]*self.img_dim[1]/height
                
        # normalizing label tensors to be between 0 and 1
        label_tensor = (label_tensor - label_tensor.min())/label_tensor.max()
        
        return img_tensor, label_tensor

if __name__ == "__main__":
    train_set = PerspectiveDataset('train_img_paths.txt', 'train.csv')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_set = PerspectiveDataset('test_img_paths.txt', 'test.csv')
    testloader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)
