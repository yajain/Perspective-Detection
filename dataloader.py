import os
import cv2
import numpy as np
import torch
#from torch.utils.data import Dataset, DataLoader

class PerspectiveDataset(Dataset):
    def __init__(self, img_paths, label_file):
        self.img_dim = (1024, 1024)
        self.data = []
        df = pd.read_csv('data.csv', header = None)
        for index, image in enumerate(os.listdir(img_paths)):
            self.data.append([image, list(df.iloc(index)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        label_tensor = torch.tensor(target)
        return img_tensor, label_tensor

if __name__ == "__main__":
    dataset = PerspectiveDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
