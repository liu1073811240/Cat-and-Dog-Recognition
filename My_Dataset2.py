import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, main_dir, is_train=True):
        self.datasets = []
        data_filename = "train" if is_train else "test"
        for img_data in os.listdir(os.path.join(main_dir, data_filename)):
            img_path = os.path.join(main_dir, data_filename, img_data)
            label = img_data.split(".")[0]
            # print(label)
            self.datasets.append([img_path, label])  # [['G:\\cat_dog3\\train\\0.1.jpeg', '0'],...]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        data = self.datasets[item]
        # data_tensor = self.trans(Image.open(data[0]))
        data_arr = (np.array(Image.open(data[0])) / 255 - 0.5) / 0.5
        data_arr = np.transpose(data_arr, [2, 0, 1])
        data_tensor = torch.tensor(data_arr, dtype=torch.float32)

        label = torch.tensor(int(data[1]))
        return data_tensor, label

    def trans(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(x)

if __name__ == '__main__':
    data_path = r"G:\cat_dog3"
    dataset = MyDataset(data_path)
    dataloader = DataLoader(dataset, 128, True)
    for img, label in dataloader:
        print(img.shape)
        print(label.shape)