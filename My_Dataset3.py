import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class Mydataset(Dataset):
    def __init__(self, main_dir, is_train=True):
        self.datasets = []
        train_data = np.load(r"./train.npy")
        test_data = np.load(r"./test.npy")
        data_filename = train_data if is_train else test_data
        for img_data in os.listdir(main_dir):  # 无法直接拿到载入的数据，跟原始数据进行判断再去取
            if img_data in data_filename:
                img_path = "{}/{}".format(main_dir, img_data)  # 拼接图片路径
                label = img_data.split(".")[0]  # 取标签

                self.datasets.append([img_path, label])  # [['G:\\cat_dog\\img/0.1.jpeg', '0']，...]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        data = self.datasets[item]
        data_arr = (np.array(Image.open(data[0])) / 255 - 0.5) / 0.5
        data_arr = np.transpose(data_arr, [2, 0, 1])
        data_tensor = torch.tensor(data_arr, dtype=torch.float32)
        label = torch.tensor(int(data[1]))

        return data_tensor, label


if __name__ == '__main__':
    data_path = r"G:\cat_dog\img"
    dataset = Mydataset(data_path, True)
    dataloader = DataLoader(dataset, 128, True)
    for img, label in dataloader:
        print(img.shape)
        print(label.shape)
