import os,  random
import numpy as np

def classify(path):
    name_list = os.listdir(path)
    num = int(len(name_list)*0.8)

    train_list = random.sample(name_list, num)
    test_list = list(set(name_list) - set(train_list))  # 将总样本、训练集转成集合方便相减得到测试集

    np.save("./train", np.array(train_list))
    np.save("./test", np.array(test_list))

    train_data = np.load("./train.npy")
    test_data = np.load("./test.npy")
    print(train_data, len(train_data))
    print(test_data, len(test_data))


if __name__ == '__main__':
    path = r"G:\cat_dog\img"
    classify(path)