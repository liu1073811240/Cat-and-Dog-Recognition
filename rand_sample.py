import os, random, shutil
import numpy as np


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    print(pathDir)
    # exit()

    filenumber = len(pathDir)
    # print(filenumber)
    rate = 0.8  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    print(len(sample))
    # exit()
    for name in sample:  # 取出4800张图片
        print(name)  # 0.2760.jpeg
        # exit()
        shutil.move(fileDir + "\\" +name, tarDir + "\\" +name)

    for name in pathDir:  # 再遍历一次猫所有的图片，如果该图片没有被取样为训练集，则把该样本放入测试集
        if name not in sample:
            print(name)
            shutil.move(fileDir + "\\" + name, tarDir2 + "\\" +name)
        else:
            pass

    return


if __name__ == '__main__':
    # fileDir = r"G:\cat_dog2\Cat"  # 源图片文件夹路径
    # tarDir = r'G:\img\TRAIN\CAT'  # 移动到新的文件夹路径
    # tarDir2 = r"G:\img\TEST\CAT"

    fileDir = r"G:\cat_dog2\Dog"
    tarDir = r"G:\img\TRAIN\DOG"
    tarDir2 = r"G:\img\TEST\DOG"

    moveFile(fileDir)
