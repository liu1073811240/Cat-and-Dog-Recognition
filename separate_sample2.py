import os, random, shutil

# 在所有图片中随机移动一部分图片出来
def classify(path1, path2):
    name_list = os.listdir(path1)  # ['0.1.jpeg', '0.10.jpeg', '0.100.jpeg'...]
    num = int(len(name_list) * 0.2)  # 2400
    sample = random.sample(name_list, num)  # 从12000图片中随机取出2400张图片出来
    print(sample)  # ['1.179.jpeg',...]
    for name in sample:
        # shutil.copy()
        shutil.move(os.path.join(path1, name), os.path.join(path2, name))
        # 从一个文件夹将图片移动另一个文件中


if __name__ == '__main__':
    path1 = r"G:\cat_dog2\img"
    path2 = r"G:\cat_dog3\train"
    classify(path1, path2)
