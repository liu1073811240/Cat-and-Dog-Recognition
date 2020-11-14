import cv2
import os


# 将猫和狗的图片文件夹分别放在两个文件夹中
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        # print(filename)  # 0.1.jpeg

        strs = filename.strip().split(".")  # ['0', '1', 'jpeg']
        strs = list(filter(bool, strs))  # ['0', '1', 'jpeg']
        if strs[0] == "0":

            img = cv2.imread(directory_name + "/" + filename)  # 读取图片数据
            # print(img)

            #####显示图片#######
            # cv2.imshow(filename, img)
            # cv2.waitKey(0)
            #####################

            #####保存图片#########
            cv2.imwrite("G:\cat_dog2\Cat" + "/" + filename, img)

        elif strs[0] == "1":
            img = cv2.imread(directory_name + "/" + filename)
            # print(img)

            #####显示图片#######
            # cv2.imshow(filename, img)
            # cv2.waitKey(0)
            #####################

            #####保存图片#########
            cv2.imwrite("G:\cat_dog2\Dog" + "/" + filename, img)

        else:
            pass


read_directory("G:\cat_dog\img")