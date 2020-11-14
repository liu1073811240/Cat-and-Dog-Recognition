import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch import softmax
from torchvision import transforms
from My_Dataset import MyDataset
from torch.optim import sgd, adam, adagrad, rmsprop, adadelta, adamax, adamw, sparse_adam, asgd
import cv2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*32*50*50

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*64*25*25

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*128*12*12

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*256*6*6

        self.conv_45 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # batch*512*3*3

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(),
        )  # batch*32*1*1

        self.fcn = nn.Sequential(
            nn.Linear(in_features=128 * 1 * 1, out_features=2),
            nn.BatchNorm1d(num_features=2)
        )

    def forward(self, x):

        y1 = self.conv_1(x)
        y2 = self.conv_2(y1)

        y3 = self.conv_3(y2)
        # print(y2.shape)
        y4 = self.conv_4(y3)
        # print(y3.shape)

        y_45 = self.conv_45(y4)
        y5 = self.conv_5(y_45)

        y5 = y5.reshape(y5.size(0), -1)
        # print(y3.shape)
        y4 = softmax(self.fcn(y5), 1)

        return y4


if __name__ == '__main__':
    batch_size = 128
    # 加载本地数据集
    data_path = r"G:\img"
    train_data = MyDataset(data_path, True)
    test_data = MyDataset(data_path, False)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load("./Param_dir3/cat_dog_params65.pth"))
    # net = torch.load("./cat_dog_net.pth").to(device)

    loss_function = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.5, dampening=0,
    #                             weight_decay=0,  nesterov=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                                 weight_decay=0, amsgrad=False)  # betas:0.9越大越平滑， 默认效果最好
    # weight_decay:表示正则化系数

    # optimizer = adagrad.Adagrad(net.parameters())
    # optimizer = adadelta.Adadelta(net.parameters())
    # optimizer = rmsprop.RMSprop(net.parameters())
    # optimizer = sgd.SGD(net.parameters(), 1e-3)
    # optimizer = adam.Adam(net.parameters())

    a = []
    b = []
    plt.ion()
    net.train()
    for epoch in range(100):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            output = net(x)

            # print(x.shape)
            # print(output[0])  # 一张图片经过神经网络输出的十个值
            # print(output.shape)  # torch.Size([100, 10])
            # print(y)
            # 在1轴里面填1， 同时将标签形状变为（N, 1）
            y = torch.zeros(y.cpu().size(0), 2).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)
            # print(y)
            # print(y.size(0))
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                a.append(i + (epoch*(len(train_data) / batch_size)))
                b.append(loss.item())
                # plt.clf()
                # plt.plot(a, b)
                # plt.pause(1)
                print("Epoch:{}, batch:{}, loss:{:.5f}".format(epoch, int(i), loss.item()))

        # print(a)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), "./Param_dir3/cat_dog_params{0}.pth".format(epoch))
        torch.save(net, "./cat_dog_net.pth")

        if epoch % 5 == 0:
            net.eval()
            eval_loss = 0
            eval_acc = 0
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                out = net(x)

                y = torch.zeros(y.cpu().size(0), 2).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)
                loss = loss_function(out, y)
                # print("Test_Loss:{:.3f}".format(loss.item()))

                eval_loss += loss.item()*y.size(0)
                arg_max = torch.argmax(out, 1)
                y = y.argmax(1)
                eval_acc += (arg_max==y).sum().item()

            mean_loss = eval_loss / len(test_data)
            mean_acc = eval_acc / len(test_data)

    # print(y)
    # print(torch.argmax(out, 1))
            print("loss:{:.5f}, Acc:{:.3f}".format(mean_loss, mean_acc))
























