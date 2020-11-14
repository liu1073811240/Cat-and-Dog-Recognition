import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from My_Dataset import MyDataset
from torch.optim import sgd, adam, adagrad, rmsprop, adadelta, adamax, adamw, sparse_adam, asgd
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=3*100*100, out_features=2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),  # N, H, W
            # nn.LayerNorm(512),  # C, H, W
            # nn.InstanceNorm1d(512),  # H, W  (要求输入数据三维)
            # nn.GroupNorm(2, 512)  # C, H, W,  将512分成两组
            nn.ReLU()
        )  # N, 512
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )  # N, 256
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )  # N, 128
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=2),
        )  # N, 10

    def forward(self, x):
        # x = torch.reshape(x, [1, x.size(0), -1])  # 形状[1, N, C*H*W]
        # print(x.shape)
        # y1 = self.layer1(x)[0]   # 这两行代码适用于在InstanceNorm1d的情况。将第一维去掉，变成两维

        x = torch.reshape(x, [x.size(0), -1])  # 形状[N, C*H*W]
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        self.y4 = self.layer4(y3)
        out = torch.softmax(self.y4, 1)

        return out


if __name__ == '__main__':
    batch_size = 100
    # 加载本地数据集
    data_path = r"G:\cat_dog1"
    train_data = MyDataset(data_path, True)
    test_data = MyDataset(data_path, False)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load("./cat_dog_params.pth"))
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
                # plt.pause(0.01)
                print("Epoch:{}, batch:{}/110, loss:{:.3f}".format(epoch, int(i), loss.item()))

        # print(a)
        torch.save(net.state_dict(), "./cat_dog_params.pth")
    #     # torch.save(net, "./cat_dog_net.pth")

        if epoch % 5 == 0:  # 每训练完五轮打印一次精度
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
            print("loss:{:.3f}, Acc:{:.3f}".format(mean_loss, mean_acc))
























