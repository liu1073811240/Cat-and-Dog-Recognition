import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from My_Dataset3 import Mydataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(100*100*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.linear4 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        y = self.linear1(x)
        y = self.linear2(y)
        y = self.linear3(y)
        y = self.linear4(y)

        return y


if __name__ == '__main__':

    img_path = r"G:\cat_dog\img"

    train_dataset = Mydataset(img_path, True)
    test_dataset = Mydataset(img_path, False)

    train_dataloader = DataLoader(train_dataset, 100, True)
    test_dataloader = DataLoader(test_dataset, 100, True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())

    for epoch in range(2):
        train_loss = 0
        train_acc = 0
        for i, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)

            out = net(img)
            loss = loss_fn(out, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()*label.size(0)

            argmax = torch.argmax(out, 1)
            train_acc += (argmax == label).sum().item()

            if i % 10 == 0:
                print("epoch:{}, train_loss:{}".format(epoch, loss.item()))

        validate_loss = 0
        validate_acc = 0
        for i, (img, label) in enumerate(test_dataloader):
            img = img.to(device)
            label = label.to(device)

            out = net(img)
            _loss = loss_fn(out, label)

            # validate_loss += _loss.item()*label.size(0)
            validate_loss += _loss.detach().cpu().numpy() * label.size(0)  # 计算损失放到cpu上运算, 节省GPU资源

            argmax = torch.argmax(out, 1)
            validate_acc += (argmax == label).sum().item()
            if i % 10 == 0:
                print("validate_loss:{}".format(_loss.item()))

        mean_loss = validate_loss / len(test_dataset)
        mean_acc = validate_acc / len(test_dataset)
        print("平均精度为：", mean_acc)
        print("平均损失为：", mean_loss)

    net.eval()
    test_loss = 0
    test_acc = 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)

        out = net(img)
        loss = loss_fn(out, label)

        test_loss += loss.item() * label.size(0)
        argmax = torch.argmax(out, 1)
        test_acc += (argmax == label).sum().item()

    test_avgloss = test_loss / len(test_dataset)
    test_avgacc = test_acc / len(test_dataset)

    print("test_avgloss:", test_avgloss)
    print("test_avgacc:", test_avgacc)





