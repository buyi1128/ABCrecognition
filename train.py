import sys
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim

from characterLoader import characterLoader
from nets.NewNet import NewNet


bs = 8
lr = 0.0001
epoch = 50
stepLength = 20

trainpath = "dataset/train_data"
valpath = "dataset/val_data"

traindata = characterLoader(trainpath)
valdata = characterLoader(valpath)
data_loader = Data.DataLoader(
    dataset=traindata,      # torch TensorDataset format
    batch_size=bs,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
val_loader = Data.DataLoader(
    dataset=valdata,
    batch_size=bs,
    num_workers=2
)

net = NewNet()
optimizer = optim.SGD(net.parameters(), lr=lr)  # optimize all cnn parameters
criterion = torch.nn.CrossEntropyLoss()

def train(e, data_loader):
    sum_loss = 0
    for step, (path, input, label) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = net.forward(input)
        # print("outputs ", outputs)
        # print("label ", label)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print('train epoch %d loss:%.03f'
          % (e, sum_loss))

def validate(val_loader):
    sum_loss = 0
    for step, (path, input, label) in enumerate(val_loader):
        outputs = net.forward(input)
        loss = criterion(outputs, label)
        sum_loss += loss.item()
    print('validation epoch %d loss:%.03f'
          % (e, sum_loss))
    return sum_loss


if __name__ == "__main__":
    val_loss = np.inf
    for e in range(epoch):
        train(e, data_loader)
        if e % 5 == 0:
            loss = validate(val_loader)
            if loss < val_loss:
                val_loss = loss
                torch.save(net.state_dict(), "model/NewNet_minLoss_model.pkl")
        if e == epoch - 1:
            torch.save(net.state_dict(), "./NewNet_epoch_{}_model.pkl".format(e))

