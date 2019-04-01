import numpy as np
import torch
import os
import cv2
import pickle
import random

import torch.utils.data as Data

import param

class characterLoader(Data.Dataset):
    def __init__(self, path):
        self.path = path
        # self.num = 700
        self.data, self.labels, self.imgpaths = self.getTrainData(self.path)


    def getTrainData(self, imgdir):
        datalist = []
        labels = []
        imgpaths = []
        for name in os.listdir(imgdir):
            if name.startswith(".DS"):
                continue

            imgfile = os.path.join(imgdir, name)
            img = cv2.imread(imgfile)
            # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)
            # print(img.shape)
            datalist.append(img)
            imgpaths.append(imgfile)
            y = int(ord(name[0]) - ord("A"))
            y = y - 1 if y == 4 else y
            labels.append(y)
        if param.isDebug:
            datalist = datalist[:20]
            labels = labels[:20]
        print(imgdir, " images: ", len(datalist))
        return datalist, labels, imgpaths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = torch.tensor(np.array(self.data[index], dtype=np.float32))
        img = torch.unsqueeze(img, 0)
        label = torch.tensor(np.array(self.labels[index]))
        return self.imgpaths[index], img, label

# trainpath = "dataset/train_data"
# dataset = characterLoader(trainpath)
# data_loader = torch.utils.data.DataLoader(
#     dataset=dataset,      # torch TensorDataset format
#     batch_size=6,      # mini batch size
#     shuffle=False,               # 要不要打乱数据 (打乱比较好)
# )
# for x, y in data_loader:
#     print(x.size(), y)