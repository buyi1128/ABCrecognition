import sys
import os
import shutil
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim

from characterLoader import characterLoader
from nets.NewNet import NewNet

testpath = "dataset/test_data"
testdata = characterLoader(testpath)
modelfile = "model/NewNet_epoch_79_model.pkl"
net = NewNet()
net.load_state_dict(torch.load(modelfile))
total = 0
right = 0
wrongDir = "./wrong1"
if not os.path.exists(wrongDir):
    os.makedirs(wrongDir)
for step, (path, input, label) in enumerate(testdata):
    input = input.unsqueeze(0)
    outputs = net.forward(input)
    index = torch.argmax(outputs, dim=-1)
    total += 1
    if index[0] == label:
        right += 1
    else:
        newname = str(index[0])+"_"+str(total)+".bmp"
        shutil.copyfile(path, os.path.join(wrongDir, newname))
        # print("output", outputs, index)
        # print("label ", label)
print("total: ", total, "  right:", right, "  accuracy: {%.3f}" % (float(right/total)))