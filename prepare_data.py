import os
import shutil

dirpath = "/Users/yuanyuan/Documents/GitHub/newdata_"
resultpath = "/Users/yuanyuan/Documents/GitHub/ABCrecognition/dataset/test_data"

for sub in os.listdir(dirpath): # sub is A, B, C, D, E
    if sub.startswith(".DS"):
        continue
    subpath = os.path.join(dirpath, sub)
    # count = 1
    for file in os.listdir(subpath):
        if sub.startswith(".DS"):
            continue
        shutil.copyfile(os.path.join(subpath, file), os.path.join(resultpath, file))
        # count += 1