import os
import shutil

dirpath = "dataset/val_data"

for file in os.listdir(dirpath): # sub is A, B, C, D, E
    if file.startswith(".DS"):
        continue
    if file[0] == 'N':
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, 'E' + file[1::]))
