import os
import shutil
from scipy import misc
import numpy as np
import pickle

img_data_file = 'img_data.pkl'
data = dict()

file_path = 'data/train'
file_list = os.listdir(file_path)
file_list.sort()
for image_path in file_list:
    if image_path.endswith('.jpg'):
        img = misc.imread(os.path.join(file_path, image_path), flatten=True, mode='L')
        # img = misc.imread(os.path.join(file_path, image_path))
        new_img = misc.imresize(img, (128, 128))
        imgg = new_img.reshape(128, 128, 1)
        data[image_path[:-4]] = imgg/255.0


file_path = 'data/test'
file_list = os.listdir(file_path)
file_list.sort()
for image_path in file_list:
    if image_path.endswith('.jpg'):
        img = misc.imread(os.path.join(file_path, image_path), flatten=True, mode='L')
        # img = misc.imread(os.path.join(file_path, image_path))
        new_img = misc.imresize(img, (128, 128))
        imgg = new_img.reshape(128, 128, 1)
        data[image_path[:-4]] = imgg/255.0

with open(img_data_file, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# if img.ndim == 3:
#     w, h, d = img.shape
#     img_grey = np.empty((w, h, 1), dtype=np.uint8)
#     img_grey[:, :, 0] = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2]) / 3.0
