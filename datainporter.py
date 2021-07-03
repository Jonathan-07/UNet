from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from natsort import natsorted


ism_folder = 'ISM'
conf_folder = 'Conf'

class superdata(torch.utils.data.Dataset):
    def __init__(self, confocal_dir, ISM_dir, x_split, y_split, transform=None):
        self.confocal_dir = confocal_dir
        self.ISM_dir = ISM_dir
        self.transform = transform
        self.files = superdata.get_file_names(self)
        self.cut_sections = np.asarray([x_split, y_split])             #Try 6,3 initially

    def __len__(self):
        # return(len(self.files) * np.prod(self.cut_sections))
        return(len(self.files))

    def get_file_names(self):
        files = []
        for file in natsorted(os.listdir(self.confocal_dir)):
            if file.endswith(".tif"):
                files.append(file)

        return(files)

    def normalise_image(self, image):
        image = (image - np.min(image)) / (np.max(image)-np.min(image))
        return (image)

    # def get_index_position(self, index):
    #     index = index % np.prod(self.cut_sections)
    #     pos_y = index % self.cut_sections[0]
    #     pos_x = index // self.cut_sections[0]
    #     return np.int(pos_x), np.int(pos_y)

    def __getitem__(self, index):
        'Generates one sample of data'
        # This allows us to load the correct image for the section of that image
        # index_number = np.int(np.floor(index/np.prod(self.cut_sections)))
        index_number = index
        files_conf = superdata.get_file_names(self)
        file_name_c = self.confocal_dir + '/' + files_conf[index_number]
        file_name_i = self.ISM_dir + '/' + files_conf[index_number]



        image_c = io.imread(file_name_c)
        image_i = io.imread(file_name_i)

        image_c = self.normalise_image(image_c)
        image_i = self.normalise_image(image_i)

        # x_sizec, y_sizec = np.shape(image_c)
        # x_sizec = int(x_sizec/self.cut_sections[0])
        # y_sizec = int(y_sizec / self.cut_sections[1])
        #
        # x_sizei, y_sizei = np.shape(image_i)
        # x_sizei = int(x_sizei / self.cut_sections[0])
        # y_sizei = int(y_sizei / self.cut_sections[1])
        #
        # x, y = self.get_index_position(index)
        #
        # image_c = image_c[y * x_sizec: (y + 1) * x_sizec, x * y_sizec: (x + 1) * y_sizec]
        # image_i = image_i[y * x_sizei: (y + 1) * x_sizei, x * y_sizei: (x + 1) * y_sizei]
        #
        # if np.shape(image_i)[0] != 2 * np.shape(image_c)[0]:
        #     image_i = image_i[:-1, :]
        # if np.shape(image_i)[1] != 2 * np.shape(image_c)[1]:
        #     image_i = image_i[:, :-1]


        return (image_c,image_i)

# training_set = superdata(conf_folder, ism_folder, 6, 3)
# train_loader = torch.utils.data.DataLoader(training_set, batch_size=2)
#
#
# print(len(training_set))
# for i, batch in enumerate(train_loader):
#     dc, di = batch
#     print(dc.shape)
#     print(di.shape)
#     plt.imshow(di[0])
#     plt.show()
#     print(np.shape(dc),np.shape(di))
#     plt.subplot(1, 2, 1)
#     plt.imshow(dc[0, :, :])
#     plt.subplot(1, 2, 2)
#     plt.imshow(di[0, :, :])
#     plt.show()

#batch = next(iter(train_loader))


def accuracy(x, y, delta, need_array=False):
    zeros = torch.zeros_like(x)
    z = x - y
    z = torch.where(abs(z) < delta, zeros, z)
    non_zero_tuple = torch.nonzero(z, as_tuple=True)
    non_zero_count = non_zero_tuple[0].shape[0]
    elements = torch.numel(z)
    acc = 1 - non_zero_count/elements
    if need_array:
        z = z.squeeze().cpu().numpy()
    return acc, z
