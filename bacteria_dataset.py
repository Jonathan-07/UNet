import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



class Bacteria(Dataset):

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        img = Image.open(file_name)
        self.number_image = img.n_frames
        self.image_width = img.size[0]
        self.image_height = img.size[1]
        img.close

    def loadback(self, index):
        img = Image.open(self.file_name)
        img.seek(index)
        phase = np.asarray(img)
        img.seek(index+1)
        mask = np.asarray(img)
        img.close()
        masker = torch.from_numpy(mask.astype(np.float32))
        return phase, mask

    def __len__(self):
        return(self.number_image/2)

    def __getitem__(self, index):
        phase, mask = self.loadback(index)

        return(phase, mask)

    def datainfo(self):
        print('There are ', self.number_image, 'images')
        print('The size', self.image_width, ' and ', self.image_height)

bacteria_data = Bacteria('/home/john/Data/test.tif')
phase, mask = bacteria_data.__getitem__(1)
print(phase)
print(mask)
# plt.imshow(phase)
# plt.show()
# plt.imshow(mask)
# plt.show()
