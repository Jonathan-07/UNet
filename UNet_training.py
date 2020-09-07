from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import pandas as pd
import time
from os.path import splitext
from os import listdir
from os import path
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import natsort

from UNet_full_network import *

# Test model shapes
# t = torch.ones([1, 1, 572, 572])
# network = UNet(n_channels=1, n_classes=2)
# pred = network(t)
# print(pred.shape)


class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager():

    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        print(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size


class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, desired_height_width):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.desired_height_width = desired_height_width
        all_imgs = listdir(img_dir)
        all_masks = listdir(mask_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.total_masks = natsort.natsorted(all_masks)

    @classmethod
    def preprocess(cls, pil_img, desired_height_width):
        newW = desired_height_width
        newH = desired_height_width
        pil_img = pil_img.resize((newW, newH))
        return pil_img

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        #image = image.convert("RGB")
        image = self.preprocess(image, self.desired_height_width)
        image = ToTensor()(image)

        mask_loc = path.join(self.mask_dir, self.total_masks[idx])
        mask = Image.open(mask_loc)
        # mask = mask.convert("RGB")
        mask = self.preprocess(mask, self.desired_height_width)
        mask = ToTensor()(mask)

        return {
            'image': image,
            'mask': mask
        }

    
class CustomDataset_from_arr(Dataset):
    def __init__(self, img_arr, mask_arr, desired_width, desired_height):
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.total_imgs = img_arr
        self.total_masks = mask_arr

    @classmethod
    def preprocess(cls, pil_img, desired_height_width):
        newW = desired_width
        newH = desired_height
        pil_img = pil_img.resize((newW, newH))
        return pil_img

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image = self.total_imgs[idx]
        #image = image.convert("RGB")
        image = self.preprocess(image, self.desired_height_width)
        image = ToTensor()(image)

        mask = self.total_masks[idx]
        # mask = mask.convert("RGB")
        mask = self.preprocess(mask, self.desired_height_width)
        mask = ToTensor()(mask)

        return {
            'image': image,
            'mask': mask
        }
    


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

phases = np.empty(190, dtype=object)
masks = np.empty(190, dtype=object)
# 190 is the number of images in this case

for i in range(189):
    phases[i] = bacteria_data.__getitem__(i)[0]
    masks[i] = bacteria_data.__getitem__(i)[1]
    
    
dir_img = '/home/john/input/train/images'
dir_mask = '/home/john/input/train/masks'
train_dataset = CustomDataset_from_arr(phases, masks)


params = OrderedDict(
    lr = [.01]
    ,batch_size = [1]
    ,momentum = [0.99]
)
m = RunManager()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for run in RunBuilder.get_runs(params):

    network = UNet(n_channels=1, n_classes=1)
    network = network.to(device)
    train_loader = DataLoader(train_dataset, batch_size=run.batch_size, shuffle=False)
    loader = train_loader
    if network.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(network.parameters(), lr = run.lr, momentum = run.momentum)

    m.begin_run(run, network, loader)
    for epoch in range(1):
        m.begin_epoch()

        i=0
        for batch in loader:
            images = batch['image']
            masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if network.n_classes == 1 else torch.long
            masks = masks.to(device=device, dtype=mask_type)

            outputs = network(images)
            images_shape = images.shape
            outputs = F.interpolate(outputs, size=(images_shape[2],images_shape[3]))
            outputs = torch.squeeze(outputs,1)
            masks = torch.squeeze(masks,1)

            # plt.imshow(outputs.cpu().detach().squeeze())
            # plt.imshow(masks.cpu().detach().squeeze())

            loss = criterion(outputs, masks)
            print(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            m.track_loss(loss)

            i += 1
            print(i)
            print(loss.item())

        print("EPOCH:   ", epoch)
        m.end_epoch()
    m.end_run()
