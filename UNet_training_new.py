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
import time
from os.path import splitext
from os import listdir
from os import path
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage import filters
from skimage import morphology
from torch.utils.data.sampler import SubsetRandomSampler

from UNet_full_network import *


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

        #df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

class Bacteria(Dataset):

    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        img = Image.open(file_name)
        self.number_image = img.n_frames
        self.image_width = img.size[0]
        self.image_height = img.size[1]
        img.close

    def loadback(self, index):
        index = index * 2
        img = Image.open(self.file_name)
        img.seek(index)
        phase = np.asarray(img)
        img.seek(index+1)
        mask = np.asarray(img)
        img.close()
        mask = self.process(mask)
        phase = torch.from_numpy(phase.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return phase, mask

    def process(self, mask):
        mask = filters.sobel(mask)
        mask = morphology.erosion(mask)
        mask = morphology.dilation(mask)
        mask = morphology.erosion(mask)
        thresh = threshold_otsu(mask)
        binary = mask > thresh
        return binary

    def __len__(self):
        length_data = np.int(self.number_image/2)
        return(length_data)

    def __getitem__(self, index):
        phase, mask = self.loadback(index)

        return(phase, mask)

    def datainfo(self):
        print('There are ', self.number_image, 'images')
        print('The size', self.image_width, ' and ', self.image_height)

def correct(outputs, mask):
    outputs = outputs > 0.9
    hold = np.logical_and(outputs, mask)
    return (hold.sum().numpy())


dataset = Bacteria('/home/john/Data/test.tif')
validation_split = .2

dataset_size = int(dataset.number_image/2)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

params = OrderedDict(lr=[.01], batch_size=[1], momentum=[0.99])
m = RunManager()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for run in RunBuilder.get_runs(params):
    network = UNet(n_channels=1, n_classes=1)
    network = network.to(device)
    train_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=False, sampler=val_sampler)
    
    if network.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.SGD(network.parameters(), lr=run.lr, momentum=run.momentum)

    m.begin_run(run, network, train_loader)
    for epoch in range(1):
        m.begin_epoch()
        i=0
        for batch in train_loader:
            images, masks = batch
            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if network.n_classes == 1 else torch.long
            masks = masks.to(device=device, dtype=mask_type)
            outputs = network(images)
            loss = criterion(outputs, masks)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            m.track_loss(loss)
            print(loss.item())
            temp = outputs.cpu().detach().numpy()
            temp = np.squeeze(np.squeeze(temp))
            # print(correct(temp, np.squeeze(np.squeeze(masks.cpu()))))
            i+=1
            # print(i)
        print("EPOCH:   ", epoch)
        m.end_epoch()
    
    total_loss_test = 0
    for batch in val_loader:
        images, masks = batch
        images = images.to(device=device,dtype=torch.float32)
        masks = masks.to(device=device, dtype=mask_type)
        outputs = network(images)
        loss = criterion(outputs,masks)
        total_loss_test += loss.item()
        print("avg loss:", total_loss_test/split, " epoch:", epoch)
        
    m.end_run()
    

      

# mask_new = masks.to(device='cpu')
# mask_new = mask_new.squeeze()
# mask_new = mask_new.squeeze()
# mask_new = mask_new.detach().numpy()
# np.save('mask', mask_new)
# temp = outputs.cpu()
# temp = temp.detach().numpy()
# np.save('output', temp)

# torch.save(network, 'trained_bacteria_UNet.pt')
