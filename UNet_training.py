from torch.utils.data import DataLoader
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from UNet_full_network import *
from dataset import *

dir_img = '/home/john/Carvana_data/train'
dir_mask = '/home/john/Carvana_data/train_masks'


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
        self.tb = SummaryWriter(comment=f'-{run}')

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Epoch time', epoch_duration, self.epoch_count)
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # print(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size


def correct(outputs, mask):
    outputs = outputs > 0.9
    hold = np.logical_and(outputs, mask)
    return (hold.sum().numpy())


dataset = CarvanaDataset(dir_img, dir_mask, scale=0.5)
validation_split = .9
epochs = 10

dataset_size = dataset.__len__()
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

params = OrderedDict(lr=[.01], batch_size=[1,2], momentum=[0.99], num_workers=[0,1,2,4,8,16])
m = RunManager()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for run in RunBuilder.get_runs(params):
    network = UNet(n_channels=3, n_classes=1)
    network = network.to(device)
    train_loader = DataLoader(dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=False, sampler=val_sampler)

    if network.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.SGD(network.parameters(), lr=run.lr, momentum=run.momentum)

    m.begin_run(run, network, train_loader)
    for epoch in range(epochs):
        m.begin_epoch()
        print(epoch)
        i=0
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if network.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            masks_pred = network(images)
            loss = criterion(masks_pred, true_masks)
            m.track_loss(loss)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            i += 1
            print(i)
            # temp = outputs.cpu().detach().numpy()
            # temp = np.squeeze(np.squeeze(temp))
            # print(correct(temp, np.squeeze(np.squeeze(masks.cpu()))))
        #         print("EPOCH:   ", epoch)


        #
        # total_loss_test = 0
        # for batch in val_loader:
        #     images, masks = batch
        #     images = images.to(device=device, dtype=torch.float32)
        #     masks = masks.to(device=device, dtype=mask_type)
        #     outputs = network(images)
        #     loss = criterion(outputs, masks)
        #     total_loss_test += loss.item()
        # print("lr:", run.lr, " epoch:", epoch + 1, " avg loss:", total_loss_test / split)

        m.end_epoch()
    m.end_run()

# mask_new = masks.to(device='cpu')
# mask_new = mask_new.squeeze()
# mask_new = mask_new.squeeze()
# mask_new = mask_new.detach().numpy()
# np.save('mask', mask_new)
# temp = outputs.cpu()
# temp = temp.detach().numpy()
# np.save('output', temp)

# torch.save(network, 'trained_bacteria_UNet.pth')
