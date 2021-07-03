from torch.utils.data import DataLoader
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.nn import DataParallel

from UNet_full_network import *
from datainporter import *

ism_folder = 'ISM_Train'
conf_folder = 'Conf_Train'


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
        self.epoch_acc = 0
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
        self.epoch_acc = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss
        acc = self.epoch_acc/dataset_size
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Epoch time', epoch_duration, self.epoch_count)
        self.tb.add_scalar('Accuracy', acc, self.epoch_count)
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = acc
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # print(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_acc(self, acc):
        self.epoch_acc += acc


def correct(outputs, mask):
    outputs = outputs > 0.9
    hold = np.logical_and(outputs, mask)
    return (hold.sum().numpy())


dataset = superdata(conf_folder, ism_folder, 6, 3)
validation_split = 0
epochs = 5000

dataset_size = dataset.__len__()
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

params = OrderedDict(lr=[0.1], batch_size=[1], momentum=[0.99], criterion=[nn.MSELoss()])
m = RunManager()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for run in RunBuilder.get_runs(params):
    network = UNet(n_channels=1, n_classes=1)
    network = network.to(device)
    train_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=False, sampler=val_sampler)


    criterion = run.criterion

    optimiser = torch.optim.SGD(network.parameters(), lr=run.lr, momentum=run.momentum)

    m.begin_run(run, network, train_loader)
    for epoch in range(epochs):
        m.begin_epoch()
        for batch in train_loader:
            images, masks = batch
            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if network.n_classes == 1 else torch.long
            masks = masks.to(device=device, dtype=mask_type)
            images = images.unsqueeze(1)
            masks = masks.unsqueeze(1)

            pred = network(images)

            loss = criterion(pred, masks)
            acc = accuracy(pred,masks,0.05)[0]
            m.track_loss(loss)
            m.track_acc(acc)
            optimiser.zero_grad()
            loss.mean().backward()
            optimiser.step()

        # """ Save model when training has optimised it"""
        if epoch + 1 in [100,200,300,500,750,1000,2000,3000,4000,5000]:
            torch.save(network.state_dict(),
                       'UNet_lr{}_bs{}_mom{}_epoch{}.pth'.format(run.lr, run.batch_size, run.momentum, epoch))

        # """Code for validation"""
        # total_loss_val = 0
        # for val_batch in val_loader:
        #     val_images, val_masks = val_batch
        #     val_images = val_images.to(device=device, dtype=torch.float32)
        #     val_masks = val_masks.to(device=device, dtype=mask_type)
        #     val_images = val_images.unsqueeze(1)
        #     val_masks = val_masks.unsqueeze(1)
        #     val_pred = network(val_images)
        #     val_loss = criterion(val_pred, val_masks)
        #     total_loss_val += val_loss.item()
        # print("lr:", run.lr, " epoch:", epoch + 1, " avg loss:", total_loss_val / split)

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
