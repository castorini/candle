import argparse
import io
import os
import random

from PIL import Image
from torch.autograd import Variable
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import candle

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

def read_idx(bytes):
    reader = io.BytesIO(bytes)
    reader.read(3)
    n_dims = int.from_bytes(reader.read(1), byteorder="big")
    sizes = []
    for _ in range(n_dims):
        sizes.append(int.from_bytes(reader.read(4), byteorder="big"))
    size = int(np.prod(sizes))
    buf = reader.read(size)
    return np.frombuffer(buf, dtype=np.uint8).reshape(sizes)

class SingleMnistDataset(data.Dataset):
    def __init__(self, images, labels, is_training):
        self.clean_images = []
        self.clean_labels = []
        for image, label in zip(images, labels):
            image = np.transpose(image)
            self.clean_images.append(Image.fromarray(image))
            self.clean_labels.append(int(label))
        self.is_training = is_training

    @classmethod
    def splits(cls, config, **kwargs):
        data_dir = config.dir
        img_files = [os.path.join(data_dir, "train-images-idx3-ubyte"),
            os.path.join(data_dir, "t10k-images-idx3-ubyte")]
        image_sets = []
        for image_set in img_files:
            with open(image_set, "rb") as f:
                content = f.read()
            arr = read_idx(content)
            image_sets.append(arr)

        lbl_files = [os.path.join(data_dir, "train-labels-idx1-ubyte"),
            os.path.join(data_dir, "t10k-labels-idx1-ubyte")]
        lbl_sets = []
        for lbl_set in lbl_files:
            with open(lbl_set, "rb") as f:
                content = f.read()
            lbl_sets.append(read_idx(content).astype(np.int))
        return cls(image_sets[0], lbl_sets[0], True, **kwargs), cls(image_sets[1], lbl_sets[1], False, **kwargs)

    def __getitem__(self, index):
        lbl = self.clean_labels[index]
        img = self.clean_images[index]
        if random.random() < 0.5 and self.is_training:
            arr = np.array(img)
            arr = np.roll(arr, random.randint(-2, 2), 0)
            arr = np.roll(arr, random.randint(-2, 2), 1)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            arr = arr.astype(np.float32)
            return torch.from_numpy(arr), lbl
        else:
            arr = np.array(img)
            arr = (arr - np.mean(arr)) / np.sqrt(np.var(arr) + 1E-6)
            return torch.from_numpy(arr.astype(np.float32)), lbl

    def __len__(self):
        return len(self.clean_images)

class DNNModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
        prune_cfg = candle.read_config()
        self.fc1 = candle.PruneLinear((784, 2048), prune_cfg)
        self.fc2 = candle.PruneLinear((2048, 1024), prune_cfg)
        self.fc3 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        return self.fc3(x)

class ConvModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
        prune_cfg = candle.read_config()
        self.conv1 = candle.PruneConv2d((1, 64, 5), prune_cfg)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = candle.PruneConv2d((64, 96, 5), prune_cfg)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = candle.PruneLinear((16 * 96, 1024), prune_cfg)
        self.fc2 = nn.Linear(1024, 10)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class TinyModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.use_cuda = True
        prune_cfg = candle.read_config()
        self.conv1 = candle.PruneConv2d((1, 17, 5), prune_cfg)#, provider=candle.WeightMaskGradientProvider)
        self.bn1 = nn.BatchNorm2d(17, affine=False)
        self.conv2 = candle.PruneConv2d((17, 10, 3), prune_cfg)#, provider=candle.WeightMaskGradientProvider)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.pool = nn.MaxPool2d(3)
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x.view(x.size(0), -1))

def train(args):
    optimizer = torch.optim.SGD(candle.list_params(model, train_prune=False), lr=0.01, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    train_set, test_set = SingleMnistDataset.splits(args)
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=min(32, len(test_set)))

    # 99.1 inverse grad, 99.4 weight track normal (both 85.461K params), 99.37 none (1729248M params)
    # 99.17 single prune after 4th epoch (85K params)
    # 99.1 single prune at 0th epoch (85K params)
    # scheduler = candle.ExponentialPruningScheduler(0.7, 2, end_idx=6, begin_idx=2) 
    # scheduler = candle.SinglePruningScheduler(95.057909565, begin_idx=0)

    # 15719 params, 99.22%, 0.9% of model
    # 97.96 for single prune at 2nd epoch
    # scheduler = candle.ExponentialPruningScheduler(0.9, 1.4, end_idx=10, begin_idx=2)
    # scheduler = candle.SinglePruningScheduler(99.090992153, begin_idx=2)

    # 1549 params, 98.12%, 0.09% of model, 20 epochs
    # 1552 params, single prune at 2nd epoch, 72%
    # scheduler = candle.ExponentialPruningScheduler(0.92, 1.45, end_idx=16, begin_idx=2)
    # scheduler = candle.SinglePruningScheduler(99.91042349, begin_idx=2)

    # 1616 params, hand-crafted tiny model, 20 epochs, 98.25%
    # 98.16%, 334 parameters with normal provider
    # 95.22%, 143 parameters with negative gradient, 15 epochs
    # 96.61%, 143 parameters with normal, 15 epochs
    # scheduler = candle.ExponentialPruningScheduler(0.94, 2, end_idx=8, begin_idx=2)
    scheduler = candle.ExponentialPruningScheduler(0.94, 3, end_idx=15, begin_idx=0)

    for n_epoch in range(args.n_epochs):
        print("Epoch: {}".format(n_epoch + 1))
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            model_in = Variable(model_in.cuda(), requires_grad=False)
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in)
            loss = criterion(scores, labels)
            loss.backward()
            candle.update_all(model)
            optimizer.step()
            if i % 16 == 0:
                n_unpruned = candle.count_params(model, type="unpruned")
                if n_unpruned >= 115:
                    candle.prune_all(model, percentage=scheduler.compute_rate())
                if n_unpruned <= 80:
                    candle.remove_activations(model)
                accuracy = (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum() / model_in.size(0)
                print("train accuracy: {:>10}, loss: {:>25}, unpruned: {:>10}".format(accuracy, loss.data[0], int(n_unpruned)))
        scheduler.step()

    model.eval()
    n = 0
    accuracy = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)
        scores = model(model_in)
        accuracy += (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum()
        n += model_in.size(0)
    print("test accuracy: {:>10}".format(accuracy / n))
    model.save(args.out_file)

def init_model(input_file=None, use_cuda=True):
    global model
    model = ConvModel()
    model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

model = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_in_file", type=str, default="")
    parser.add_argument("--dir", type=str, default="local_data")
    parser.add_argument("--in_file", type=str, default="")
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--n_epochs", type=int, default=10)
    args, _ = parser.parse_known_args()
    global model
    init_model(input_file=args.in_file)
    train(args)

if __name__ == "__main__":
    main()