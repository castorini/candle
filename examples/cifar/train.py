import argparse
import os
import pickle
import random

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from . import resnet
import candle

class Dataset(data.Dataset):
    def __init__(self, examples, name, labels=None, training=False):
        self.name = name
        examples = (examples - np.mean(examples, 0)) / (np.std(examples, 0) + 1E-6)
        self.examples = []
        self.training = training
        for example in examples:
            self.examples.append(example.reshape(3, 32, 32))
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.training:
            if random.random() < 0.3:
                ex += np.random.normal(0, 0.05, size=(3, 32, 32))
            if random.random() < 0.15:
                ex = np.ascontiguousarray(np.flip(ex, 1))
            if random.random() < 0.15:
                ex = np.ascontiguousarray(np.flip(ex, 2))
        if self.labels is None:
            return ex, -1
        return ex, self.labels[idx]

    @classmethod
    def splits(cls, folder):
        with open(os.path.join(folder, "test_data"), "rb") as f:
            test_set = Dataset(pickle.load(f).astype(np.float32), "test")
        with open(os.path.join(folder, "train_data"), "rb") as f:
            images = pickle.load(f).astype(np.float32)
            labels = pickle.load(f)

            im_cutoff = int(0.9 * len(images))
            lbl_cutoff = int(0.9 * len(labels))
            train_set = Dataset(images[:im_cutoff], "train", labels[:lbl_cutoff], training=True)
            dev_set = Dataset(images[im_cutoff:], "dev", labels[lbl_cutoff:])
        return train_set, dev_set, test_set

def train(args):
    random.seed(0)
    optimizer = torch.optim.SGD(list(filter(lambda x: x.requires_grad, model.parameters())), nesterov=True,
        lr=args.lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    train_set, dev_set, test_set = Dataset.splits(args.dir)
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(64, len(test_set)))
    test_loader = data.DataLoader(test_set, batch_size=min(64, len(test_set)))
    best_dev = 0

    for n_epoch in range(args.n_epochs):
        print("Epoch: {}".format(n_epoch + 1))
        for i, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            model_in = Variable(model_in.cuda(), requires_grad=False).float()
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            if i % 8 == 0:
                # if n_epoch < args.n_epochs - 4:
                #     candle.prune_all(model, percentage=1.5 * 0.6**n_epoch)
                n_unpruned = candle.count_params(model, type="unpruned")
                accuracy = (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum() / model_in.size(0)
                print("{} train accuracy: {:>10}, loss: {:>25}, unpruned: {:>10}".format((n_epoch * len(train_set)) // 128 + i, 
                    accuracy, loss.data[0], int(n_unpruned)))

        tot_accuracy = 0
        tot_sum = 0
        for i, (model_in, labels) in enumerate(dev_loader):
            model_in = Variable(model_in.cuda(), requires_grad=False).float()
            labels = Variable(labels.cuda(), requires_grad=False)
            scores = model(model_in)
            loss = criterion(scores, labels)

            tot_accuracy += (torch.max(scores, 1)[1].view(model_in.size(0)).data == labels.data).sum()
            tot_sum += model_in.size(0)
        print("dev accuracy: {:>10}".format(tot_accuracy / tot_sum))
        if tot_accuracy > best_dev:
            best_dev = tot_accuracy
            print("Saving best model...")
            model.save(args.out_file)

    model.eval()
    results = []
    for model_in, labels in test_loader:
        model_in = Variable(model_in.cuda(), volatile=True).float()
        scores = model(model_in)
        out_arr = torch.max(scores, 1)[1].view(model_in.size(0)).cpu().numpy().tolist()
        results.extend(out_arr)
    results = ["{},{}".format(i, r) for i, r in enumerate(results)]
    print("\n".join(results))
    model.save(args.out_file)

def init_model(input_file=None, use_cuda=True):
    global model
    model = resnet.WideResNet()
    model.cuda()
    if input_file:
        model.load(input_file)
    model.eval()

model = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_in_file", type=str, default="")
    parser.add_argument("--dir", type=str, default="local_data")
    parser.add_argument("--lr", type=float, default=1E-1)
    parser.add_argument("--in_file", type=str, default="")
    parser.add_argument("--out_file", type=str, default="output.pt")
    parser.add_argument("--n_epochs", type=int, default=60)
    args, _ = parser.parse_known_args()
    global model
    init_model(input_file=args.in_file)
    train(args)

if __name__ == "__main__":
    main()