from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable

import candle

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = ctx = candle.StepQuantizeContext()
        self.scale = scale = nn.Parameter(torch.Tensor([0.5]))
        self.net = nn.Sequential(
            ctx.wrap(nn.Linear(784, 1024), scale=scale, soft=not args.ste),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            ctx.wrap(nn.Linear(1024, 512), scale=scale, soft=not args.ste),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            ctx.wrap(nn.Linear(512, 10), scale=scale, soft=not args.ste))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.net(x), dim=1)

# https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py
class LeNet5(nn.Module): # broken right now
    def __init__(self):
        super().__init__()
        self.ctx = ctx = candle.GroupPruneContext(stochastic=True)
        self.convnet = nn.Sequential(
            ctx.wrap(nn.Conv2d(1, 6, 5)),
            nn.Tanh(),
            nn.MaxPool2d(2),
            ctx.wrap(nn.Conv2d(6, 16, 5)),
            nn.Tanh(),
            nn.MaxPool2d(2))

        self.fc = nn.Sequential(
            ctx.bypass(nn.Linear(16 * 25, 120)),
            nn.Tanh(),
            ctx.bypass(nn.Linear(120, 84)),
            nn.Tanh(),
            ctx.bypass(nn.Linear(84, 10)))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convnet(x)
        x = x.view(-1, 16 * 25)
        return F.log_softmax(self.fc(x), dim=1)

class PTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = ctx = candle.StepQuantizeContext()
        self.scale = scale = nn.Parameter(torch.Tensor([0.5]))
        self.conv1 = ctx.wrap(nn.Conv2d(1, 10, kernel_size=5), scale=scale, soft=not args.ste)
        self.bn_c1 = nn.BatchNorm2d(10)
        self.conv2 = ctx.wrap(nn.Conv2d(10, 20, kernel_size=5), scale=scale, soft=not args.ste)
        self.bn_c2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = ctx.wrap(nn.Linear(320, 50), scale=scale, soft=not args.ste)
        self.bn_fc1 = nn.BatchNorm1d(50)
        self.fc2 = ctx.wrap(nn.Linear(50, 10), scale=scale, soft=not args.ste)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn_c1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn_c2(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = self.bn_fc1(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

nets = dict(ptnet=PTNet, lenet=LeNet, lenet5=LeNet5)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5E-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--net', type=str, default='ptnet', choices=list(nets.keys()),
                    help='what model to use')
parser.add_argument('--prune', action='store_true', default=False,
                    help='to prune or not to prune')
parser.add_argument('--l0-decay', type=float, default=1E-1,
                    help='the l0 decay to use')
parser.add_argument('--save', type=str, default='out.pt',
                    help='save path')
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--ste', action='store_true', default=False,
                    help='Use straight-through estimator')
args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
net_cls = nets[args.net]
model = net_cls()
if args.cuda:
    model.cuda()

params = model.ctx.list_params()
optimizer = optim.Adam(params, lr=args.lr)
if args.restore:
    model.load_state_dict(torch.load(args.save), strict=False)

def train(epoch):
    model.train()
    model.ctx.print_info()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # Optionally add quantize_loss here (doesn't do much)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScale: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], model.scale.data[0]))

    torch.save(model.state_dict(), args.save)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    model.ctx.print_info()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
