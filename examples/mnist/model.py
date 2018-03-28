from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

import candle

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        def wrap_debug(layer, name):
            return layer
            return ctx.debug(layer, [candle.PrintHook], 
                type=candle.DebugType.BACKWARD_WEIGHTS.value, name=name)
        def make_block(in_units, out_units, dropout=0.5):
            return nn.Sequential(
                wrap_debug(ctx.wrap(nn.Linear(in_units, out_units), scale=scale, soft=use_soft),
                    f"lin.{in_units}.{out_units}"),
                ctx.bypass(nn.BatchNorm1d(out_units)),
                ctx.activation("tanh", scale=scale, soft=use_soft, limit=None),
                nn.Dropout(dropout))

        self.ctx = ctx = candle.StepQuantizeContext()
        self.scale = scale = nn.Parameter(torch.Tensor([0.5]))
        use_soft = not args.ste
        num_units = 4096
        bn = nn.BatchNorm1d(10)
        lin = nn.Linear(num_units, 10)
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            make_block(784, num_units),
            make_block(num_units, num_units),
            make_block(num_units, num_units),
            wrap_debug(ctx.wrap(lin, scale=scale, soft=use_soft), "final out"),
            ctx.bypass(bn))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = ctx = candle.StepQuantizeContext()
        self.scale = scale = nn.Parameter(torch.Tensor([0.5]))
        use_soft = not args.ste
        self.net = nn.Sequential(
            ctx.wrap(nn.Linear(784, 1024), scale=scale, soft=use_soft),
            candle.BinaryBatchNorm(1024),
            ctx.activation("tanh", scale=scale, soft=use_soft),
            nn.Dropout(0.4),
            ctx.wrap(nn.Linear(1024, 512), scale=scale, soft=use_soft),
            candle.BinaryBatchNorm(512),
            nn.Dropout(0.4),
            ctx.activation("tanh", scale=scale, soft=use_soft),
            ctx.wrap(nn.Linear(512, 10), scale=scale, soft=use_soft))

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
        use_soft = not args.ste
        
        self.conv1 = ctx.wrap(nn.Conv2d(1, 64, 5), scale=scale, soft=use_soft, limit=args.weight_limit)
        self.bn1 = ctx.bypass(nn.BatchNorm2d(64))
        self.a1 = ctx.activation("tanh", scale=scale, soft=use_soft, limit=args.activation_limit)

        self.conv2 = ctx.wrap(nn.Conv2d(64, 96, 5), scale=scale, soft=use_soft, limit=args.weight_limit)
        self.bn2 = ctx.bypass(nn.BatchNorm2d(96))
        self.a2 = ctx.activation("tanh", scale=scale, soft=use_soft, limit=args.activation_limit)

        self.dropout = nn.Dropout(0.6)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = ctx.wrap(nn.Linear(16 * 96, 1024), scale=scale, soft=use_soft, limit=args.weight_limit)
        self.bn4 = ctx.bypass(nn.BatchNorm1d(1024))
        self.a4 = ctx.activation("sigmoid", scale=scale, soft=use_soft, limit=args.activation_limit)
        self.fc2 = ctx.wrap(nn.Linear(1024, 10), scale=scale, soft=use_soft, limit=args.weight_limit)

    def forward(self, x):
        x = self.a1(self.pool(self.bn1(self.conv1(x))))
        x = self.a2(self.pool(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.a4(self.bn4(self.fc1(x))))
        return self.fc2(x)

class PTNetFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = scale = nn.Parameter(torch.Tensor([0.5]))
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.a1 = nn.Tanh()

        self.conv2 = nn.Conv2d(64, 96, 5)
        self.bn2 = nn.BatchNorm2d(96)
        self.a2 = nn.Tanh()

        self.dropout = nn.Dropout(0.6)
        self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(16 * 96, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.a4 = nn.Tanh()

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.a1(self.pool(self.bn1(self.conv1(x))))
        x = self.a2(self.pool(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.a4(self.bn4(self.fc1(x))))
        return self.fc2(x)

nets = dict(ptnet=PTNet, lenet=LeNet, lenet5=LeNet5, fcnet=FCNet)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1E-1, metavar='LR',
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
parser.add_argument('--scale-rate', type=float, default=1,
                    help='the scaling LR rate')
parser.add_argument('--sgd', action='store_true', default=False,
                    help='use SGD')
parser.add_argument('--weight-limit', type=float, default=2)
parser.add_argument('--activation-limit', type=float, default=None)
args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
net_cls = nets[args.net]
model = net_cls()
if args.cuda:
    model.cuda()

params = model.ctx.list_model_params()
# params = list(model.parameters())
if args.sgd:
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
else:
    optimizer = candle.SignSGD(params, lr=args.lr, momentum=0.9)
# optimizer = optim.Adam(params, lr=5E-4)
if args.restore:
    model.load_state_dict(torch.load(args.save), strict=False)

scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.3, mode="max")

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        data = data * 2 - 1
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if model.scale.cpu().data[0] < 50:
            model.scale.data.add_(args.scale_rate)
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
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        data = data * 2 - 1
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    scheduler.step(correct)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
