import argparse

import numpy as np
import torch
import torch.autograd as ag
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import candle

class TuckerFunction(candle.Function):
    def __init__(self, t):
        self.t = t

    def __call__(self, b):
        return (b - self.t)**2

def train_reinforce(args):
    f = TuckerFunction(args.t)
    theta = candle.Package([nn.Parameter(torch.Tensor([0]))])
    p = candle.SoftBernoulliDistribution(theta)
    estimator = candle.REINFORCEEstimator(f, p)
    optimizer = optim.Adam([theta.singleton], args.lr)

    losses = []
    estimates = []
    for step in range(args.steps):
        optimizer.zero_grad()
        theta_grad = estimator.estimate_gradient(theta)
        loss = f(p.draw())
        losses.append(loss.data[0].reify()[0])
        estimates.append(theta_grad.data[0].reify()[0])
        if len(losses) > 400:
            loss = sum(losses) / len(losses) * 100
            var = np.log(np.var(estimates))
            print(f"Step #{step}: theta={theta.data[0].reify()[0]:.4}, loss={loss:.4}, log var={var:.4}")
            losses = []
            estimates = []

        theta.singleton.grad = theta_grad.reify()[0]
        optimizer.step()

def train_rise(args):
    f = TuckerFunction(args.t)
    theta = candle.Package([nn.Parameter(torch.Tensor([0]))])
    pi = candle.Package([nn.Parameter(torch.Tensor([0]))])
    p = candle.SoftBernoulliDistribution(theta)
    p_i = candle.SoftBernoulliDistribution(pi)
    estimator = candle.RISEEstimator(f, p, p_i)
    optimizer = optim.Adam([theta.singleton, pi.singleton], args.lr)

    losses = []
    estimates = []
    for step in range(args.steps):
        optimizer.zero_grad()
        theta_grad, pi_grad = estimator.estimate_gradient(theta, pi)
        loss = f(p.draw()) * 100
        losses.append(loss.data[0].reify()[0])
        estimates.append(theta_grad.data[0].reify()[0])
        if len(losses) >= 400:
            loss = sum(losses) / len(losses)
            var = np.log(np.var(estimates))
            print(f"Step #{step}: theta={theta.data[0].reify()[0]:.4} pi={pi.data[0].reify()[0]:.4}, loss={loss:.4}, log var={var:.4}")
            losses = []
            estimates = []
        
        pi.singleton.grad = pi_grad.reify()[0]
        theta.singleton.grad = theta_grad.reify()[0]
        optimizer.step()

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 10)
        self.lin2 = nn.Linear(10, 1)
        self.lin1.bias.data.zero_()
        self.lin2.bias.data.zero_()
        nn.init.xavier_uniform(self.lin1.weight)
        nn.init.xavier_uniform(self.lin2.weight)

    def forward(self, theta, noise):
        theta = theta.reify()[0].sigmoid()
        noise = noise.reify()[0]
        x = 2 * torch.cat([theta, noise]) - 1
        x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return candle.Package([x])

def train_relax(args, use_rebar=False):
    f = TuckerFunction(args.t)
    theta = candle.Package([nn.Parameter(torch.Tensor([0]))])
    if use_rebar:
        phi = candle.Package([nn.Parameter(torch.Tensor([0.5]))])
        c = candle.RebarFunction(f, phi)
    else:
        c = SimpleNet()
        phi = candle.Package(list(c.parameters()))

    p = candle.SoftBernoulliDistribution(theta)
    z = candle.BernoulliRelaxation(theta)
    z_tilde = candle.ConditionedBernoulliRelaxation(theta)

    estimator = candle.RELAXEstimator(f, c, p, z, z_tilde, candle.Heaviside())
    optimizer = optim.Adam([theta.singleton] + phi.reify(flat=True), args.lr)

    losses = []
    estimates = []
    for step in range(args.steps):
        optimizer.zero_grad()
        theta_grad, phi_grad = estimator.estimate_gradient(theta, phi)
        loss = f(p.draw()) * 100
        losses.append(loss.data[0].reify()[0])
        estimates.append(theta_grad.data[0].reify()[0])
        if len(losses) >= 400:
            loss = sum(losses) / len(losses)
            var = np.log(np.var(estimates))
            print(f"Step #{step}: theta={theta.data[0].reify()[0]:.4} loss={loss:.4}, log var={var:.4}")
            losses = []
            estimates = []
        
        theta.singleton.grad = theta_grad.reify()[0].detach()
        phi.iter_fn(candle.apply_gradient, phi_grad)
        optimizer.step()

def train_rice(args):
    f = TuckerFunction(args.t)
    theta = candle.Package([nn.Parameter(torch.Tensor([0]))])
    pi = candle.Package([nn.Parameter(torch.Tensor([0]))])
    phi = candle.Package([nn.Parameter(torch.Tensor([0.5]))])
    c = RebarFunction(f, phi)

    p = candle.SoftBernoulliDistribution(theta)
    p_i = candle.SoftBernoulliDistribution(pi)
    z = candle.BernoulliRelaxation(theta)
    z_tilde = candle.ConditionedBernoulliRelaxation(theta)

    estimator = candle.RICEEstimator(f, c, p, p_i, z, z_tilde, candle.Heaviside())
    optimizer = optim.Adam([theta.singleton, pi.singleton, phi.singleton], args.lr)

    losses = []
    estimates = []
    for step in range(args.steps):
        optimizer.zero_grad()
        theta_grad, pi_grad, phi_grad = estimator.estimate_gradient(theta, pi, phi)
        loss = f(p.draw()) * 100
        losses.append(loss.data[0].reify()[0])
        estimates.append(theta_grad.data[0].reify()[0])
        if len(losses) >= 400:
            loss = sum(losses) / len(losses)
            var = np.log(np.var(estimates))
            print(f"Step #{step}: theta={theta.data[0].reify()[0]:.4} loss={loss:.4}, log var={var:.4}")
            losses = []
            estimates = []
        
        theta.singleton.grad = theta_grad.reify()[0].detach()
        pi.singleton.grad = pi_grad.reify()[0].detach()
        phi.iter_fn(candle.apply_gradient, phi_grad)
        optimizer.step()

def main():
    def train(name):
        print("=" * 20)
        print(f"{name.upper()}")
        print("=" * 20)
        if name == "reinforce":
            train_reinforce(args)
        elif name == "relax":
            train_relax(args)
        elif name == "rebar":
            train_relax(args, use_rebar=True)
        elif name == "rise":
            train_rise(args)
        elif name == "rice":
            train_rice(args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--t", type=float, default=0.499)
    parser.add_argument("--type", type=str, default="relax")
    args, _ = parser.parse_known_args()
    train(args.type)

if __name__ == "__main__":
    main()