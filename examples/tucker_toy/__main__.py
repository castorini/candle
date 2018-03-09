import argparse

import torch
import torch.autograd as ag
import torch.optim as optim
import torch.nn as nn

import candle

class TuckerFunction(candle.Function):
    def __init__(self, t):
        self.t = t

    def __call__(self, b):
        return (b - self.t)**2

def train(args):
    f = TuckerFunction(args.t)
    p = candle.BernoulliDistribution()
    p_i = candle.BernoulliDistribution()
    estimator = candle.RISEEstimator(f, p, p_i)
    theta = nn.Parameter(torch.Tensor([0.3]))
    pi = nn.Parameter(torch.Tensor([0.3]))
    optimizer = optim.Adam([theta, pi], args.lr)

    losses = []
    for step in range(args.steps):
        optimizer.zero_grad()
        theta_grad, pi_grad, loss = estimator.estimate_gradient(theta, pi)
        losses.append(loss)
        if len(losses) > 500:
            print((sum(losses) / len(losses)).data[0])
            print("Step #{}: theta={:.8} pi={:.8}".format(step, theta.data[0], pi.data[0]))
            losses = []
        
        pi.grad = pi_grad
        theta.grad = theta_grad
        optimizer.step()
        theta.data.clamp_(0, 1)
        pi.data.clamp_(0, 1)
        # print("Step #{}: theta={:.8} pi={:.8}".format(step, theta.data[0], pi[0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5E-3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--t", type=float, default=0.499)
    args, _ = parser.parse_known_args()
    train(args)

if __name__ == "__main__":
    main()