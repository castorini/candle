import argparse

import torch

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

    theta = torch.Tensor([0.5])
    pi = torch.Tensor([0.2])
    for step in range(args.steps):
        theta_grad, pi_grad = estimator.estimate_gradient(theta, pi)
        
        # pi -= args.lr * pi_grad
        theta -= args.lr * theta_grad
        theta.clamp_(0, 1)
        pi.clamp_(0, 1)
        print("Step #{}: theta={:.8} pi={:.8}".format(step, theta[0], pi[0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1E-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--t", type=float, default=0.19)
    args, _ = parser.parse_known_args()
    train(args)

if __name__ == "__main__":
    main()