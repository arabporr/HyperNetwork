## Standard libraries
import time
import json
import math
import numpy as np
import scipy
from scipy.linalg import fractional_matrix_power as frac_mat_pow

## Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

def setup_gpu(seed=42):
    # Function for setting the seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Fetching the device that will be used throughout this notebook
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device -> ", device)

def fan_diagram(X):
    quantiles = torch.quantile(X, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    fig = plt.figure(figsize=(16,8))
    plt.plot(quantiles[0])
    plt.plot(quantiles[1])
    plt.plot(quantiles[2])
    plt.plot(X[0])

class Projection_Problem:
    def __init__(self):
        self.N = 1000
        self.T = 100
        self.d = 1
        self.Sigma = torch.eye(self.d)
        self.Lambda = 0
        self.memory = 0
        self.X = [] 
        self.Drift = []
        self.Diffusion = []
        self.input_output_pairs = []
    
    def mu(self, X):
        raise NotImplementedError()

    def varsigma(self, X):
        raise NotImplementedError()

    def Algorithm3(self):
        N = self.N
        T = self.T 
        d = self.d
        Mu = self.mu
        Varsigma = self.varsigma 
        Sigma = self.Sigma 
        Lambda = self.Lambda 
        memory = self.memory
        X = [] 
        Drift = []
        Diffusion = []

        for t in range(T+1):
            if t == 0:
                X.append(torch.zeros(N, d)) # X[-1]
                Drift.append(torch.zeros(N, d))
                Diffusion.append(torch.zeros(N, d))
                X.append(torch.zeros(N, d)) # X[0]
                Drift.append(torch.zeros(N, d)) 
                Diffusion.append(torch.zeros(N, d))
            else:
                Z = torch.randn(N, d)
                B = torch.bernoulli(torch.tensor([0.5]*N))
                drift_t = memory * Mu(X[t-2]) + (1-memory) * Mu(X[t-1])
                S = torch.einsum("n,dk->nk", (B * Lambda), torch.eye(d))
                diffusion_t = S + torch.einsum("nd,dk->nk", Varsigma(X[t-1]), Sigma)
                x_t = X[t-1] + drift_t + (diffusion_t * Z)
                X.append(x_t)
                Drift.append(drift_t)
                Diffusion.append(diffusion_t)
        
        X_NTd = torch.stack(X, dim=1)
        Drift_NTd = torch.stack(Drift, dim=1)
        Diffusion_NTd = torch.stack(Diffusion, dim=1)

        self.X = X_NTd[:, 2:, :]
        self.Drift = Drift_NTd[:, 2:, :]
        self.Diffusion = Diffusion_NTd[:, 2:, :]
        
        return X
    
    def Algorithm1(self):
        N = self.N
        T = self.T 
        d = self.d
        Mu = self.mu
        Varsigma = self.varsigma 
        Sigma = self.Sigma 
        Lambda = self.Lambda 
        X = self.X 
        Drift = self.Drift
        sigma_2 = torch.matrix_power(Sigma,2)
        sigma_neg_2 = torch.matrix_power(torch.linalg.pinv(Sigma), 2)
        Y = torch.Tensor.new_empty(N, T, d*(d+1)/2)

        for t in range(T-1):
            x_t = X[:, t+1, :] # the X was indexed from -1
            drift_t = Drift[:, t+1, :] # Drift_n_t (indexes for drift are one ahead)
            mu_x_t =  x_t + drift_t

            iner_power_arg = Lambda * torch.einsum("a,ij->aij", Varsigma(x_t), Sigma)
            iner_power = torch.matrix_power(iner_power_arg, 2)

            outer_power_arg = torch.einsum("ij,ajk->aik", sigma_neg_2, iner_power)
            outer_power = outer_power_arg**(1/2)

            sigma_x_t = torch.einsum("a,ij,ajk->aik", Varsigma(x_t), sigma_2, outer_power)
            
        self.input_output_pairs = sigma_x_t
        return self.input_output_pairs

class OU_Projection_Problem(Projection_Problem):
    def __init__(self):
        super().__init__()
        self.long_term_mean = 0.01
        self.mean_reversion = 0.5
        self.volitality = 0.001
    
    def mu(self, X):
        return (self.long_term_mean - X) * self.mean_reversion

    def varsigma(self, X):
        return torch.ones(X.shape[0]) * self.volitality
  

