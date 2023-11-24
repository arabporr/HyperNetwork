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
    print("Using device", device)

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
        self.Sigma = torch.eye(1)
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

    def simulate_x(self):
        N = self.N
        T = self.T 
        d = self.d 
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
                drift_t = memory * self.mu(X[t-2]) + (1-memory) * self.mu(X[t-1])     # Drift[t] but in place [t+1]
                eye_reshaped = torch.eye(d).reshape([1,d,d])
                B_reshaped = B.reshape([N, 1])
                diffusion_t = torch.einsum("n,ik,nk->ni", self.varsigma(X[t-1]), Sigma, Z) + ((B_reshaped * Lambda) * Z) # Diffusion[t] but in place [t+1]
                x_t1 = X[t-1] + drift_t + diffusion_t  # X[t+1]
                X.append(x_t1)
                Drift.append(drift_t)
                Diffusion.append(diffusion_t)
        
        X_all = torch.stack(X, dim=1)
        Drift_all = torch.stack(Drift, dim=1)
        Diffusion_all = torch.stack(Diffusion, dim=1)

        X = X_all[:, 2:, :]
        Drift = Drift_all[:, 2:, :]
        Diffusion = Diffusion_all[:, 2:, :]

        self.X = X
        self.Drift = Drift
        self.Diffusion = Diffusion

        return X
    
    def Algorithm1(self):
        N = self.N
        T = self.T 
        d = self.d 
        Sigma = self.Sigma 
        Lambda = self.Lambda 
        #memory = self.memory
        X = self.X 
        Drift = self.Drift
        #Diffusion = self.Diffusion
        sigma_2 = torch.matrix_power(Sigma,2)
        sigma_neg_2 = torch.matrix_power(torch.linalg.pinv(Sigma), 2)
        Z_N = []
        Y = torch.Tensor.new_empty(N, T, d*(d+1)/2)

        # for n in range(N):
        #     x_n = X[n]
        #     drift_n = Drift[n]
        #     Y_X = []
        for t in range(T-1):
            x_t = X[:, t+1, :] # the X was indexed from -1
            drift_t = Drift[:, t+1, :] # Drift_n_t (indexes for drift are one ahead)
            mu_x_t =  x_t + drift_t

            # iner_power_arg = torch.add(
            #         torch.mul(Lambda, torch.eye(d)),
            #         torch.mul(torch.squeeze(self.varsigma(x_t), 2), Sigma))
            iner_power_arg = Lambda * torch.einsum("a,ij->aij", self.varsigma(x_t), Sigma)
            iner_power = torch.matrix_power(iner_power_arg, 2)


            outer_power_arg = torch.einsum("ij,ajk->aik", sigma_neg_2, iner_power)
            #outer_power = torch.from_numpy( frac_mat_pow(outer_power_arg, 1/2)).type(torch.FloatTensor) 
            outer_power = outer_power_arg**(1/2)
            sigma_x_t = torch.einsum("a,ij,ajk->aik", self.varsigma(x_t), sigma_2, outer_power)
            
            # Y_X.append([mu_x_t, sigma_x_t]) #
            # Z = [x_n, Y_X]
            # Z_N.append(Z)
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
  

