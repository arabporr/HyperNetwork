# Setup
import itertools
import torch
import numpy as np

import projection
import MLP_Handler
import HN_Handler
import Test_and_plot

seed = 42
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Sigma, Mu, and Varsigma values
sigmas = []
sigmas.append(torch.eye(4))
# sigma[1], sigma[2], ...


mus = []
long_term_mean = [0.1]
mean_reversion = [5]

for ltm in long_term_mean:
    for mr in mean_reversion:
        mu_func_type1 = lambda X, ind: (ltm - X) * mr
        mus.append(mu_func_type1)


volatility = [0.01]
varsigmas = []
for vol in volatility:
    varsigma_func_type1 = lambda X: torch.ones(X.shape[0]) * vol
    varsigmas.append(varsigma_func_type1)


# TEST CONFIGURATIONS
params_list_with_states = {
    "N": [400],
    "T": [100],
    "d": [4],
    "lambda": [0],
    "memory": [0],
    "sigma": sigmas,
    "mu": mus,
    "varsigma": varsigmas,
}


# Generating Data Instance
params_perm = list(params_list_with_states.values())
Permutations = list(itertools.product(*params_perm))
for data_index, params in enumerate(Permutations):
    problem = projection.OU_Projection_Problem([data_index, params])
    problem.Auto_Generate()
    MLP_Handler.Run(data_index)
    HN_Handler.Run(data_index)
    Test_and_plot.Run(data_index)
