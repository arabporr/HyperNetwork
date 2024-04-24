# Setup
import os
import shutil
import itertools
import torch
import numpy as np


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


# VALUES LISTS


## SIGMA
sigmas = []
sigmas.append(torch.eye(10))


## MU
mus = []

long_term_mean = 0.01
mean_reversion = 0.5
mus.append(f"lambda X, ind: ({long_term_mean} - X) * {mean_reversion}")
mus.append("lambda X, ind: torch.ones(X.shape) * 0.01")
mus.append("lambda X, ind: torch.ones(X.shape) * 0.1")
mus.append("lambda X, ind: 0.1 * X + 0.01")
mus.append("lambda X, ind: 0.1 * X + torch.cos(0.01 * X)")
mus.append("lambda X, ind: torch.exp(-X) + torch.cos(0.01 * X)")

## VARSIGMA
varsigmas = []

volatility = [0.001, 0.005, 0.01, 0.05, 0.1]
for vol in volatility:
    varsigmas.append(f"lambda X: torch.ones(X.shape[0]) * {vol}")


# TEST CONFIGURATIONS
params_list_with_states = {
    "N": [1000],
    "T": [200],
    "d": [10, 2, 5, 10, 20],
    "lambda": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "memory": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "sigma": sigmas,
    "mu": mus,
    "varsigma": varsigmas,
}


# Generating Data Params
Directory = "Data_Params/"
if not os.path.exists(Directory):
    os.mkdir(Directory)
else:
    shutil.rmtree(Directory)
    os.mkdir(Directory)
params_perm = list(params_list_with_states.values())
Permutations = list(itertools.product(*params_perm))
Permutations_Count = len(Permutations)
for data_index, params in enumerate(Permutations):
    projection_input = [data_index, params]
    torch.save(projection_input, "Data_Params/params_" + str(data_index) + ".pt")

print(Permutations_Count, "permutations generated and saved in: Data_Params/")


file = open("Commands.sh", "w")
for i in range(Permutations_Count):
    file.write(f"python main.py --data_index {i}\n")
file.close()

print("Commands for running each problem generated and saved at: Commands.sh")
