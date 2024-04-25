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
sigmas.append("torch.eye(self.d)")


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
    "d": [2, 5, 10, 20],
    "lambda": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "memory": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "sigma": sigmas,
    "mu": mus,
    "varsigma": varsigmas,
}

default_values = {
    "N": [1000],
    "T": [200],
    "d": [10],
    "lambda": [0.1],
    "memory": [0.5],
    "sigma": ["torch.eye(self.d)"],
    "mu": ["lambda X, ind: (0.01 - X) * 0.5"],
    "varsigma": ["lambda X: torch.ones(X.shape[0]) * 0.001"],
}

# Generating Data Params
Directory = "Data_Params/"
if not os.path.exists(Directory):
    os.mkdir(Directory)
else:
    shutil.rmtree(Directory)
    os.mkdir(Directory)


# De we need all permutation (Grid search for all the param simultaneity or one by one)
all_permutations = False
if all_permutations:
    params_perm = list(params_list_with_states.values())
    Permutations = list(itertools.product(*params_perm))
else:
    Permutations = set()
    key_index = 0
    def_params = list(list(itertools.product(*list(default_values.values())))[0])
    for key, values in params_list_with_states.items():
        for replacement in values:
            new_params = def_params.copy()
            new_params[key_index] = replacement
            Permutations.add(tuple(new_params))
        key_index += 1
    Permutations = list(Permutations)

Permutations_Count = len(Permutations)

file = open("permutations_list.txt", "w")
for data_index, params in enumerate(Permutations):
    projection_input = [data_index, params]
    file.write(str(projection_input) + "\n")
    torch.save(projection_input, "Data_Params/params_" + str(data_index) + ".pt")
file.close()
print(Permutations_Count, "permutations generated and saved in: Data_Params/")


file = open("Commands.sh", "w")
for i in range(Permutations_Count):
    file.write(f"python3 main.py --data_index {i}\n")
file.close()

print("Commands for running each problem generated and saved at: Commands.sh")
