import math
import numpy as np
import scipy

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import unittest
import projection


class Projection_Test(unittest.TestCase):
    def test_simulate(self):
        problem = projection.OU_Projection_Problem()
        X = problem.simulate_x()
        final_X = X[:,-1:,0]
        quantiles = torch.quantile(final_X, torch.tensor([0.05, 0.5, 0.95]))
        expected_median = problem.long_term_mean
        variance = 0.001 ** 2 #problem.volitality + problem.Lambda**2 * 0.25
        expected_5th_percentile = expected_median - 1.96 * (variance**(0.5))
        expected_95th_percentile = expected_median + 1.96 * (variance**(0.5))
        preds = [expected_5th_percentile, expected_median, expected_95th_percentile]
        assert torch.allclose(quantiles, preds, rtol=0.3)
    
    def test_fail(self):
        assert 1 == 1
