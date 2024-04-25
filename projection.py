# Standard libraries
import matplotlib.pyplot as plt
import torch
import tensorflow as tf


def fan_diagram(X):
    quantiles = torch.quantile(X, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    plt.figure(figsize=(16, 8))
    plt.plot(quantiles[0])
    plt.plot(quantiles[1])
    plt.plot(quantiles[2])
    plt.plot(X[0])


class Projection_Problem:

    def __init__(self, params):
        self.X = []
        self.Drift = []
        self.Diffusion = []
        self.input_output_pairs = []
        (
            self.N,
            self.T,
            self.d,
            self.lambda_,
            self.memory,
            self.sigma,
            self.mu,
            self.varsigma,
        ) = params[1]
        self.sigma = eval(self.sigma)
        self.mu = eval(self.mu)
        self.varsigma = eval(self.varsigma)

    def Algorithm3(self):
        X = []
        Drift = []
        Diffusion = []
        for t in range(self.T + 1):
            if t == 0:
                X.append(torch.zeros(self.N, self.d))  # X[-1]
                Drift.append(torch.zeros(self.N, self.d))
                Diffusion.append(torch.zeros(self.N, self.d, self.d))
                X.append(torch.zeros(self.N, self.d))  # X[0]
                Drift.append(torch.zeros(self.N, self.d))
                Diffusion.append(torch.zeros(self.N, self.d, self.d))
            else:
                Z = torch.randn(self.N, self.d)  # N x d
                B = torch.bernoulli(torch.tensor([0.5] * self.N))  # N
                drift_t = self.memory * self.mu(X[t - 2], t - 2) + (
                    1 - self.memory
                ) * self.mu(
                    X[t - 1], t - 1
                )  # N x d
                S = torch.einsum(
                    "n,dk->ndk", (B * self.lambda_), torch.eye(self.d)
                )  # N x d x d
                diffusion_t = S + torch.einsum(
                    "n,dk->ndk", self.varsigma(X[t - 1]), self.sigma
                )  # N x d x d
                x_t = (
                    X[t - 1] + drift_t + torch.einsum("nk,nkd->nd", Z, diffusion_t)
                )  # N x d
                X.append(x_t)
                Drift.append(drift_t)
                Diffusion.append(diffusion_t)
        X_NTd = torch.stack(X, dim=1)
        Drift_NTd = torch.stack(Drift, dim=1)
        Diffusion_NTd = torch.stack(Diffusion, dim=1)
        self.X = X_NTd[:, 0:, :]  # t = -1 to t = T
        self.Drift = Drift_NTd[:, 0:, :]  # t = -1 to t = T
        self.Diffusion = Diffusion_NTd[:, 0:, :]  # t = -1 to t = T
        return self.X

    def Algorithm1(self):
        X = self.X
        Drift = self.Drift
        sigma_2 = torch.matrix_power(self.sigma, 2)
        sigma_neg_2 = torch.matrix_power(torch.linalg.pinv(self.sigma), 2)
        Z = []
        for t in range(2, self.T + 2):
            x_t_2 = X[:, t - 2, :]
            x_t_1 = X[:, t - 1, :]
            x_t = X[:, t, :]
            drift_t = Drift[:, t, :]
            mu_x_t = x_t_1 + drift_t  # N x d
            inner_power_arg = self.lambda_ * torch.eye(self.d).reshape(
                (1, self.d, self.d)
            ).repeat(self.N, 1, 1) + torch.einsum(
                "n,dk->ndk", self.varsigma(x_t), self.sigma
            )  # N x d x d
            inner_power = torch.linalg.matrix_power(inner_power_arg, 2)

            outer_power_arg_torch = torch.einsum(
                "ndk,lk->ndk", inner_power, sigma_neg_2
            )
            outer_power_arg_tf = tf.convert_to_tensor(outer_power_arg_torch)
            outer_power = torch.from_numpy(
                tf.linalg.sqrtm(outer_power_arg_tf).numpy()
            )  # N x d x d

            outer_mul = torch.einsum("ndk,lk->ndk", outer_power, sigma_2)  # N x d x d
            covariance = torch.einsum(
                "n,ndk->ndk", self.varsigma(x_t), outer_mul
            )  # N x d x d

            Y = [mu_x_t, covariance]
            Z.append(
                [torch.cat((x_t_2, x_t_1, torch.full((self.N, 1), (t - 1))), dim=-1), Y]
            )
        self.input_output_pairs = Z
        return self.input_output_pairs


class OU_Projection_Problem(Projection_Problem):
    def __init__(self, data_index):
        PATH = "Data_Params/params_" + str(data_index) + ".pt"
        params = torch.load(PATH)
        self.data_index = data_index
        self.params = params
        super(OU_Projection_Problem, self).__init__(self.params)
        self.Auto_Generate()

    def Auto_Generate(self):
        self.Algorithm3()
        self.Algorithm1()
        PATH = "problem_instance_" + str(self.data_index) + ".pt"
        result = [
            self.N,
            self.T,
            self.d,
            self.X,
            self.Drift,
            self.Diffusion,
            self.input_output_pairs,
        ]
        torch.save(result, PATH)
