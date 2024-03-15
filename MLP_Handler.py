import os
import shutil

import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

global device
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Device in MLP_Handler file:", device)


class MLP_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.generate_dataset(X, Y)

    def generate_dataset(self, X, Y):
        self.data = X

        self.size = len(self.data)

        labels = []
        for index in range(self.size):
            label = []
            label.extend(Y[0][index])

            matrix_upper_values = []
            for row in range(len(Y[1][index])):
                matrix_upper_values.extend(Y[1][index][row][row:])

            label.extend(matrix_upper_values)
            label = np.array(label)
            labels.append(label)

        self.label = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data_point = self.data[index]
        data_label = self.label[index]
        return data_point, data_label


def Dataset_Generator():
    MLPs_datasets = []
    for index in range(T):
        dataset_index = MLP_Dataset(
            input_output_pairs[index][0], input_output_pairs[index][1]
        )
        MLPs_datasets.append(dataset_index)

    PATH = Directory + "MLPs_Datasets.pt"
    torch.save(MLPs_datasets, PATH)
    return MLPs_datasets


class Exp_layer(nn.Module):
    def __init__(self):
        super(Exp_layer, self).__init__()

    def forward(self, input_data):
        if torch.sum(input_data.isnan()) > 0:
            print("input_data is nan!")
            raise Exception("input_data is nan in exp layer!")
        inp = input_data[0]
        d_ = int((-3 + np.sqrt(9 + 8 * (inp.shape[0]))) / 2)
        inp = inp[d_:]
        inp = inp.to(device)
        temp = torch.zeros(d_, d_)
        temp = temp.to(device)

        indices = torch.triu_indices(d_, d_)
        indices.to(device)

        temp[indices[0], indices[1]] = inp
        temp[indices[1], indices[0]] = inp
        matrix = torch.linalg.matrix_exp(temp)

        result = input_data
        result[0][d_:] = matrix[indices[0], indices[1]]
        if torch.sum(result.isnan()) > 0:
            print("result is nan!")
            raise Exception("result is nan in exp layer!")
        return result


class MainNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        layers = []

        layers += [nn.Linear(in_features=(2 * d + 1), out_features=256)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(in_features=256, out_features=128)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(in_features=128, out_features=128)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(in_features=128, out_features=((d) + (d * (d + 1) // 2)))]

        layers += [Exp_layer()]
        self.layers = nn.Sequential(*layers)

    def forward(self, input_data):
        if torch.sum(input_data.isnan()) > 0:
            print("input_data is nan!")
            raise Exception("input_data is nan in first layer!")
        result = self.layers(input_data)
        return result


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        d_ = int((-3 + np.sqrt(9 + 8 * (targets[0].shape[0]))) / 2)
        preds = predictions[0]
        mu_pred = preds[:d_]
        cov_pred = preds[d_:]
        cov_pred = cov_pred.to(device)
        temp = torch.zeros(d_, d_)
        temp = temp.to(device)
        indices = torch.triu_indices(d_, d_)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_pred
        temp[indices[1], indices[0]] = cov_pred
        cov_pred = temp

        trgt = targets[0]
        mu_trgt = trgt[:d_]
        cov_trgt = trgt[d_:]
        cov_trgt = cov_trgt.to(device)
        temp = torch.zeros(d_, d_)
        temp = temp.to(device)
        indices = torch.triu_indices(d_, d_)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_trgt
        temp[indices[1], indices[0]] = cov_trgt
        cov_trgt = temp

        loss = 0.0

        # Mean Component
        loss += torch.mean((mu_pred - mu_trgt) ** 2)

        # Covariant Component
        A = torch.matmul(torch.linalg.pinv(cov_pred), cov_trgt)
        A_eigen = torch.linalg.eig(A)
        A_eigen = A_eigen.eigenvalues
        A_eigen = torch.abs(torch.real(A_eigen))
        A_eigen = torch.mean(torch.log(A_eigen) ** 2)
        loss += A_eigen / 2
        if torch.sum(loss.isnan()) > 0:
            print("loss is nan!")
            raise Exception("Loss is nan!")
        return loss.mean()


def MLP_train_model_with_logger(
    model,
    optimizer,
    data_loader,
    eval_data_loader,
    loss_module,
    model_index,
    with_eval,
    num_epochs,
):
    # Create TensorBoard logger
    logging_dir = Directory + "logger/MLP_" + str(model_index)
    writer = SummaryWriter(logging_dir)
    model_plotted = False
    # Set model to train mode
    model.train()
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # For the very first batch, we visualize the computation graph in TensorBoard
            if not model_plotted:
                writer.add_graph(model, data_inputs)
                model_plotted = True

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(
                dim=1
            )  # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()

        # Add average loss to TensorBoard
        epoch_loss /= len(data_loader)
        writer.add_scalar("training_loss", epoch_loss, global_step=epoch + 1)
        print("\nepoch:", epoch, "avg. loss:", epoch_loss, end=" ")

        if with_eval:
            model.eval()
            epoch_eval_loss = 0.0
            with torch.no_grad():  # Deactivate gradients for the following code
                for data_inputs, data_labels in eval_data_loader:
                    data_inputs = data_inputs.to(device)
                    data_labels = data_labels.to(device)
                    preds = model(data_inputs)
                    preds = preds.squeeze(dim=1)
                    loss = loss_module(preds, data_labels.float())
                    epoch_eval_loss += loss.item()

            epoch_eval_loss /= len(eval_data_loader)
            writer.add_scalar(
                "eval_loss_training", epoch_eval_loss, global_step=epoch + 1
            )
            print("avg. eval loss:", epoch_eval_loss, end=" ")

    writer.close()


def MLP_train_model(model, optimizer, data_loader, loss_module, num_epochs):
    # Set model to train mode
    model.train()
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(
                dim=1
            )  # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()


def MLP_eval_model(model, data_loader, loss_module, model_index):
    logging_dir = Directory + "logger/MLP_" + str(model_index)
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    model.eval()

    data_index = 0
    num_preds = 0
    losses = []

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            loss = loss_module(preds, data_labels.float())
            losses.append(loss)

            num_preds += data_labels.shape[0]
            data_index += 1
            writer.add_scalar("eval_loss", loss, global_step=data_index)

    writer.close()
    Average_loss = sum(losses) / num_preds
    print("---- EVAL RESULTS ----\n", "Average loss : ", Average_loss)


## Creating And Training The MLP Instances
def DataLoader_for_Model(model_index, test_ratio=0.2, batch_size=1):
    full_dataset = MLPs_datasets[model_index]
    test_size = int(test_ratio * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True
    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_data_loader, test_data_loader


def Model_Maker():
    model = MainNetwork()
    model = model.to(device)
    return model


def Model_Trainer(
    model,
    train_data_loader,
    test_data_loader,
    optimizer,
    loss_module,
    model_index=0,
    with_logger=False,
    with_eval=False,
    num_epochs=100,
):
    if with_logger:
        MLP_train_model_with_logger(
            model,
            optimizer,
            train_data_loader,
            test_data_loader,
            loss_module,
            model_index,
            with_eval,
            num_epochs,
        )
    else:
        MLP_train_model(model, optimizer, train_data_loader, loss_module, num_epochs)
    if with_eval:
        MLP_eval_model(model, test_data_loader, loss_module, model_index)
    return model


def Model_Param_Extractor(model):
    parameters_vectors_sizes = []
    parameters_vectors_sizes_flatten = []
    parameters_vectors_values = []

    for param_tensor in model.state_dict():
        layer = model.state_dict()[param_tensor]
        parameters_vectors_sizes.append(layer.shape)
        parameters_vectors_sizes_flatten.append(torch.flatten(layer).shape[0])
        parameters_vectors_values += torch.flatten(layer).tolist()

    model_params_and_info = [
        parameters_vectors_sizes,
        parameters_vectors_sizes_flatten,
        parameters_vectors_values,
    ]
    return model_params_and_info


def MainModel_Saver(model, model_index):
    PATH = Directory + "/MLP_model_" + str(model_index) + ".pt"
    torch.save(model, PATH)


def Run(data_index):
    global Directory
    Directory = "MLP_Log_problem_" + str(data_index) + "/"
    if not os.path.exists(Directory):
        os.mkdir(Directory)
    else:
        shutil.rmtree(Directory)
        os.mkdir(Directory)

    global N, T, d, X, Drift, Diffusion, input_output_pairs
    PATH = "problem_instance_" + str(data_index) + ".pt"
    problem_instance = torch.load(PATH)
    (
        N,
        T,
        d,
        X,
        Drift,
        Diffusion,
        input_output_pairs,
    ) = problem_instance

    global MLPs_datasets
    MLPs_datasets = Dataset_Generator()

    global MLPs_parameters
    MLPs_parameters = []

    blank_model = MainNetwork()
    PATH = Directory + "/blank_model_instance.pt"
    torch.save(blank_model, PATH)

    number_of_MLPs = T
    # from (X[0,1] -> mean and cov for T=1) to (X[T-1,T] -> mean and cov for T=T)

    for index in range(number_of_MLPs):
        print(
            "=" * 20, "\n", "Start working on MLP", index + 1, "out of ", number_of_MLPs
        )

        print("---- Creating Datasets ----")
        MLP_train_data, MLP_test_data = DataLoader_for_Model(
            index, test_ratio=0.2, batch_size=1
        )

        if index == 0 or index == 2:
            print("---- Creating model instance ----")
            MLP_Model = Model_Maker()

        print("---- Training model instance ----")
        loss_module = CustomLoss()
        if index == 2:
            optimizer = torch.optim.Adam(MLP_Model.parameters(), lr=0.01)
            Model_Trainer(
                MLP_Model,
                MLP_train_data,
                MLP_test_data,
                optimizer,
                loss_module,
                index,
                with_logger=True,
                with_eval=True,
                num_epochs=20,
            )
        elif index > 2:
            optimizer = torch.optim.Adam(MLP_Model.parameters(), lr=0.005)
            Model_Trainer(
                MLP_Model,
                MLP_train_data,
                MLP_test_data,
                optimizer,
                loss_module,
                index,
                with_logger=True,
                with_eval=True,
                num_epochs=10,
            )
        else:
            pass

        print("---- Extracting model parameters ----")
        MLP_Params_and_Info = Model_Param_Extractor(MLP_Model)
        MLPs_parameters.append(MLP_Params_and_Info)

        print("---- Saving model state ----")
        MainModel_Saver(MLP_Model, index)

        print("---- Clearing GPU's cache memory ----")
        torch.cuda.empty_cache()

        print("Done with making MLP", index + 1, "\n")

    PATH = Directory + "MLPs_parameters.pt"
    torch.save(MLPs_parameters, PATH)
