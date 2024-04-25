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
print("Device in HN_Handler file:", device)


### Dataset creation for HyperNetwork
class HN_Dataset(torch.utils.data.Dataset):
    def __init__(self, input):
        super().__init__()
        self.generate_dataset(input)

    def generate_dataset(self, input):
        self.data = torch.from_numpy(np.array(input[:-1])).type(torch.FloatTensor)
        self.size = len(self.data)
        self.label = torch.from_numpy(np.array(input[1:])).type(torch.FloatTensor)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data_point = self.data[index]
        data_label = self.label[index]
        return data_point, data_label


def HN_dataset_generator(input, test_split=0.2):
    pos = int((len(input) * (1 - test_split)) - 1)
    train_dataset = HN_Dataset(input[: pos + 2])
    test_dataset = HN_Dataset(input[pos:])
    return (train_dataset, test_dataset)


class HyperNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        layers = []

        layers += [nn.Linear(input_size, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, input_size)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input_data):
        result = self.layers(input_data)
        return result


def HN_train_model(
    model, optimizer, data_loader, eval_data_loader, loss_module, num_epochs=100
):
    logging_dir = Directory + "logger/"
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    model.train()

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:

            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            if not model_plotted:
                writer.add_graph(model, data_inputs)
                model_plotted = True

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            loss = loss_module(preds, data_labels.float())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        epoch_eval_loss = 0.0

        with torch.no_grad():
            for data_inputs, data_labels in eval_data_loader:
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)
                loss = loss_module(preds, data_labels.float())
                epoch_eval_loss += loss.item()

        epoch_loss /= len(data_loader)
        writer.add_scalar("training_loss", epoch_loss, global_step=epoch + 1)
        epoch_eval_loss /= len(eval_data_loader)
        writer.add_scalar("eval_loss_training", epoch_eval_loss, global_step=epoch + 1)
        print(
            "epoch:",
            epoch,
            "avg. loss:",
            epoch_loss,
            "avg. eval loss:",
            epoch_eval_loss,
        )
    writer.close()


def HN_eval_model(model, data_loader, loss_module):
    logging_dir = Directory + "logger/eval_log"
    writer = SummaryWriter(logging_dir)

    model.eval()

    num_preds = 0
    losses = []
    _index = 0

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:

            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            loss = loss_module(preds, data_labels.float())

            losses.append(loss)
            num_preds += data_labels.shape[0]
            _index += 1
            writer.add_scalar("eval_loss", loss, global_step=_index)

    writer.close()
    Average_loss = sum(losses) / num_preds
    print("\t\t---- EVAL RESULTS ----\n", "\t\tAverage loss : ", Average_loss)


def Run(data_index):
    global Directory
    Directory = "HN_Log_problem_" + str(data_index) + "/"
    if not os.path.exists(Directory):
        os.mkdir(Directory)
    else:
        shutil.rmtree(Directory)
        os.mkdir(Directory)

    global MLPs_parameters
    PATH = "MLP_Log_problem_" + str(data_index) + "/MLPs_parameters.pt"
    MLPs_parameters = torch.load(PATH)
    MLPs_parameters_count = len(MLPs_parameters[0][2])
    MLPs_weights_and_biases = []
    for index in range(2, len(MLPs_parameters)):
        MLPs_weights_and_biases.append(MLPs_parameters[index][2])

    ### Model Training
    print("=" * 20, "\n", "Start working on Hyper Network")
    print("---- Creating Datasets ----")
    HN_train_dataset, HN_test_dataset = HN_dataset_generator(
        MLPs_weights_and_biases, test_split=0.2
    )
    HN_train_data_loader = torch.utils.data.DataLoader(
        HN_train_dataset, batch_size=1, drop_last=True
    )
    HN_test_data_loader = torch.utils.data.DataLoader(
        HN_test_dataset,
        batch_size=1,
    )

    print("---- Creating model ----")
    print("input size :", MLPs_parameters_count)
    HN_model = HyperNetwork(MLPs_parameters_count)
    HN_model = HN_model.to(device)
    HN_optimizer = torch.optim.Adam(HN_model.parameters(), lr=1e-5)
    HN_loss_module = nn.MSELoss()

    print("---- Training model ----")
    HN_train_model(
        HN_model,
        HN_optimizer,
        HN_train_data_loader,
        HN_test_data_loader,
        HN_loss_module,
        num_epochs=100,
    )

    print("---- Evaluating model ----")
    HN_eval_model(HN_model, HN_test_data_loader, HN_loss_module)

    print("---- Saving model state ----")
    PATH = Directory + "/HN_model.pt"
    torch.save(HN_model, PATH)

    print("---- Clearing GPU's cache memory ----")
    torch.cuda.empty_cache()

    print("Done with making Hyper Network")
