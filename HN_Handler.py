import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)


CNNs_parameters = torch.load("CNNs_parameters.pt")


### Dataset creation for HyperNetwork
class HN_Dataset(torch.utils.data.Dataset):
    def __init__(self, CNNs_parameters):
        super().__init__()
        self.generate_dataset(CNNs_parameters)

    def generate_dataset(self, CNNs_parameters):
        self.data = torch.from_numpy(np.array(CNNs_parameters[:-1])).type(
            torch.FloatTensor
        )
        self.size = len(self.data)
        self.label = torch.from_numpy(np.array(CNNs_parameters[1:])).type(
            torch.FloatTensor
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data_point = self.data[index]
        data_label = self.label[index]
        return data_point, data_label


CNNs_parameters_count = len(CNNs_parameters[0][2])
CNNs_weights_and_biases = []
for index in range(len(CNNs_parameters)):
    CNNs_weights_and_biases.append(CNNs_parameters[index][2])


def HN_dataset_generator(test_split=0.2):
    pos = (
        int((len(CNNs_weights_and_biases) * (1 - test_split)) - 1)
        if test_split < 1
        else int(len(CNNs_weights_and_biases) - test_split - 1)
    )

    train_dataset = HN_Dataset(CNNs_weights_and_biases[:pos])
    test_dataset = HN_Dataset(CNNs_weights_and_biases[pos:])

    return (train_dataset, test_dataset)


HN_Train, HN_Test = HN_dataset_generator(test_split=0.1)


# TESTING
# print(CNNs_weights_and_biases[:2])
# print("Size of dataset:", len(HN_dataset))
# print(len(HN_dataset[0][0]))
# print("Dataset : ", [HN_dataset[i] for i in range(len(HN_dataset))])
# HN_data_loader = data.DataLoader(HN_dataset, batch_size=1, shuffle=True, drop_last=True)
# for data_inputs, data_labels in HN_data_loader:
#     print("Data inputs", data_inputs.shape, "\n", data_inputs)
#     print("Data labels", data_labels.shape, "\n", data_labels)
### Model Architecture Definition
class HyperNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        # Create the network based on the specified hidden sizes
        layers = []

        layers += [nn.Linear(input_size, 1024)]  # Affine Layer
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, 1024)]  # Affine Layer
        layers += [nn.ReLU()]

        layers += [nn.Linear(1024, input_size)]  # Affine Layer

        self.layers = nn.Sequential(
            *layers
        )  # nn.Sequential summarizes a list of modules into a single module, applying them in sequence

    def forward(self, input_data):
        result = self.layers(input_data)
        return result


def HN_train_model(
    model, optimizer, data_loader, eval_data_loader, loss_module, num_epochs=100
):
    # Create TensorBoard logger
    logging_dir = "logger/HyperNetwork"
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
        writer.add_scalar("eval_loss_training", epoch_eval_loss, global_step=epoch + 1)
    writer.close()


def HN_eval_model(model, data_loader, loss_module):
    # Create TensorBoard logger
    logging_dir = "logger/HyperNetwork"
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    # Set model to eval mode
    model.eval()

    num_preds = 0
    losses = []
    data_index = 0

    with torch.no_grad():  # Deactivate gradients for the following code
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
    print("\t\t---- EVAL RESULTS ----\n", "\t\tAverage loss : ", Average_loss)


def HN_predict(model, data):
    model.eval()  # Set model to eval mode
    with torch.no_grad():  # Deactivate gradients for the following code
        data = data.to(device)
        preds = model(data)
    return preds


### Model Training
print("=" * 20, "\n", "Start working on Hyper Network")
print("---- Creating Datasets ----")
HN_train_data_loader = torch.utils.data.DataLoader(
    HN_Train, batch_size=1, shuffle=False, drop_last=True
)
HN_test_data_loader = torch.utils.data.DataLoader(
    HN_Test, batch_size=1, shuffle=False, drop_last=False
)

print("---- Creating model ----")
print("input size :", CNNs_parameters_count)
HN_model = HyperNetwork(CNNs_parameters_count)
HN_model = HN_model.to(device)
HN_optimizer = torch.optim.Adam(HN_model.parameters(), lr=0.0001)
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
if not os.path.exists("HN_Model/"):
    os.mkdir("HN_Model/")
PATH = "./HN_Model/HyperNetwork.pt"
torch.save(HN_model.state_dict(), PATH)

print("---- Clearing GPU's cache memory ----")
torch.cuda.empty_cache()

print("Done with Hyper Network")


### Building the predicted CNN
def CNN_pred_instance(model, data):
    predicted_params = HN_predict(model, data)
    layers_parameters = []
    position = 0
    for index in range(len(CNNs_parameters[0][1])):
        layer_size = CNNs_parameters[0][1][index]
        layer_values = predicted_params[position : position + layer_size]
        position += layer_size
        layer_shape = CNNs_parameters[0][0][index]
        layer_params = torch.unflatten(layer_values, 0, layer_shape)
        layers_parameters.append(layer_params)

    Predicted_CNN = MainNetwork()
    state_dict_temp = Predicted_CNN.state_dict()
    layer_number = 0
    for param in state_dict_temp:
        state_dict_temp[param] = layers_parameters[layer_number]
        layer_number += 1

    Predicted_CNN.load_state_dict(state_dict_temp)
    return Predicted_CNN


logging_dir = "logger/Pred_VS_Real"
writer = SummaryWriter(logging_dir)
model_plotted = False
Losses = []

for CNN_index_1 in range(1, len(CNNs_weights_and_biases) - 1):
    index = CNN_index_1 + 1
    input_data = torch.from_numpy(np.array(CNNs_weights_and_biases[CNN_index_1])).type(
        torch.FloatTensor
    )
    CNN_pred = CNN_pred_instance(HN_model, input_data).to(device)

    real_cnn_path = "./CNN_Models/CNN" + str(index) + ".pt"
    Real_state_dict = torch.load(real_cnn_path)
    CNN_real = MainNetwork().to(device)
    CNN_real.load_state_dict(Real_state_dict)
    CNN_real.eval()

    full_dataset = data.DataLoader(
        CNNs_datasets[index], batch_size=1, shuffle=False, drop_last=False
    )
    PredCNN_Preds = torch.cat(CNN_predict(CNN_pred, full_dataset), dim=0)
    RealCNN_Preds = torch.cat(CNN_predict(CNN_real, full_dataset), dim=0)
    Pred_Real_loss = ((PredCNN_Preds - RealCNN_Preds)).mean()

    writer.add_scalar(
        "Real_CNN_Loss",
        CNN_eval_loss(CNN_real, full_dataset, CustomLoss()),
        global_step=index,
    )
    writer.add_scalar(
        "Pred_CNN_Loss",
        CNN_eval_loss(CNN_pred, full_dataset, CustomLoss()),
        global_step=index,
    )
    writer.add_scalar("Diff in Preds", Pred_Real_loss, global_step=index)

writer.close()
