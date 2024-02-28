import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

global device
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Device in Test and plot file:", device)


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        preds = predictions[0]  # 1 x d(d+1)/2
        mu_pred = preds[:d]
        cov_pred = preds[d:]
        cov_pred = cov_pred.to(device)
        temp = torch.zeros(d, d)
        temp = temp.to(device)
        indices = torch.triu_indices(d, d)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_pred
        temp[indices[1], indices[0]] = cov_pred
        cov_pred = temp

        trgt = targets[0]  # 1 x d(d+1)/2
        mu_trgt = trgt[:d]
        cov_trgt = trgt[d:]
        cov_trgt = cov_trgt.to(device)
        temp = torch.zeros(d, d)
        temp = temp.to(device)
        indices = torch.triu_indices(d, d)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_trgt
        temp[indices[1], indices[0]] = cov_trgt
        cov_trgt = temp

        loss = 0.0

        # Mean Component
        loss += torch.mean(torch.pow(mu_pred - mu_trgt, 2))

        # Covariant Component
        A = torch.matmul(torch.linalg.pinv(cov_pred), cov_trgt)

        A = torch.nan_to_num(A)

        A_eigen = torch.linalg.eig(A)
        A_eigen = A_eigen.eigenvalues
        A_eigen = torch.real(A_eigen)
        A_eigen = torch.log(A_eigen) ** 2
        A_eigen = torch.mean(A_eigen)
        loss += A_eigen / 2
        return loss.mean()


def HN_predict(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        preds = model(data)
    return preds


def MLP_predict(model, data_loader):
    results = []
    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            results.append(preds)
    return results


def MLP_eval_loss(model, data_loader, loss_module):
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
    Average_loss = sum(losses) / num_preds
    return Average_loss


def Blank_MLP():
    blank_model = torch.load(Blank_MLP_Path)
    blank_model.eval()
    return blank_model


### Building the predicted MLP
def HN_pred_to_instance(model, data):
    predicted_params = HN_predict(model, data)
    Predicted_MLP = Blank_MLP()
    state_dict_temp = Predicted_MLP.state_dict()
    position = 0
    for param in enumerate(state_dict_temp):
        layer = state_dict_temp[param]
        layer_size = torch.flatten(layer).shape[0]
        layer_values = predicted_params[position : position + layer_size]
        position += layer_size
        layer_params = torch.unflatten(layer_values, 0, layer.shape)
        state_dict_temp[param] = layer_params
    Predicted_MLP.load_state_dict(state_dict_temp)
    return Predicted_MLP


def Run(data_index):
    global Directory
    Directory = "Testing_Results_problem_" + str(data_index) + "/"

    global N, T, d
    PATH = "problem_instance_" + str(data_index) + ".pt"
    N, T, d = torch.load(PATH)[:3]

    global Blank_MLP_Path
    Blank_MLP_Path = "MLP_Log_problem_" + str(data_index) + "/blank_model_instance.pt"

    global HN_model
    PATH = "HN_Log_problem_" + str(data_index) + "/HN_model.pt"
    HN_model = torch.load(PATH)
    HN_model.eval()

    PATH = "MLP_Log_problem_" + str(data_index) + "/MLPs_parameters.pt"
    MLPs_parameters = torch.load(PATH)
    MLPs_weights_and_biases = MLPs_parameters[:][2]

    PATH = "MLP_Log_problem_" + str(data_index) + "/MLPs_Datasets.pt"
    MLPs_datasets = torch.load(PATH)

    logging_dir = Directory + "logger/Pred_VS_Real"
    writer = SummaryWriter(logging_dir)

    for input_index in range(len(MLPs_weights_and_biases) - 1):
        output_index = input_index + 1
        input_data = torch.from_numpy(
            np.array(MLPs_weights_and_biases[input_index])
        ).type(torch.FloatTensor)
        MLP_pred = HN_pred_to_instance(HN_model, input_data).to(device)

        real_MLP_path = (
            "MLP_Log_problem_"
            + str(data_index)
            + "/MLP_model_"
            + str(output_index)
            + ".pt"
        )
        Real_state_dict = torch.load(real_MLP_path)
        MLP_real = Blank_MLP().to(device)
        MLP_real.load_state_dict(Real_state_dict)
        MLP_real.eval().to(device)

        full_dataset = torch.utils.data.DataLoader(
            MLPs_datasets[output_index], batch_size=1, shuffle=False, drop_last=False
        )
        PredMLP_preds = torch.cat(MLP_predict(MLP_pred, full_dataset), dim=0)
        RealMLP_preds = torch.cat(MLP_predict(MLP_real, full_dataset), dim=0)
        Pred_Real_loss = ((PredMLP_preds - RealMLP_preds)).mean()

        writer.add_scalar(
            "Real_MLP_Loss",
            MLP_eval_loss(MLP_real, full_dataset, CustomLoss()),
            global_step=output_index,
        )
        writer.add_scalar(
            "Pred_MLP_Loss",
            MLP_eval_loss(MLP_pred, full_dataset, CustomLoss()),
            global_step=output_index,
        )
        writer.add_scalar(
            "difference in predictions", Pred_Real_loss, global_step=output_index
        )

    writer.close()
