import os
import shutil
import copy

import numpy as np

import torch
import tensorflow

from MLP_Handler import CustomLoss

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
        loss += torch.sqrt((mu_pred - mu_trgt) ** 2) ** 2

        # Covariant Component
        C2_tf = tensorflow.convert_to_tensor(cov_pred)
        C2_sqrt = torch.from_numpy(tensorflow.linalg.sqrtm(C2_tf).numpy())
        C2sqrt_C1_C2sqrt = torch.matmul(torch.matmul(C2_sqrt, cov_trgt), C2_sqrt)
        C2sqrt_C1_C2sqrt_tf = tensorflow.convert_to_tensor(C2sqrt_C1_C2sqrt)
        SQRT_C2sqrt_C1_C2sqrt = torch.from_numpy(
            tensorflow.linalg.sqrtm(C2sqrt_C1_C2sqrt_tf).numpy()
        )
        inside_trace = cov_trgt + cov_pred + 2 * SQRT_C2sqrt_C1_C2sqrt
        loss += torch.trace(inside_trace)
        if torch.sum(loss.isnan()) > 0:
            print("loss is nan!")
            raise Exception("Loss is nan!")
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
    model_clone = copy.deepcopy(blank_model)
    return model_clone


def HN_pred_to_instance(predicted_params):
    Predicted_MLP = Blank_MLP()
    state_dict_temp = Predicted_MLP.state_dict()
    position = 0
    for layer_index, param in enumerate(state_dict_temp):
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
    if not os.path.exists(Directory):
        os.mkdir(Directory)
    else:
        shutil.rmtree(Directory)
        os.mkdir(Directory)

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
    MLPs_weights_and_biases = []
    for index in range(2, len(MLPs_parameters)):
        MLPs_weights_and_biases.append(MLPs_parameters[index][2])

    PATH = "MLP_Log_problem_" + str(data_index) + "/MLPs_Datasets.pt"
    MLPs_datasets = torch.load(PATH)

    logging_dir = Directory + "logger/"
    writer = SummaryWriter(logging_dir)

    for input_index in range(len(MLPs_weights_and_biases) - 1):
        print("creating testing visualizations for index:", input_index + 1)
        output_index = input_index + 1
        input_data = torch.from_numpy(
            np.array(MLPs_weights_and_biases[input_index])
        ).type(torch.FloatTensor)
        predicted_params = HN_predict(HN_model, input_data)
        if input_index < 0.8 * len(MLPs_weights_and_biases):
            last_pred = predicted_params
        else:
            last_pred = HN_predict(HN_model, last_pred)
            MLP_pred_recur = HN_pred_to_instance(last_pred).to(device)
            MLP_pred_recur.eval()
        MLP_pred = HN_pred_to_instance(predicted_params).to(device)
        MLP_pred.eval()
        real_MLP_path = (
            "MLP_Log_problem_"
            + str(data_index)
            + "/MLP_model_"
            + str(output_index)
            + ".pt"
        )
        MLP_real = torch.load(real_MLP_path)
        MLP_real = MLP_real.to(device)
        MLP_real.eval()

        full_dataset = torch.utils.data.DataLoader(
            MLPs_datasets[output_index + 2],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        PredMLP_preds = torch.cat(MLP_predict(MLP_pred, full_dataset), dim=0)
        RealMLP_preds = torch.cat(MLP_predict(MLP_real, full_dataset), dim=0)
        Pred_Real_loss = ((PredMLP_preds - RealMLP_preds) ** 2).mean()

        if input_index < 0.8 * len(MLPs_weights_and_biases):
            writer.add_scalars(
                "Loss Plot",
                {
                    "Real MLP's Loss": MLP_eval_loss(
                        MLP_real, full_dataset, CustomLoss()
                    ),
                    "Predicted MLP's Loss": MLP_eval_loss(
                        MLP_pred, full_dataset, CustomLoss()
                    ),
                },
                global_step=output_index,
            )
            writer.add_scalar(
                "Difference in predictions between Real and Predicted MLP",
                Pred_Real_loss,
                global_step=output_index,
            )
        else:
            RecurPredMLP_preds = torch.cat(
                MLP_predict(MLP_pred_recur, full_dataset), dim=0
            )
            RecurPred_Pred_loss = ((RecurPredMLP_preds - PredMLP_preds) ** 2).mean()
            writer.add_scalars(
                "Loss Plot",
                {
                    "Real MLP's Loss": MLP_eval_loss(
                        MLP_real, full_dataset, CustomLoss()
                    ),
                    "Predicted MLP's Loss": MLP_eval_loss(
                        MLP_pred, full_dataset, CustomLoss()
                    ),
                    "Recurrently Predicted MLP's Loss": MLP_eval_loss(
                        MLP_pred_recur, full_dataset, CustomLoss()
                    ),
                },
                global_step=output_index,
            )
            writer.add_scalar(
                "Difference in predictions between Real and Predicted MLP",
                Pred_Real_loss,
                global_step=output_index,
            )
            writer.add_scalar(
                "Difference in predictions between Predicted and Recurrently Predicted MLP",
                RecurPred_Pred_loss,
                global_step=output_index,
            )

    writer.close()
