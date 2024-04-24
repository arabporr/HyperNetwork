import os
import shutil
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import tensorflow as tf

from MLP_Handler import CustomLoss

from torch.utils.tensorboard import SummaryWriter

global device
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Device in Test and plot file:", device)


def Wasserstein_Distance(vec1, vec2):
    Total_Loss = []
    for v1, v2 in zip(vec1, vec2):
        d_ = int((-3 + np.sqrt(9 + 8 * (v2.shape[0]))) / 2)
        mu_v1 = v1[:d_]
        cov_v1 = v1[d_:]
        cov_v1 = cov_v1.to(device)
        temp = torch.zeros(d_, d_)
        temp = temp.to(device)
        indices = torch.triu_indices(d_, d_)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_v1
        temp[indices[1], indices[0]] = cov_v1
        cov_v1 = temp + torch.eye(d_).to(device) * 1e-4

        mu_v2 = v2[:d_]
        cov_v2 = v2[d_:]
        cov_v2 = cov_v2.to(device)
        temp = torch.zeros(d_, d_)
        temp = temp.to(device)
        indices = torch.triu_indices(d_, d_)
        indices.to(device)
        temp[indices[0], indices[1]] = cov_v2
        temp[indices[1], indices[0]] = cov_v2
        cov_v2 = temp + torch.eye(d_).to(device) * 1e-4

        loss = 0.0

        # Mean Component
        loss += torch.linalg.vector_norm((mu_v1 - mu_v2), 2)

        # Covariant Component
        cov_v1 = cov_v1.to(torch.device("cpu"))
        cov_v2 = cov_v2.to(torch.device("cpu"))
        C2_tf = tf.convert_to_tensor(cov_v2)
        C2_sqrt = torch.from_numpy(tf.linalg.sqrtm(C2_tf).numpy())
        C2sqrt_C1_C2sqrt = torch.matmul(torch.matmul(C2_sqrt, cov_v1), C2_sqrt)
        C2sqrt_C1_C2sqrt_tf = tf.convert_to_tensor(C2sqrt_C1_C2sqrt)
        SQRT_C2sqrt_C1_C2sqrt = torch.from_numpy(
            tf.linalg.sqrtm(C2sqrt_C1_C2sqrt_tf).numpy()
        )

        inside_trace = cov_v1 + cov_v2 - 2 * SQRT_C2sqrt_C1_C2sqrt
        loss += torch.trace(inside_trace)
        if torch.sum(loss.isnan()) > 0:
            print("loss is nan!")
            raise Exception("Loss is nan!")

        Total_Loss.append(loss.to(torch.device("cpu")))
    Total_Loss = np.abs(np.array(Total_Loss))
    return Total_Loss.mean()


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


def MLP_Custom_loss(model, data_loader, loss_module):
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
    for index in range(len(MLPs_parameters)):
        MLPs_weights_and_biases.append(MLPs_parameters[index][2])

    PATH = "MLP_Log_problem_" + str(data_index) + "/MLPs_Datasets.pt"
    MLPs_datasets = torch.load(PATH)

    logging_dir = Directory + "logger/"
    writer = SummaryWriter(logging_dir)

    Plots_Data_Dict = {
        "MSE_Real_vs_Pred": [],
        "Accual_Loss_Real": [],
        "Accual_Loss_Pred": [],
        "Accual_Loss_Recur_Pred": [],
        "T": [],
    }

    for input_index in range(2, len(MLPs_weights_and_biases) - 1):
        Plots_Data_Dict["T"].append(input_index + 1)
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
            MLPs_datasets[output_index],
            batch_size=100,
            shuffle=False,
            drop_last=False,
        )
        PredMLP_preds = torch.cat(MLP_predict(MLP_pred, full_dataset), dim=0)
        RealMLP_preds = torch.cat(MLP_predict(MLP_real, full_dataset), dim=0)
        Pred_Real_loss = ((PredMLP_preds - RealMLP_preds) ** 2).mean()
        Plots_Data_Dict["MSE_Real_vs_Pred"].append(Pred_Real_loss.cpu().numpy())

        if input_index < 0.8 * len(MLPs_weights_and_biases):
            Real_Loss = MLP_Custom_loss(MLP_real, full_dataset, CustomLoss())
            Pred_Loss = MLP_Custom_loss(MLP_pred, full_dataset, CustomLoss())
            Plots_Data_Dict["Accual_Loss_Real"].append(Real_Loss.cpu().numpy())
            Plots_Data_Dict["Accual_Loss_Pred"].append(Pred_Loss.cpu().numpy())
            Plots_Data_Dict["Accual_Loss_Recur_Pred"].append(
                torch.tensor(0).cpu().numpy()
            )

            writer.add_scalars(
                "Loss Plot",
                {
                    "Real MLP's Loss": Real_Loss,
                    "Predicted MLP's Loss": Pred_Loss,
                },
                global_step=output_index,
            )
            writer.add_scalar(
                "Difference in predictions between Real and Predicted MLP",
                Pred_Real_loss,
                global_step=output_index,
            )
            # writer.add_scalar(
            #     "Wasserstein Distance between predictions of Real and Predicted MLP",
            #     Wasserstein_Distance(RealMLP_preds, PredMLP_preds),
            #     global_step=output_index,
            # )
        else:
            RecurPredMLP_preds = torch.cat(
                MLP_predict(MLP_pred_recur, full_dataset), dim=0
            )
            # RecurPred_Pred_loss = ((RecurPredMLP_preds - PredMLP_preds) ** 2).mean()

            Real_Loss = MLP_Custom_loss(MLP_real, full_dataset, CustomLoss())
            Pred_Loss = MLP_Custom_loss(MLP_pred, full_dataset, CustomLoss())
            Recur_Pred_Loss = MLP_Custom_loss(
                MLP_pred_recur, full_dataset, CustomLoss()
            )
            Plots_Data_Dict["Accual_Loss_Real"].append(Real_Loss.cpu().numpy())
            Plots_Data_Dict["Accual_Loss_Pred"].append(Pred_Loss.cpu().numpy())
            Plots_Data_Dict["Accual_Loss_Recur_Pred"].append(
                Recur_Pred_Loss.cpu().numpy()
            )

            writer.add_scalars(
                "Loss Plot",
                {
                    "Real MLP's Loss": Real_Loss,
                    "Predicted MLP's Loss": Pred_Loss,
                    "Recurrently Predicted MLP's Loss": Recur_Pred_Loss,
                },
                global_step=output_index,
            )
            writer.add_scalar(
                "Difference in predictions between Real and Predicted MLP",
                Pred_Real_loss,
                global_step=output_index,
            )
            # writer.add_scalar(
            #     "Difference in predictions between Predicted and Recurrently Predicted MLP",
            #     RecurPred_Pred_loss,
            #     global_step=output_index,
            # )
            # writer.add_scalar(
            #     "Wasserstein Distance between predictions of Real and Predicted MLP",
            #     Wasserstein_Distance(RealMLP_preds, PredMLP_preds),
            #     global_step=output_index,
            # )
            # writer.add_scalar(
            #     "Wasserstein Distance between predictions of Real and Recurrently Predicted MLP",
            #     Wasserstein_Distance(RealMLP_preds, RecurPredMLP_preds),
            #     global_step=output_index,
            # )

    writer.close()

    Directory = "Results_" + str(data_index) + "/"
    if not os.path.exists(Directory):
        os.mkdir(Directory)
    else:
        shutil.rmtree(Directory)
        os.mkdir(Directory)
    plots_data_df = pd.DataFrame.from_dict(Plots_Data_Dict).set_index("T")
    plots_data_df.to_csv(Directory + "plot_data.csv")

    df = plots_data_df.copy()
    df = df.iloc[150:]

    ax = plt.figure(figsize=(20, 10))
    sns.set_style("darkgrid")
    plt.title(
        "Loss function values for the outputs of Trained MLPs, Predicted MLPs, and Recurrently Predicted MLPs"
    )
    plt.xlabel("Time (T)")
    plt.ylabel("Log 10 of the loss values (calculated with the formula in paper)")

    lg = lambda x: np.log10(x)
    for col in df.drop(columns=["MSE_Real_vs_Pred"]).columns:
        plt.plot(df[col].apply(lg), label=col)

    plt.legend()
    plt.savefig(Directory + "Loss_Plot.pdf")
    plt.show()
