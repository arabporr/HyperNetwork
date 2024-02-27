### Building the predicted MLP
def MLP_pred_instance(model, data):
    predicted_params = HN_predict(model, data)
    layers_parameters = []
    position = 0
    for index in range(len(MLPs_parameters[0][1])):
        layer_size = MLPs_parameters[0][1][index]
        layer_values = predicted_params[position : position + layer_size]
        position += layer_size
        layer_shape = MLPs_parameters[0][0][index]
        layer_params = torch.unflatten(layer_values, 0, layer_shape)
        layers_parameters.append(layer_params)

    Predicted_MLP = MainNetwork()
    state_dict_temp = Predicted_MLP.state_dict()
    layer_number = 0
    for param in state_dict_temp:
        state_dict_temp[param] = layers_parameters[layer_number]
        layer_number += 1

    Predicted_MLP.load_state_dict(state_dict_temp)
    return Predicted_MLP


logging_dir = Directory + "logger/Pred_VS_Real"
writer = SummaryWriter(logging_dir)
model_plotted = False
Losses = []
for MLP_index_1 in range(1, len(MLPs_weights_and_biases) - 1):
    index = MLP_index_1 + 1
    input_data = torch.from_numpy(np.array(MLPs_weights_and_biases[MLP_index_1])).type(
        torch.FloatTensor
    )
    MLP_pred = MLP_pred_instance(HN_model, input_data).to(device)

    real_MLP_path = "./MLP_Models/MLP" + str(index) + ".pt"
    Real_state_dict = torch.load(real_MLP_path)
    MLP_real = MainNetwork().to(device)
    MLP_real.load_state_dict(Real_state_dict)
    MLP_real.eval()

    full_dataset = data.DataLoader(
        MLPs_datasets[index], batch_size=1, shuffle=False, drop_last=False
    )
    PredMLP_Preds = torch.cat(MLP_predict(MLP_pred, full_dataset), dim=0)
    RealMLP_Preds = torch.cat(MLP_predict(MLP_real, full_dataset), dim=0)
    Pred_Real_loss = ((PredMLP_Preds - RealMLP_Preds)).mean()

    writer.add_scalar(
        "Real_MLP_Loss",
        MLP_eval_loss(MLP_real, full_dataset, CustomLoss()),
        global_step=index,
    )
    writer.add_scalar(
        "Pred_MLP_Loss",
        MLP_eval_loss(MLP_pred, full_dataset, CustomLoss()),
        global_step=index,
    )
    writer.add_scalar("Diff in Preds", Pred_Real_loss, global_step=index)

writer.close()
