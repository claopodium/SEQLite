import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model.CNN import CNN
from model.SLP import SLP
from model.MLP import MLP
from model.ResNet import ResNet
from model.Transformer import Trans
from model.TV import tv

from data_process.normalize import Normalize
from data_process.VHSE_encoding import to_VHSE_tensor, VHSE_encoding
from data_process.onehot_encoding import to_oh_tensor, one_hot_encode
from vision.loss_plot import loss_plot


def nn_train(df, model_name, encode_name, estimator, epoch_num, batchsize, max_depth, lr, task) -> list:
    save_path = f"./weight/tmp.pth"
    if encode_name == "VHSE":
        encode = to_VHSE_tensor
        map = VHSE_encoding
    elif encode_name == "ONEHOT":
        encode = to_oh_tensor
        map = one_hot_encode

    # model determination
    if model_name == "SLP":
        model = SLP(len(df['x'][1]), encode_name)
    elif model_name == "MLP":
        model = MLP(len(df['x'][1]), encode_name, task)
    elif model_name == "CNN":
        model = CNN()
    elif model_name == "ResNet":
        model = ResNet(encode_name)
    elif model_name == "Transformer":
        model = Trans(encode)
    else:
        raise ValueError("No such model")

    X,y = encode(df)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_log = []
    dataset = TensorDataset(X, y)

    if batchsize is None:
        batchsize = len(X)

    loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True
    )

    if task == "Regression":
        # to correct
        y = Normalize(y)

        criterion = nn.MSELoss()

    elif task == "Binary":
        criterion = nn.CrossEntropyLoss()

    print("Training begin")

    for epoch in range(epoch_num):
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            batch_loss = 0.0

            optimizer.zero_grad()

            y_hat= model(X_batch)

            batch_loss = criterion(y_hat, y_batch)
            if model_name == "SLP":
                batch_loss += tv(model.theta) * batchsize * 1e-4
            epoch_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
        
        loss_log.append(epoch_loss / len(loader))

        if (epoch % 10 == 9):
            print(f"epoch {epoch+1:>4}: Loss = {round(epoch_loss / len(loader),3)}")

    torch.save(model.state_dict(), save_path)

    print("Finished Training")
    loss_plot(loss_log)

    return model