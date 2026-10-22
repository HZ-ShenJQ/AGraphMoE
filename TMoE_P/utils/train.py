import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_metrics(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, rmse, r2


def val(model, dataloader, device):
    model.eval()
    all_loss = 0
    test_logits = []
    test_label = []

    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss = nn.MSELoss()(logits.squeeze(-1), y)
            all_loss += loss.item()
            test_logits.append(logits)
            test_label.append(y)

    test_logits = torch.cat(test_logits, dim=0)
    test_label = torch.cat(test_label, dim=0)
    avg_loss = all_loss / len(dataloader.dataset)

    mse, mae, rmse, r2 = get_metrics(test_logits, test_label)
    return avg_loss, test_logits, test_label, mse, mae, rmse, r2


def train(model, args, train_loader, valid_loader=None, output_print=True):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if len(args.gpu_ids) == 1:
        device = f'cuda:{args.gpu_ids[0]}'
        model = model.to(device)
    elif len(args.gpu_ids) > 1:
        device = f'cuda:{args.gpu_ids[0]}'
        model = nn.DataParallel(model, device_ids=args.gpu_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    Training_list = []
    Val_list = []

    min_loss = float('inf')

    for i_epoch in range(1, args.epoch + 1):
        model.train()
        Train_Loss = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.squeeze().to(device)

            logits = model(x)
            loss = nn.MSELoss()(logits.squeeze(-1), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            Train_Loss += loss.item()

        Training_sample = Train_Loss / len(train_loader.dataset)
        Training_list.append(Training_sample)

        if valid_loader is not None:
            val_loss, _, _, mse, mae, rmse, r2 = val(model, valid_loader, device)
            Val_list.append(val_loss)

            if val_loss < min_loss:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), args.model_save_path)
                else:
                    torch.save(model.state_dict(), args.model_save_path)

                min_loss = val_loss

            if i_epoch % 5 == 0 and output_print:
                print(
                    f'[Prediction Epoch {i_epoch}/{args.epoch}] | Train Loss:{Train_Loss / len(train_loader.dataset):.8f} | Val Loss:{val_loss:.8f} | Val MSE: {mse:.4f} | Val MAE: {mae:.4f} | Val R2: {r2:.4f}')

        else:
            if Train_Loss < min_loss:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), args.model_save_path)
                else:
                    torch.save(model.state_dict(), args.model_save_path)
                min_loss = Train_Loss
            if i_epoch % 5 == 0 and output_print:
                print(f'[Epoch {i_epoch}/{args.epoch}] Training Loss:{Train_Loss / len(train_loader.dataset):.8f}')

    return Training_list, Val_list
