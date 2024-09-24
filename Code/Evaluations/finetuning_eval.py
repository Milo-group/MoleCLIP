import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from sklearn.metrics import  roc_auc_score, recall_score, precision_score, accuracy_score

def ft_eval(model, ff_head, val_dataset, labels_len = 1, task = "classification", device = "cuda", subset = "val"):

    model.eval()
    ff_head.eval()

    data = iter(val_dataset)

    all_preds = torch.zeros((1, labels_len)).to(device)
    all_labels = torch.zeros((1, labels_len)).to(device)

    for it in tqdm(range(len(data))):

        images, labels = next(data)

        embeddings = model.model_image(images).float()
        output = ff_head(embeddings)
        
        if task == "classification":
            output = torch.sigmoid(output)
        
        all_preds = torch.cat([all_preds, output], dim = 0)
        all_labels = torch.cat([all_labels, labels], dim = 0)

    if task == "regression":
        MAE = nn.L1Loss()
        MSE = nn.MSELoss()

        MAE_score = MAE (all_preds[1:], all_labels[1:])
        MSE_score = MSE (all_preds[1:], all_labels[1:])
        RMSE_score = MSE_score ** 0.5

        preds = np.row_stack((all_labels[1:].cpu().detach().numpy().flatten(), all_preds[1:].cpu().detach().numpy().flatten()))

        if subset == 'train':
            return {'Train/MAE': MAE_score, 'Train/MSE': MSE_score, 'Train/RMSE': RMSE_score}
        if subset == 'val':
            return {'Val/MAE': MAE_score, 'Val/MSE': MSE_score, 'Val/RMSE': RMSE_score}
        if subset == 'test':
            return preds, {'Test/MAE': MAE_score, 'Test/MSE': MSE_score, 'Test/RMSE': RMSE_score}
        
    if task == "classification":
        BCE = nn.BCELoss()
        BCE_loss = BCE (all_preds[1:], all_labels[1:])

        prob_np = all_preds[1:].cpu().detach().numpy()
        labels_np = all_labels[1:].cpu().detach().numpy()
        preds_np = np.where(prob_np >= 0.5, 1, 0)

        recall = recall_score(labels_np, preds_np, average = 'micro')
        precision = precision_score(labels_np, preds_np, average = 'micro')
        rocauc = roc_auc_score(labels_np, prob_np)

        labels_np, preds_np = labels_np.flatten(), preds_np.flatten()
        acc = accuracy_score(labels_np, preds_np)

        preds = np.row_stack((all_labels[1:].cpu().detach().numpy().flatten(), all_preds[1:].cpu().detach().numpy().flatten()))

        if subset == 'train':
            return {'Train/accuracy': acc, 'Train/BCE': BCE_loss, 'Train/recall':recall, 'Train/precision':precision, 'Train/roc_auc':rocauc}
        if subset == 'val':
            return {'Val/accuracy': acc, 'Val/BCE': BCE_loss, 'Val/recall':recall, 'Val/precision':precision, 'Val/roc_auc':rocauc}
        if subset == 'test':
            return preds, {'Test/accuracy': acc, 'Test/BCE': BCE_loss, 'Test/recall':recall, 'Test/precision':precision, 'Test/roc_auc':rocauc}
