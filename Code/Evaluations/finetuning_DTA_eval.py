import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from sklearn.metrics import  roc_auc_score, recall_score, precision_score, accuracy_score


def dta_eval (model, model_prot, ff_head, test_dataset, device = "cuda"):

    model.eval()
    ff_head.eval()
    model_prot.eval()
    
    data = iter(test_dataset)

    all_preds = torch.zeros(1).to(device)
    all_labels = torch.zeros(1).to(device)

    for it in tqdm(range(len(data))):
        
        labels, smiles, images, targets = next(data)

        mol_embeddings = model.model_image(images).float()
        prot_embeddings = model_prot(targets).float()

        embeddings = torch.cat((mol_embeddings, prot_embeddings), 1)
        output = ff_head(embeddings).flatten()
            
        all_preds = torch.cat([all_preds, output], dim = 0)
        all_labels = torch.cat([all_labels, labels], dim = 0)
    
    MAE = nn.L1Loss()
    MSE = nn.MSELoss()

    MAE_score = MAE (all_preds[1:], all_labels[1:])
    MSE_score = MSE (all_preds[1:], all_labels[1:])
    RMSE_score = MSE_score ** 0.5

    return {'Test/MAE': MAE_score, 'Test/MSE': MSE_score, 'Test/RMSE': RMSE_score}
        