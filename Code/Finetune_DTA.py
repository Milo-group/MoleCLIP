import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import requests

import torch
from torch import nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from Data.Dataloaders import DTADataset
from Model.Load_model import Model, create_mlp, Conv1D
from Evaluations.finetuning_DTA_eval import dta_eval
from Pretrain import save_checkpoint

import wandb

def download_MoleCLIP_weights (output_path):

    # Download the .gz file
    url = "https://zenodo.org/records/13826016/files/MoleCLIP%20-%20Primary.pth?download=1"
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as handle:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading MoleCLIP primary") as pbar:
            for data in response.iter_content(chunk_size=1024):
                handle.write(data)
                pbar.update(len(data))


def train_model(model, train_dataset, test_dataset, dataset_name, run_name = '', checkpoint = None):

    print ("""

           starting new training...

           """)

    print (f"dataset: {dataset_name}")

    ff_head = create_mlp(input_dim = 1024, inner_layers=args.layers, inner_dim=args.d_ff, dropout_rate=args.dropout, output_dim=1).to(args.device)
    model_prot = Conv1D(num_features_xt=25, n_filters=32, embed_dim=128, output_dim=512).to(args.device)

    optim1 = torch.optim.Adam(model.model_image.parameters(), lr=args.lr_bb, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wd) 

    optim2 = torch.optim.Adam(
        list(ff_head.parameters()) + list(model_prot.parameters()), 
        lr=args.lr_ff, 
        weight_decay=args.wd
    )

    if checkpoint is not None and 'optimizer' in checkpoint:
        checkpoint['optimizer']['param_groups'][0]['lr'] = args.lr_bb
        checkpoint['optimizer']['param_groups'][0]['weight_decay'] = args.wd
        optim1.load_state_dict(checkpoint['optimizer'])    
    

    loss_func = nn.MSELoss()
    
    loss = 0
    total_loss = 0

    step = 1
    val_improvement = 0 
    min_val = -1

    start_time = time.time()
    current_time = start_time

    best_test = -1

    #start training
    for epoch in range(args.epochs):

        epoch_it = iter(train_dataset)

        for it in range(len(epoch_it)):
            t = time.time()

            optim1.zero_grad() 
            optim2.zero_grad() 
            
            labels, smiles, images, targets = next(epoch_it)

            mol_embeddings = model.model_image(images).float()
            prot_embeddings = model_prot(targets).float()


            embeddings = torch.cat((mol_embeddings, prot_embeddings), 1)
            output = ff_head(embeddings)

            loss = loss_func(output, labels.view(-1, 1))


            loss.backward()
            optim1.step()
            optim2.step()
            
            total_loss += loss 
                   
            if step % args.log_every == 0: #log every a defined number of epochs
                
                it_dir = {'specs/time': (time.time() - start_time) // 60, 'specs/iteration': it + 1,
                'Train/loss':loss.item(), 'specs/time per iteration': (time.time() - current_time) / args.log_every,
                'Train/avg_training_loss':(total_loss / args.log_every).item(), 'specs/epoch':step / len(epoch_it)}

                print (it_dir)
                
                if args.wandb:
                    wandb.log(it_dir, step = step)

                total_loss = 0
                current_time = time.time()

            step += 1
        
        if (epoch + 1) % args.eval_every == 0: #evaluate every a defined number of epochs
            print (f"start eval epoch {epoch + 1}")
            with torch.no_grad():

                # model.eval()
                # ff_head.eval()
                # model_prot.eval()

                it_dir = dta_eval(model, model_prot, ff_head, test_dataset, device = args.device)
                print ("test:")
                print (it_dir)
                print (f'Test/{args.test_metric}')

                current_test = it_dir[f'Test/{args.test_metric}']

                if args.wandb:
                    wandb.log(it_dir, step = step)

                model.train()
                ff_head.train()
                model_prot.train()

                if best_test == -1:
                    best_test = current_test
 
                if (args.test_metric in ["RMSE", "MSE", "MAE", "BCE"] and current_test <= best_test) or (args.test_metric in ["roc_auc", "accuracy"] and current_test >= best_test):

                    best_test = current_test
                    it_dir = {'Max/best_test': best_test}

                    if args.wandb:
                        wandb.log(it_dir, step = step)
                

        if args.epochs_cp > 0:
            if not os.path.isdir(f"{args.cp_path}/finetuning"):
                os.mkdir(f"{args.datasets_path}/{args.dataset_name}/finetuning") 

            if (epoch + 1) % args.epochs_cp == 0:
                if not os.path.isdir(f"{args.cp_path}/finetuning/{run_name}"):
                    os.mkdir(f"{args.cp_path}/finetuning/{run_name}")              
                save_checkpoint(model, optim1, args.cp_path, f"finetuning/{run_name}", epoch+1, step, optim2=optim2)
    
    print ("""
    
    finetuning summary:""")
    print ("-------------------")
    print (f"dataset: {dataset_name}")
    print (f"parameters: {hp}")
    print (f"best test evaluation: {best_test}")
    print (f"test evaluation in minimum validation loss: {test_in_min}")
    print ("-------------------")

def load_model(args, dataset_name):

    model = Model(args.vit_model, args.no_parallel, args.device, classes = []).to(args.device)

    full_cp_name = f"{args.cp_name.split(' - ')[0]}/{args.cp_name}"
    model_path = f"{args.cp_path}/{full_cp_name}"

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    checkpoint['model'] = {key: value for key, value in checkpoint['model'].items() if 'cls' not in key and 'clip_model' not in key}

    model.load_state_dict(checkpoint['model'])

    config = {
    "batch_size": args.batch,
    "lr_ff": args.lr_ff,
    "lr_bb": args.lr_bb,
    "dropout": args.dropout,
    "d_ff": args.d_ff,
    "layers": args.layers,
    "wd": args.wd,
    "pretrain": model_path.split("/")[-1],
    "dataset": dataset_name,
    "val_metric": args.test_metric,
    "test_metric": args.test_metric
    }

    run_name = f"{datetime.now().strftime('%D %H-%M').replace('/', '-')}_{args.comment}"

    return model, checkpoint, config, run_name


def main(args):

    if args.wandb:
        wandb.login(key = args.wandb_key)

    full_cp_name = f"{args.cp_name.split(' - ')[0]}/{args.cp_name}"
    
    if not os.path.isfile(f"{args.cp_path}/{full_cp_name}"):

        if cp == "MoleCLIP - Primary.pth" and args.download:
            if not os.path.isdir(f"{args.cp_path}/MoleCLIP"):
                os.mkdir(f"{args.cp_path}/MoleCLIP")

            download_MoleCLIP_weights (f"{args.cp_path}/{full_cp_name}")

        else:
            raise(Exception(f"{cp} file wasn't found")) 

    print ("start finetuning")


    for dataset_name in args.dataset_names: 

        # Example usage
        train_data = DTADataset(dataset_name, mode = "train", device=args.device)  # Replace with your actual data
        train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)

        # Example usage
        test_data = DTADataset(dataset_name, mode = "test", device=args.device)  # Replace with your actual data
        test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=False)

        print ("Datasets were sucssesfully loaded")
   
        #load model
        model, checkpoint, config, run_name = load_model(args, dataset_name)
        print ("model was sucssesfully loaded")

        if args.wandb:
            wandb.init(project=args.wandb_project, name = run_name, config = config)
        
        train_model (model, train_dataloader, test_dataloader, dataset_name, run_name, checkpoint = checkpoint)
        
        if args.wandb:
            wandb.finish()
        
        torch.cuda.empty_cache()


def parse_args():

    parser = argparse.ArgumentParser(description='Parameters for pretraining ImageMol')

    parser.add_argument('-device', default="cuda", type=str, choices=["cuda", "cpu"], help='Device to use: "cuda" or "cpu".')
    parser.add_argument('-no_parallel', action='store_false', help='Disable multi-GPU training. Set to use only a single GPU.')

    parser.add_argument('-lr_bb', default=5e-6, nargs='+', type=float, help='Initial learning rate for the model backbone.')
    parser.add_argument('-lr_ff', default=5e-4, nargs='+', type=float, help='Initial learning rate for the feed-forward layers.')
    parser.add_argument('-wd', default=1e-5, nargs='+', type=float, help='Weight decay coefficient(L2 regularization).')
    parser.add_argument('-epochs', default=500, type=int, help='Total number of epochs.')
    parser.add_argument('-dropout', default=0, nargs='+', type=float, help='Dropout rate for the layers.')
    parser.add_argument('-batch', default=256, nargs='+', type=int, help='Batch size')
    parser.add_argument('-d_ff', default=512, nargs='+', type=int, help='Dimension of the feed-forward middle layers.')
    parser.add_argument('-layers', default=1, nargs='+', type=int, help='Number of feed-forward middle layers.')

    parser.add_argument('-epochs_cp', default=0, type=int, help='Number of epochs between checkpoints.')
    parser.add_argument('-log_every', default=100, type=int, help='Iteration interval for logging.')
    parser.add_argument('-eval_every', default=1, type=int, help='Number of epochs between evaluations.')

    parser.add_argument('-datasets_path', default="../Datasets/Finetuning", type=str, help='Path to the directory containing the dataset.')
    parser.add_argument('-dataset_names', default=["kiba", "davis"], nargs='+', type=str, help='Names of the dataset to be used.')

    parser.add_argument('-test_metric', default="MSE", choices=["MAE", "RMSE", "MSE", "BCE", "roc_auc", "accuracy"], help='Metric for evaluating test performance.')

    parser.add_argument('-cp_path', default='../Checkpoints', type=str, help='Directory for storing checkpoints.')
    parser.add_argument('-cp_name', default='MoleCLIP - Primary.pth', type=str, help='Name of the checkpoint to use (.pth extension)')
    parser.add_argument('-vit_model', type=str, default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"], help='Variant of the Vision Transformer model.')

    parser.add_argument('-download', action='store_true', help='Whether to use Weights & Biases for logging.')
    parser.add_argument('-wandb', action='store_true', help='Whether to use Weights & Biases for logging.')
    parser.add_argument('-wandb_key', default="", type=str, help='Weights & Biases API key.')
    parser.add_argument('-wandb_project', default="", type=str, help='Name of the wandb project for logging runs.')
    parser.add_argument('-comment', default="DTA", type=str, help='Comment to add to the run name.')

    parser.set_defaults(wandb=False, download=False, resume=False, no_parallel=True, save_predictions=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

