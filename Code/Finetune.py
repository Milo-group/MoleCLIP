import sys
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import requests

import torch
import torchvision
from torch import nn
import numpy as np

from sklearn.model_selection import ParameterGrid

from Data.Dataloaders import Load_finetuning_dataset
from Model.Load_model import Model, create_mlp
from Evaluations.finetuning_eval import ft_eval
from Pretrain import save_checkpoint

import wandb

dataset_tasks_dict = {"bace":["classification", "BCE", "roc_auc"],
                    "bbbp":["classification", "BCE", "roc_auc"],
                    "esol":["regression", "MAE", "RMSE"],
                    "freesolv":["regression", "MAE", "RMSE"],
                    "DHBDs":["regression", "MAE", "MAE"],
                    "NHCs":["regression", "MAE", "MAE"],
                    "Phosphines_yield":["regression", "MAE", "MAE"],
                    "Phosphines_selectivity":["regression", "MAE", "MAE"],
                    "Kraken_buried":["regression", "MAE", "MAE"]}

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


def train_model(model, train_dataset, val_dataset, test_dataset, hp, labels_len, dataset_name, run_name = '', checkpoint = None, device = "cuda"):

    print ("""

           starting new training...

           """)

    print (f"dataset: {dataset_name}")
    print (f"parameters: {hp}")

    ff_head = create_mlp(inner_layers = hp['layers'], inner_dim = hp['d_ff'], dropout_rate = hp['dropout'], output_dim = labels_len).to(device)
    
    optim1 = torch.optim.Adam(model.model_image.parameters(), lr = hp['lr_bb'], betas=(0.9,0.98), eps=1e-6, weight_decay=hp['wd']) 
    optim2 = torch.optim.Adam(ff_head.parameters(), lr = hp['lr_ff'], weight_decay=hp['wd'])

    if checkpoint != None and 'optimizer' in checkpoint:
        checkpoint['optimizer']['param_groups'][0]['lr'] = hp['lr_bb']
        checkpoint['optimizer']['param_groups'][0]['weight_decay'] = hp['wd']
        optim1.load_state_dict(checkpoint['optimizer'])    
    
    if hp["task"] == 'classification':
        loss_func = nn.BCEWithLogitsLoss()

    if hp["task"] == 'regression':
        loss_func = nn.L1Loss() 
    
    loss = 0
    total_loss = 0

    step = 1
    val_improvement = 0 
    min_val = -1

    start_time = time.time()
    current_time = start_time
    #start training
    for epoch in range(args.epochs):

        epoch_it = iter(train_dataset)

        for it in range(len(epoch_it)):
            t = time.time()

            optim1.zero_grad() 
            optim2.zero_grad() 
            
            images, labels = next(epoch_it)
            
            embeddings = model.model_image(images).float()
            output = ff_head(embeddings)

            loss = loss_func(output, labels)

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

                it_dir = ft_eval(model, ff_head, val_dataset, labels_len, hp["task"], device = device, subset='val')
                print ("val:")
                print (it_dir)

                if hp["task"] == "classification":
                    current_val = it_dir[f'Val/{hp["val_metric"]}']
                else:
                    current_val = it_dir[f'Val/{hp["val_metric"]}']
                if args.wandb:
                    wandb.log(it_dir, step = step)

                current_predictions, it_dir = ft_eval(model, ff_head, test_dataset, labels_len, hp["task"], device = device, subset='test')
                print ("test:")
                print (it_dir)

                if hp["task"] == "classification":
                    current_test = it_dir[f'Test/{hp["test_metric"]}']
                else:
                    current_test = it_dir[f'Test/{hp["test_metric"]}']

                if args.wandb:
                    wandb.log(it_dir, step = step)

                model.train()
                ff_head.train()

                if min_val == -1: #initialize current values
                    min_val = current_val
                    best_test = current_test
                
                if (hp["val_metric"] in ["RMSE", "MSE", "MAE", "BCE"] and current_val <= min_val) or (hp["val_metric"] in ["roc_auc", "accuracy"] and current_val >= min_val):
                    min_val = current_val
                    test_in_min = current_test
                    it_dir = {'Max/min_val': min_val, 'Max/test_in_min_val': test_in_min}
                    
                    if args.save_predictions:
                        res_df = pd.DataFrame(current_predictions)
                        res_df.to_csv(f"Predictions/test-in-min/{run_name}test-in-min.csv")

                    if args.wandb:
                        wandb.log(it_dir, step = step)

                    val_improvement = 0

                else:
                    val_improvement += 1

                if (hp["test_metric"] in ["RMSE", "MSE", "MAE", "BCE"] and current_test <= best_test) or (hp["test_metric"] in ["roc_auc", "accuracy"] and current_test >= best_test):

                    best_test = current_test
                    it_dir = {'Max/best_test': best_test}

                    if args.save_predictions:
                        res_df = pd.DataFrame(current_predictions)
                        res_df.to_csv(f"Predictions/best/{run_name}best.csv")

                    if args.wandb:
                        wandb.log(it_dir, step = step)
                

        if args.epochs_cp > 0:
            if not os.path.isdir(f"{args.cp_path}/finetuning"):
                os.mkdir(f"{args.dataset_path}/{args.dataset_name}/finetuning") 

            if (epoch + 1) % cp_every == 0:
                if not os.path.isdir(f"{args.cp_path}/finetuning/{run_name}"):
                    os.mkdir(f"{args.cp_path}/finetuning/{run_name}")              
                save_checkpoint(model, optim1, None, args.cp_path, f"finetuning/{run_name}", epoch+1, step, optim2=optim2)
    
    print ("""
    
    finetuning summary:""")
    print ("-------------------")
    print (f"dataset: {dataset_name}")
    print (f"parameters: {hp}")
    print (f"best test evaluation: {best_test}")
    print (f"test evaluation in minimum validation loss: {test_in_min}")
    print ("-------------------")

def load_model(hp, args, idx, r, dataset_name, seed):

    model = Model(args.vit_model, args.no_parallel, args.device, classes = []).to(args.device)

    full_cp_name = f"{hp['cp'].split(' - ')[0]}/{hp['cp']}"
    model_path = f"{args.cp_path}/{full_cp_name}"

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    checkpoint['model'] = {key: value for key, value in checkpoint['model'].items() if 'cls' not in key and 'clip_model' not in key}

    model.load_state_dict(checkpoint['model'])
    
    config = {
    "batch_size": hp['batch'],
    "lr_ff": hp['lr_ff'],
    "lr_bb": hp['lr_bb'],
    "dropout": hp['dropout'],
    "d_ff": hp['d_ff'],
    "layers": hp['layers'],
    "wd": hp['wd'],
    "pretrain": model_path.split("/")[-1],
    "dataset": dataset_name,
    "random_seed": seed,
    "splitting": hp['splitting'],
    "val_ratio": args.val_test_ratio,
    "test_ratio": args.val_test_ratio,
    "augmentation": hp["augmentation"],
    "val_metric": hp["val_metric"],
    "test_metric": hp["test_metric"]
    }

    run_name = f"{datetime.now().strftime('%D %H-%M').replace('/', '-')}_{idx+1}_seed{seed}_{r+1}_{args.comment}"

    return model, checkpoint, config, run_name


def main(pg, args):

    if args.wandb:
        wandb.login(key = args.wandb_key)

    if args.save_predictions: 
        if not os.path.isdir(f"Predictions/test-in-min"):
            os.mkdir(f"Predictions/test-in-min") 
        if not os.path.isdir(f"Predictions/best"):
            os.mkdir(f"Predictions/best") 
    
    print ("start finetuning")

    for dataset_name in args.dataset_names: 

        for idx, hp in enumerate(list(pg)):
            
            if args.random_seeds == []:
                args.random_seeds = [0] if hp['splitting'] == 'scaffold' else [1,2,3]

            hp["task"] = dataset_tasks_dict[dataset_name][0] if args.task == None else args.task
            hp["val_metric"] = dataset_tasks_dict[dataset_name][1] if args.val_metric == None else args.val_metric
            hp["test_metric"] = dataset_tasks_dict[dataset_name][2] if args.test_metric == None else args.test_metric

            print (f"hyperparmeter set no. {idx + 1} out of {len(pg)}")

            for seed in args.random_seeds:
                for r in range (args.repeats):
                    
                    #load dataset
                    train_dataset, val_dataset, test_dataset, labels_len = Load_finetuning_dataset (args.dataset_path, dataset_name, args.dataset_type, splitter = hp['splitting'], 
                                                                                                    val_size = args.val_test_ratio, test_size = args.val_test_ratio, batch_size = hp['batch'],
                                                                                                    seed = seed, aug = hp['augmentation'], device = args.device)   
                    print ("datasets were sucssesfully loaded")

                    #load model
                    model, checkpoint, config, run_name = load_model(hp, args, idx, r, dataset_name, seed)
                    print ("model was sucssesfully loaded")

                    if args.wandb:
                        wandb.init(project=args.wandb_project, name = run_name, config = config)
                    
                    train_model (model, train_dataset, val_dataset, test_dataset, hp, labels_len, dataset_name, run_name, checkpoint = checkpoint, device = args.device)
                    
                    if args.wandb:
                        wandb.finish()
                    
                    torch.cuda.empty_cache()

def gen_pg (args):
    
    if args.cp_list == None:
        args.cp_list = [args.cp_name]

    else:
        with open(args.cp_list, "r") as f:
            args.cp_list = [i[:-1] if i[-1] == "\n" else i for i in list(f)]

    for cp in args.cp_list:

        full_cp_name = f"{cp.split(' - ')[0]}/{cp}"
        
        if not os.path.isfile(f"{args.cp_path}/{full_cp_name}"):

            if cp == "MoleCLIP - Primary.pth" and args.download:
                if not os.path.isdir(f"{args.cp_path}/MoleCLIP"):
                    os.mkdir(f"{args.cp_path}/MoleCLIP")

                download_MoleCLIP_weights (f"{args.cp_path}/{full_cp_name}")

            else:
                raise(Exception(f"{cp} file wasn't found")) 
    
    param_grid = {'lr_bb':args.lr_bb,
                  'lr_ff':args.lr_ff,
                  'dropout':args.dropout,
                  'batch':args.batch,
                  'd_ff':args.d_ff,
                  'layers':args.layers,
                  'wd':args.wd,
                  'splitting':args.splitting,
                  'cp': args.cp_list,
                  'augmentation': args.augmentation,
                  }

    return ParameterGrid(param_grid)

def parse_args():

    parser = argparse.ArgumentParser(description='Parameters for pretraining ImageMol')

    parser.add_argument('-device', default="cuda", type=str, choices=["cuda", "cpu"], help='Device to use: "cuda" or "cpu".')
    parser.add_argument('-no_parallel', action='store_false', help='Disable multi-GPU training. Set to use only a single GPU.')

    parser.add_argument('-lr_bb', default=[5e-6], nargs='+', type=float, help='Initial learning rate for the model backbone.')
    parser.add_argument('-lr_ff', default=[0.0004], nargs='+', type=float, help='Initial learning rate for the feed-forward layers.')
    parser.add_argument('-wd', default=[0.00001], nargs='+', type=float, help='Weight decay coefficient(L2 regularization).')
    parser.add_argument('-epochs', default=180, type=int, help='Total number of epochs.')
    parser.add_argument('-dropout', default=[0], nargs='+', type=float, help='Dropout rate for the layers.')
    parser.add_argument('-batch', default=[64], nargs='+', type=int, help='Batch size (default: 64).')
    parser.add_argument('-d_ff', default=[512], nargs='+', type=int, help='Dimension of the feed-forward middle layers.')
    parser.add_argument('-layers', default=[3], nargs='+', type=int, help='Number of feed-forward middle layers.')
    parser.add_argument('-augmentation', default=["default"], nargs='+', type=str, choices=["none", "default", "intense"], help='Level of data augmentation: "none", "default", or "intense".')

    parser.add_argument('-epochs_cp', default=0, type=int, help='Number of epochs between checkpoints.')
    parser.add_argument('-log_every', default=5, type=int, help='Iteration interval for logging.')
    parser.add_argument('-eval_every', default=1, type=int, help='Number of epochs between evaluations.')
    parser.add_argument('-save_predictions', action='store_true', help='Whether to save test predictions.')
    parser.add_argument('-random_seeds', default=[1], nargs='+', type=int, help='Random seeds for reproducibility in data splitting.')
    parser.add_argument('-repeats', default=1, type=int, help='Number of training repeats per hyperparameter set.')

    parser.add_argument('-dataset_path', default="../Datasets", type=str, help='Path to the directory containing the dataset.')
    parser.add_argument('-dataset_names', default=["bace"], nargs='+', type=str, help='Names of the dataset to be used.')
    parser.add_argument('-dataset_type', default="images", type=str, choices=["images", "smiles"], help='Format of the dataset: "images" or "SMILES".')
    parser.add_argument('-splitting', default=["random"], nargs='+', choices=["random", "scaffold", "random_scaffold", "none"], help='Data splitting method.')

    parser.add_argument('-task', default=None, choices=["regression", "classification"], help='Task type: "regression" or "classification".')
    parser.add_argument('-test_metric', default=None, choices=["MAE", "RMSE", "MSE", "BCE", "roc_auc", "accuracy"], help='Metric for evaluating test performance.')
    parser.add_argument('-val_metric', default=None, choices=["MAE", "RMSE", "MSE", "BCE", "roc_auc", "accuracy"], help='Metric for evaluating validation performance.')
    parser.add_argument('-val_test_ratio', default=0.1, type=float, help='Ratio of validation/test data in splitting.')

    parser.add_argument('-cp_path', default='../Checkpoints', type=str, help='Directory for storing checkpoints.')
    parser.add_argument('-cp_list', default=None, type=str, help='Path to a .txt file listing checkpoints to use.')
    parser.add_argument('-cp_name', default='MoleCLIP - Primary.pth', type=str, help='Name of the checkpoint to use (.pth extension)')
    parser.add_argument('-vit_model', type=str, default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"], help='Variant of the Vision Transformer model.')

    parser.add_argument('-download', action='store_true', help='Whether to use Weights & Biases for logging.')
    parser.add_argument('-wandb', action='store_true', help='Whether to use Weights & Biases for logging.')
    parser.add_argument('-wandb_key', default="", type=str, help='Weights & Biases API key.')
    parser.add_argument('-wandb_project', default="", type=str, help='Name of the wandb project for logging runs.')
    parser.add_argument('-comment', default="", type=str, help='Comment to add to the run name.')

    parser.set_defaults(wandb=False, download=False, resume=False, no_parallel=True, save_predictions=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    parmeters_grid = gen_pg(args)
    main(parmeters_grid, args)

