import sys
sys.path.append('..')
import argparse
from datetime import datetime
import os

from Data.Dataloaders import Load_contrastive_dataset
from Model.Load_model import Model
from Evaluations.pretraining_eval import pt_eval, features_to_logits

import torch
import torchvision
from torch import nn
from torch.optim import lr_scheduler

from itertools import islice
import random
import time
from tqdm import tqdm

import wandb

def save_checkpoint(model, optim, cp_path, run_name, epoch, step, temp = False, scheduler=None, optim2=None):

    if not os.path.isdir(f"{cp_path}/{run_name}"):
        os.mkdir(f"{cp_path}/{run_name}")    
        
    save_dict = {'model': model.state_dict(), 'optimizer': optim.state_dict(),
                    'epoch':epoch, 'step':step, 'optimizer2': optim2.state_dict()} 

    if scheduler != None:
        save_dict.update({'scheduler': scheduler.state_dict()})  

    if optim2 != None:
        save_dict.update({'optimizer2': optim2.state_dict()})  
    
    if temp: 
        torch.save(save_dict, f'{cp_path}/{run_name}/{run_name} - temp.pth')
    
    else:
        torch.save(save_dict, f'{cp_path}/{run_name}/{run_name} - {epoch}.pth')
        
    
def bulid_scheduler(optim, lr0, total_iterations, scheduler_type = 'WarmCosine', warmsteps = 200):

    if scheduler_type == 'WarmCosine':
        scheduler1 = lr_scheduler.LinearLR(optim, start_factor=0.05, total_iters = warmsteps)
        scheduler2 = lr_scheduler.CosineAnnealingLR(optim, T_max = total_iterations - warmsteps, eta_min=1e-7) 
        scheduler = lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

    elif scheduler_type == 'Cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max = total_iterations, eta_min=1e-7) 

    elif scheduler_type == 'Constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR (optim)
    
    return scheduler      
      
def train_model(model, train_dataset, val_dataset, args, run_name = '', 
                checkpoint = None, classes = [300, 3000], device='cuda'):
    
    optim1 = torch.optim.Adam(model.model_image.parameters(), lr=args.lr0, betas=(0.9,0.98), eps=1e-6, weight_decay=args.wd) 
    optim2 = torch.optim.Adam([
    {'params': model.cls_1.parameters(), 'lr': args.lr_class},
    {'params': model.cls_2.parameters(), 'lr': args.lr_class}], betas=(0.9,0.98)) 

    scheduler = bulid_scheduler (optim1, args.lr0, args.epochs * len(train_dataset), scheduler_type = args.scheduler_type, warmsteps = args.warmsteps)  
   
    step = 1
    epoch_0 = 0

    if checkpoint != None:

        optim1.load_state_dict(checkpoint['optimizer'])        
        scheduler.load_state_dict(checkpoint['scheduler'])

        if args.resume:
            if 'optimizer2' in checkpoint:
                optim2.load_state_dict(checkpoint['optimizer2'])
            
            if 'step' in checkpoint:
                step = checkpoint['step']

            if 'epoch' in checkpoint:
                epoch_0 = int(float(checkpoint['epoch']))

       
    loss_function = nn.CrossEntropyLoss()

    model.train()
    
    start_time = time.time()
    temp_time = time.time()
    
    total_singles_loss = 0
    total_cls1_loss = 0
    total_cls2_loss = 0
    total_loss = 0
    
    labels_template = torch.arange(args.batch).to(device)

    for epoch in range(epoch_0, args.epochs):

        if args.dataset_type == "smiles":
            epoch_it = iter(train_dataset)
            order = list(range(len(epoch_it)))
            random.shuffle (order)

        if args.dataset_type == "images":
            epoch_it = iter(train_dataset)

        for it in range(len(epoch_it)):

            optim1.zero_grad() 
            optim2.zero_grad() 
            
            image1, image2, cls1_labels, cls2_labels = next(epoch_it)

            image1_features = model.model_image(image1).float()
            image2_features = model.model_image(image2).float()

            cls_1_preds = model.cls_1(image1_features)
            cls_2_preds = model.cls_2(image1_features)

            labels = labels_template[:image1.shape[0]]

            logits_per_image1, logits_per_image2 = features_to_logits (model, image1_features, image2_features, args.temperature)

            loss_singles = (loss_function(logits_per_image1, labels) + loss_function(logits_per_image2, labels))/2 
            
            cls1_loss = loss_function(cls_1_preds, cls1_labels) 
            cls2_loss = loss_function(cls_2_preds, cls2_labels)
            loss = loss_singles + cls1_loss/args.cls_1_l + cls2_loss/args.cls_2_l  

            loss.backward()
            optim1.step()
            optim2.step()

            total_singles_loss += loss_singles
            total_cls1_loss += cls1_loss
            total_cls2_loss += cls2_loss
            total_loss += loss 
          
            if step % args.log_every == 0:
                it_dir = {'specs/time (hours)': (time.time() - start_time) / 3600, 'specs/iteration': it + 1, 'specs/time per iteration (sec)': (time.time() - temp_time) / args.log_every,
                           'specs/lr': scheduler.get_last_lr()[0], 'specs/lr_clss':args.lr_class, 'Train/avg_loss': total_loss / args.log_every, 'Train/avg_singles_loss': total_singles_loss / args.log_every, 
                           f'Train/total_cls{classes[0]}_loss': total_cls1_loss / args.log_every, f'Train/total_cls{classes[1]}_loss': total_cls2_loss / args.log_every,
                           'specs/epoch':"{:.2f}".format(step / len(epoch_it))}

                if args.wandb:
                    wandb.log(it_dir, step = step)
                
                total_singles_loss = 0
                total_cls1_loss = 0
                total_cls2_loss = 0
                total_loss = 0

                print (it_dir)
                temp_time = time.time()

            if step % args.eval_every == 0:
                print ("start evaluation:")
                with torch.no_grad():
                    it_dir = pt_eval(model, val_dataset, classes, args = args, eval_part = 1, device = device)
                
                it_dir.update({'specs/eval time': time.time() - temp_time})

                if args.wandb:
                    wandb.log(it_dir, step = step)

                model.train()
                print (it_dir)
                temp_time = time.time()
                            
            scheduler.step()
            step += 1

            if args.steps_cp > 0:
                if step % args.steps_cp == 0:
                    save_checkpoint(model, optim1, args.cp_path, run_name, "{:.2f}".format(step / len(epoch_it)), step, scheduler=scheduler, optim2=optim2)

            if args.temp_cp > 0:
                if step % args.temp_cp == 0:
                    save_checkpoint(model, optim1, args.cp_path, run_name, "{:.2f}".format(step / len(epoch_it)), step, temp = True, scheduler=scheduler, optim2=optim2)
        
        if args.epochs_cp > 0:
            if step % args.epochs_cp == 0:
                save_checkpoint(model, optim1, args.cp_path, run_name, epoch+1, step, scheduler=scheduler, optim2=optim2)


def run_config(args, train_dataset, val_dataset, classes):
    return {
    "batch_size": args.batch,
    "train size": len(train_dataset),
    "eval size": len(val_dataset),
    "dataset": args.dataset_name,
    "singles_temp": args.temperature,
    "wd":args.wd,
    "classes":"+".join([str(i) for i in classes]),
    "augmentation":args.augmentation
    }

def main(args):

    train_dataset, val_dataset, classes = Load_contrastive_dataset(args.datasets_path, args.dataset_name, args.dataset_type, batch_size = args.batch, 
                                                                   val_size = args.val_ratio, mix_factor = args.mix_factor, aug = args.augmentation, 
                                                                   device = args.device)
    print ("datasets were sucssesfully loaded")

    model = Model(args.vit_model, args.no_parallel, args.device, classes).to(args.device)
    print ("model was sucssesfully loaded")

    config = run_config (args, train_dataset, val_dataset, classes)

    experiment_comment = args.dataset_name + f"_{args.dataset_type}_" + 'pretrain_' + args.comment if args.comment!="" \
                    else args.dataset_name + f"_{args.dataset_type}_" + 'pretrain'

    run_name = datetime.now().strftime("%D %H-%M").replace("/", "-") + "-" + experiment_comment

    if args.cp_name != None:
        if os.path.isdir(args.cp_path):
            full_cp_name = args.cp_name.split(" - ")[0] + "/" + args.cp_name
            cp_full_path = args.cp_path + full_cp_name if args.cp_path[-1] == "/" else args.cp_path + "/" + full_cp_name + ".pth"
            checkpoint = torch.load(cp_full_path, map_location=torch.device('cpu'))

            checkpoint['model'] = {param:checkpoint['model'][param] for param in checkpoint['model'] if "clip_model" not in param}

            model.load_state_dict(checkpoint['model'])
            
            config.update({"initial_weights":args.cp_name})

        else:
            raise(Exception("checkpoint directory not exist"))
    
    else:
        checkpoint = None

    if args.wandb:
        wandb.login(key = args.wandb_key)
        run = wandb.init(project=args.wandb_project, name = run_name, config = config)
    
    train_model (model, train_dataset, val_dataset, args, run_name, checkpoint, classes, device = args.device)

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters and settings for pretraining the ImageMol model.')

    parser.add_argument('-device', type=str, default="cuda", choices=["cuda", "cpu"], help='Device to use: "cuda" or "cpu".')
    parser.add_argument('-no_parallel', action='store_false', help='Disable multi-GPU training. Set to use only a single GPU.')

    parser.add_argument('-lr0', default=5e-6, type=float, help='Initial learning rate for the model backbone.')
    parser.add_argument('-lr_class', default=0.01, type=float, help='Initial learning rate for the classification layer.')
    parser.add_argument('-wd', default=0.1, type=float, help='Weight decay coefficient(L2 regularization).')
    parser.add_argument('-temperature', default=15, type=int, help='Temperature parameter contrastive learning.')

    parser.add_argument('-cls_1_l', default=1, type=int, help='scaling factor for class 1 loss in classification.')
    parser.add_argument('-cls_2_l', default=1, type=int, help='scaling factor for class 2 loss in classification.')
    parser.add_argument('-scheduler_type', default='Constant', type=str, help='Type of learning rate scheduler to use during training.')
    parser.add_argument('-warmsteps', default=1000, type=int, help='Number of iterations for learning rate warmup at the start of training.')

    parser.add_argument('-epochs', type=int, default=10, help='Total number of epochs.')
    parser.add_argument('-batch', default=256, type=int, help='Batch size (default: 256).')
    parser.add_argument('-steps_cp', type=int, default=0, help='Number of iterations between checkpoints.')
    parser.add_argument('-epochs_cp', type=int, default=1, help='Number of epochs between checkpoints.')
    parser.add_argument('-temp_cp', type=int, default=100, help='Save a model checkpoint every specified number of iterations.')
    parser.add_argument('-log_every', type=int, default=150, help='Iteration interval for logging.')
    parser.add_argument('-eval_every', type=int, default=750, help='Number of epochs between evaluations.')
  
    parser.add_argument('-datasets_path', type=str, default="../Datasets/Pretraining", help='Path to the directory containing the dataset.')
    parser.add_argument('-dataset_name', type=str, default="chembl_25", help='Name of the dataset to be used.')
    parser.add_argument('-dataset_type', type=str, default="images", choices=["images", "smiles"], help='Format of the dataset: "images" or "SMILES".')

    parser.add_argument('-val_ratio', type=float, default=0.001, help='Ratio of validation/test data in splitting.')
    parser.add_argument('-mix_factor', type=int, default=8, help='Number of sub-batches each mini-batch is divided into. Must divide the batch size.')
    parser.add_argument('-augmentation', type=str, default="default", choices=["none", "default", "intense"], help='Level of data augmentation: "none", "default", or "intense".')
 
    parser.add_argument('-resume', action='store_true', help='Resume training from the latest checkpoint if available.')
    parser.add_argument('-cp_path', default='../Checkpoints', help='Directory for storing checkpoints.')
    parser.add_argument('-cp_name', default=None, help='Name of the checkpoint to use.')
    parser.add_argument('-vit_model', type=str, default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"], help='Variant of the Vision Transformer model')
 
    parser.add_argument('-wandb', action='store_true', help='EWhether to use Weights & Biases for logging.')
    parser.add_argument('-wandb_key', default="", type=str, help='Weights & Biases API key.')
    parser.add_argument('-wandb_project', type=str, default="", help='Name of the wandb project for logging runs.')
    parser.add_argument('-comment', default="", type=str, help='Comment to add to the run name.')

    parser.set_defaults(wandb=False, resume=False, no_parallel=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
