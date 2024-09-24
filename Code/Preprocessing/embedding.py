import os
import sys
sys.path.append('..')
import argparse
import csv
from tqdm import tqdm

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score


from Data.Dataloaders import Load_finetuning_dataset
from Model.Load_model import Model, Classification_head



def create_scatter_plot(data, labels, f_name, args):

    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto')
    embeddings_2d = tsne.fit_transform(data)

    plt.set_cmap('Paired')
    font = {'size': 12}

    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(7,7))
    
    if labels == None:
        labels = [1] * embeddings_2d.shape[0]
        
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, s=32)

    ax.set_xlabel("TSNE-1", fontsize=16)
    ax.set_ylabel("TSNE-2", fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f"{args.dataset_path}/{args.dataset_name}/embeddings/scatters/{f_name[:-4]}.png")
    plt.close()


def calc_db(data, labels, f_name, args):

    db = davies_bouldin_score(data, labels)
    print (f"{f_name[:-4]} : {db}")

    return db


def load_model(cp, model_type, args):

    model = Model(args.vit_model, args.no_parallel, args.device).to(args.device)


    if model_type == 'base':

        full_cp_name = cp.split(" - ")[0] + "/" + cp
        cp_full_path = args.cp_path + full_cp_name if args.cp_path[-1] == "/" else args.cp_path + "/" + full_cp_name + ".pth"
        checkpoint = torch.load(cp_full_path, map_location=torch.device('cpu'))

        checkpoint['model'] = {param:checkpoint['model'][param] for param in checkpoint['model'] if ("clip_model" not in param and "cls_" not in param)}
        
        model.load_state_dict(checkpoint['model'])

    
    elif model_type == 'ImageMol':

        model = torchvision.models.resnet18(pretrained=False).to(torch.device("cuda"))
        
        try:
            full_cp_name = f"{args.cp_name.split(' - ')[0]}/{args.cp_name}"
            model_path = f"{args.cp_path}/{full_cp_name}.pth"
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
               
        except:
            try:
                model_path = f"{args.cp_path}/{args.cp_name}.pth"  
                checkpoint = torch.load(model_path)['state_dict']
            except:
                model_path = f"{args.cp_path}/{args.cp_name}.pth.tar"  
                checkpoint = torch.load(model_path)['state_dict']

        ckp_keys = list(checkpoint)
        cur_keys = list(model.state_dict())
        model_sd = model.state_dict()
        ckp_keys = ckp_keys[:120]
        cur_keys = cur_keys[:120]

        for ckp_key, cur_key in zip(ckp_keys, cur_keys):
            model_sd[cur_key] = checkpoint[ckp_key]

        model.load_state_dict(model_sd)
        model.fc = nn.Identity()

    return model


def embed(args):
    
    if not os.path.isdir(f"{args.dataset_path}/{args.dataset_name}/embeddings"):
        os.mkdir(f"{args.dataset_path}/{args.dataset_name}/embeddings") 

    for cp in args.cp_list:
        model_type = "ImageMol" if "ImageMol" in args.cp_name else "base"

        full_data, _, _, _ = Load_finetuning_dataset (args.dataset_path, args.dataset_name, args.dataset_type, splitter = "none", uselabels=False, \
                                                      batch_size = args.batch, model_type = model_type, shuffle = False, aug = False)                    
        model = load_model(cp, model_type, args)

        print ("model loaded")

        model.eval()

        all_images = torch.zeros([1, 3, 224, 224]).to(args.device)
        all_embeddings = torch.zeros([1, 512]).to(args.device)
        all_results = torch.tensor([0]).to(args.device)

        img_idx = 0

        with torch.no_grad():

            for data in [iter(full_data)]:
                for it in tqdm(range(len(data))):
                    
                    batch = next(data)
                    with torch.no_grad():
                        images = batch[0]
                        embeddings = model.model_image(images).float()   
                        all_embeddings = torch.cat([all_embeddings, embeddings], dim = 0)

        emb_np = all_embeddings[1:].cpu().detach().numpy()
        np.savetxt(f'{args.dataset_path}/{args.dataset_name}/embeddings/{cp}_emb.txt', emb_np, fmt='%.4e')



def parse_args():
    parser = argparse.ArgumentParser(description='parameters for embedding')

    parser.add_argument('-device', type=str, default="cuda", choices=["cuda", "cpu"], help='cuda or cpu')
    parser.add_argument('-no_parallel', action='store_false', help='weather to parallize on multiple gpus')

    parser.add_argument('-batch', default=64, type=int, help='batch size (default: 256)')
    
    parser.add_argument('-cp_list', type=str, default=None, help='A .txt list of checkpoint names to embed')
    parser.add_argument('-cp_path', default="../../checkpoints", help='checkpoints directory path')
    parser.add_argument('-cp_name', default=None, help='load checkpoint weights')
    parser.add_argument('-dataset_path', type=str, default="../../Datasets", help='dataset path')
    parser.add_argument('-dataset_name', type=str, default="Oscar_DHBDs", help='dataset name')
    parser.add_argument('-dataset_type', type=str, default="smiles", choices=["images", "smiles"], help='weather the dataset storaged as images or SMILES')

    parser.add_argument('-classes_list_path', type=str, default=None, help='A .txt list of classes in respective order to embeddings. \
                                                                      Needed for scattering or for db_index calculation')

    parser.add_argument('-function', default = "embed", choices=["embed", "db_index", "scatter"], help='choose function to execute')

    parser.add_argument('-vit_model', type=str, default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"], help='clip base-model')
    
    parser.set_defaults(no_parallel = True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.cp_list != None:
        try:
            with open(args.cp_list, "r") as f:
                args.cp_list = [i[:-1] if i[-1] == "\n" else i for i in list(f)]

        except:
            raise(Exception("checkpoint list file not exist"))

    elif args.cp_name != None:
        args.cp_list = [args.cp_name]

    else:
        raise(Exception("no checkpoint list file or checkpoint file exist"))

    if args.function == 'embed':
        embed(args)
    
    else:     
        for cp in args.cp_list:
            embeddings_path = f'{args.dataset_path}/{args.dataset_name}/embeddings/{cp}_emb.txt'
            try:
                with open(embeddings_path, 'r') as f:       
                    data = np.array([[float(x) for x in row.split(" ")] for row in list(f)])
            
            except:
                raise(Exception(f"embeddings file is not exist or not in the right format"))

            try:
                with open(args.classes_list_path, 'r') as f:
                    labels = [int(i[:-1]) if i[-1] == "\n" else int(i) for i in list(f)]
            except:
                labels = None
                print ("Labels file wasn't defined or incorrect")

        
        if args.function == 'db_index':
            if labels == None:
                raise(Exception(f"can't calculate db_index without labels"))

            calc_db(data, labels, args.file_name, args)

        if args.function == 'scatter':
            if not os.path.isdir(f"{args.dataset_path}/{args.dataset_name}/embeddings/scatters"):
                os.mkdir(f"{args.dataset_path}/{args.dataset_name}/embeddings/scatters") 

            create_scatter_plot(data, labels, args.cp_name, args)





