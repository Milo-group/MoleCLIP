import sys
sys.path.append('..')

import os
import concurrent.futures

import torch
from torch.utils.data import DataLoader, Subset, Dataset 

import pandas as pd
import numpy as np
import csv
from PIL import Image
import cv2

from .splitters import random_scaffold_split, scaffold_split, random_split
from .Img_generation import mol_to_img
from .Augmentation import augmentation
  

class DatasetImages(Dataset):

    def __init__(self, datasets_path, smiles, labels = None, dataset_type = 'images', uselabels = True, aug = "none", device="cuda"):

        self.device = device
        self.datasets_path = datasets_path

        self.smiles = smiles
        self.labels = labels
        self.uselabels = uselabels
        self.dataset_type = dataset_type

        self.transform = augmentation (aug = aug)

        if dataset_type == "images":

            try:
                self.len_dataset = len(os.listdir(datasets_path + "/imgs"))
            except:
                raise(Exception("dataset folder is not exist, or no imgs folder exists in folder"))
            
            if self.len_dataset != len(self.smiles):
                raise(Exception(f"Number of images ({self.len_dataset} not equal to number of smiles {self.data_df.shape[0]})"))
        
        else:
            self.len_dataset = len(self.smiles)

        if dataset_type == "fp":
            from rdkit import Chem
            from rdkit.Chem import AllChem

            self.fps = []
            for smi in self.smiles:
                mol = Chem.MolFromSmiles(smi)
                
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                fingerprint = torch.tensor(np.array(fingerprint, dtype=np.float32))

                self.fps.append(fingerprint)

    def get_image(self, img_name):
        
        if self.dataset_type == 'smiles':
            img = mol_to_img(img_name, False)
        if self.dataset_type == 'images':
            img = cv2.imread(f'{self.datasets_path}/imgs/{img_name + 1}.png')

        return self.transform(image = img)['image']

    def __getitem__(self, idx):
        
        if self.dataset_type == "images":
            if self.uselabels:
                output = [self.get_image(idx).float().to(self.device), torch.tensor([float(i) for i in self.labels[idx][1]]).to(self.device)]
            else:
                output = [self.get_image(idx).float().to(self.device)]

        if self.dataset_type == "smiles":
            if self.uselabels:
                output = [self.get_image(self.smiles[idx]).float().to(self.device), torch.tensor([float(i) for i in self.labels[idx][1]]).to(self.device)]
            else:
                output = [self.get_image(self.smiles[idx]).float().to(self.device)]

        if self.dataset_type == "fp":
            if self.uselabels:
                output = [self.fps[idx].float().to(self.device), torch.tensor([float(i) for i in self.labels[idx][1]]).to(self.device)]
            else:
                output = [self.fps[idx].float().to(self.device)]

        return output

    def __len__(self):
        return self.len_dataset
    

def Load_finetuning_dataset (datasets_path, dataset_name, dataset_type = "images", val_size = 0.1, test_size = 0.1, batch_size = 64, 
                             seed = 1, splitter = "scaffold", uselabels = True, shuffle = True, aug = "none", device = "cuda"):
    
    if not os.path.exists(datasets_path):
        raise(Exception("Datasets directory not exist"))

    full_path = f"{datasets_path}/{dataset_name}"

    if os.path.isfile(full_path + f"/{dataset_name}.csv"):

        with open(full_path + f"/{dataset_name}.csv", "r") as file:
            data = list(csv.reader(file, delimiter=","))

        indices = list(range(len(data[1:])))
        smiles = [i[1] for i in data[1:]]

        if uselabels:
            labels = [[i[1], [float(n) for n in i[2].split(" ") if n != ""]] for i in data[1:]]
            labels_len = len(labels[0][1])
        
        else:
            labels = None
            labels_len = 0

    else:
        raise(Exception("Dataset smiles file is not exist (should be csv)"))
    
    full_dataset = DatasetImages(full_path, smiles, labels, dataset_type, uselabels, aug = aug, device = device)
    full_dataset_no_aug = DatasetImages(full_path, smiles, labels, dataset_type, uselabels, aug = "none", device = device)
    
    if splitter == "random":
        train_indices, val_indices, test_indices = random_split (indices, val_size, test_size, seed)
    
    elif splitter == "scaffold":
        train_indices, val_indices, test_indices = scaffold_split (smiles, frac_train=1-val_size-test_size, frac_valid=val_size, frac_test=test_size)

    elif splitter == "random_scaffold":
        train_indices, val_indices, test_indices = random_scaffold_split (smiles, frac_train=1-val_size-test_size, frac_valid=val_size, frac_test=test_size, seed=seed)

    elif splitter == "none":
        train_indices, val_indices, test_indices = (indices, [], [])
    
    else:
        raise(Exception("splitter not exist"))
    
    train_dataset = DataLoader(Subset(full_dataset, train_indices), batch_size = batch_size, shuffle = shuffle)
    val_dataset = DataLoader(Subset(full_dataset_no_aug, val_indices), batch_size = batch_size) if len(val_indices) > 0 else None
    test_dataset = DataLoader(Subset(full_dataset_no_aug, test_indices), batch_size = batch_size) if len(test_indices) > 0 else None

    return train_dataset, val_dataset, test_dataset, labels_len
    

    

class Dataset_contrastive(Dataset):
    def __init__(self, datasets_path, dataset_name, dataset_type = 'images', batch_size=64, img_size = 224, mix_factor = 8, aug = "none", device="cuda"):
        
        self.device = device
        self.datasets_path = datasets_path
        self.batch_size = batch_size
        self.dataset_type = dataset_type

        self.transform = augmentation (aug = aug)
        
        if dataset_type == "images":
            try:
                len_dataset1 = len(os.listdir(datasets_path + "/" + "aug_1"))
                len_dataset2 = len(os.listdir(datasets_path + "/" + "aug_2"))
            except:
                raise(Exception("aug_1 and/or aug_2 folders are not exist"))
            
            if len_dataset1 != len_dataset2:
                raise(Exception("aug_1 and aug_2 lengths should be equal"))
        
        csv_path = datasets_path + "/" + dataset_name + ".csv"
        df = pd.read_csv (csv_path)
        self.classes = [int(c.split("_")[-1]) for c in list(df.columns.values)[2:]]
        ordered_df = df.sort_values(by=[f'classes_{max(self.classes)}']) 

        if batch_size % mix_factor != 0 or mix_factor < 1:
            raise(Exception(f"mix_factor ({mix_factor}) must be positive and a divisor of the batch size ({batch_size})"))

        sub_batch_size = int(batch_size / mix_factor) 
        self.batches = [ordered_df[idx:idx+sub_batch_size] for idx in range(0, ordered_df.shape[0], sub_batch_size)]
        
        if dataset_type == "images":
            if len_dataset1 != ordered_df.shape[0]:
                raise(Exception(f"""The number of molecular images in the dataset folder ({len_dataset1}) ins't equal to the 
                                number of molecules in the csv file ({ordered_df.shape[0]})"""))
        

    def load_image_1(self, img_name):
        if self.dataset_type == "smiles":
            return self.transform(image = mol_to_img(img_name, False))['image']
        if self.dataset_type == "images":
            return self.transform(image = cv2.imread(f'{self.datasets_path}/aug_1/{img_name}.png'))['image']
    
    def load_image_2(self, img_name):
        if self.dataset_type == "smiles":
            return self.transform(image = mol_to_img(img_name, True))['image']
        if self.dataset_type == "images":
            return self.transform(image = cv2.imread(f'{self.datasets_path}/aug_2/{img_name}.png'))['image']

    def get_image(self, batch_index):

        if self.dataset_type == "smiles":
            batch = self.batches[batch_index]['smiles'].tolist()
        if self.dataset_type == "images":
            batch = self.batches[batch_index]['indices'].tolist()
         
        with concurrent.futures.ThreadPoolExecutor() as executor:
            imgs1 = list(executor.map(self.load_image_1, batch))
            imgs2 = list(executor.map(self.load_image_2, batch))
        
        img1_batch = torch.stack(imgs1, dim = 0).to(self.device)
        img2_batch = torch.stack(imgs2, dim = 0).to(self.device)

        outputs = [img1_batch, img2_batch]

        for c in self.classes:
            outputs.append(torch.tensor(self.batches[batch_index][f'classes_{c}'].values).to(self.device))

        return outputs

    def __getitem__(self, batch_index):     
        return self.get_image(batch_index)

    def __len__(self):
        return len(self.batches)
 
    def classes(self):
        return self.classes
    
def collate_data(batch): 
    return (torch.cat([i[n] for i in batch], dim = 0) for n in range(len(batch[0])))


def Load_contrastive_dataset (datasets_path, dataset_name, dataset_type = 'images', batch_size = 64, val_size = 0.01, mix_factor = 8, shuffle = True, seed = 42, aug = "none", device = "cuda"):

    if not os.path.exists(datasets_path):
        raise(Exception("Datasets directory not exist"))

    if datasets_path[-1] == "/":
        full_path = datasets_path + dataset_name
    else:
        full_path = datasets_path + "/" + dataset_name
   
    full_dataset = Dataset_contrastive (full_path, dataset_name, dataset_type = dataset_type, batch_size = batch_size, mix_factor = mix_factor, aug = aug, device = device)
    train_indices, val_indices, _ = random_split(list(range(len(full_dataset))), 0, val_size, seed)

    train_dataset = DataLoader(Subset(full_dataset, train_indices), collate_fn=collate_data, batch_size = mix_factor, shuffle = shuffle)
    val_dataset = DataLoader(Subset(full_dataset, val_indices), collate_fn=collate_data, batch_size = mix_factor)
    
    return train_dataset, val_dataset, full_dataset.classes

class DTADataset(Dataset):
    def __init__(self, dataset, mode = "train", device="cuda"):
        self.device = device

        data, _ = torch.load(f"../Datasets/Finetuning_DTA/data/processed/{dataset}_{mode}.pt")

        self.y = data.y
        self.smiles = data.smiles
        self.targets = data.target

        self.imgs_dict = torch.load("../Datasets/Finetuning_DTA/data/processed/smile_image.pt")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        sample = (
                self.y[idx].to(self.device),
                self.smiles[idx],
                self.imgs_dict[self.smiles[idx]].float().to(self.device),
                torch.tensor(self.targets[idx], dtype=torch.long).to(self.device)
            )

        return sample
