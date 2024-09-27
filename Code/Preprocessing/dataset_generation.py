import sys
sys.path.append('..')

import random
import argparse
import csv
import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from cairosvg import svg2png
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger 

from rdkit.Chem import rdmolops
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from clustering import train, inference
from download import *
from Data.Img_generation import mol_to_img
RDLogger.DisableLog('rdApp.*') 




fontlist = ['DejaVuSerif-BoldItalic.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSansMono-BoldOblique.ttf', 'DejaVuSans.ttf', 'cmmi10.ttf', 
                'cmtt10.ttf', 'STIXGeneral.ttf', 'DejaVuSerif-Italic.ttf', 'STIXGeneralItalic.ttf', 'cmb10.ttf', 'DejaVuSansMono-Bold.ttf',
                'DejaVuSansMono-Oblique.ttf', 'cmss10.ttf', 'cmr10.ttf', 'DejaVuSerif.ttf', 'STIXGeneralBol.ttf', 'STIXGeneralBolIta.ttf', 
                'DejaVuSans-Bold.ttf', 'DejaVuSans-BoldOblique.ttf', 'DejaVuSans-Oblique.ttf', 'DejaVuSansMono.ttf']


def calculate_2d_descriptors(mol):

    descriptor_values = ""
    for descriptor, calculator in Descriptors.descList:

        value = calculator(mol)
        descriptor_values += str(value) + " "
        
    return descriptor_values


def gen_fingerprints(smi, radius = 2, bits = 1024, fp_type = "rdkit"):
    
    mol = Chem.MolFromSmiles(smi[1])  
    
    if fp_type == "rdkit":
        fingerprint = rdmolops.RDKFingerprint(mol, fpSize=2048, minPath=1, maxPath=7) 

    elif fp_type == "maccs":
        fingerprint = MACCSkeys.GenMACCSKeys(mol) 

    elif "morgan" in fp_type:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits)
    
    elif fp_type == "descriptors":
        desc = calculate_2d_descriptors(mol)
        fingerprints_file.write(desc + "\n")
        return

    fp_str = fingerprint.ToBitString()
    fingerprints_file.write(fp_str + "\n")


def gen_imgs(smi):

    cp_path = f"{args.datasets_path}/{args.dataset_name}"

    try:

        mol = Chem.MolFromSmiles(smi[1]) 

        if args.dataset_type == "pretraining":
            mol_to_img(mol, rand = False, filename = f"{cp_path}/aug_1/{smi[0]+1}.png")
            mol_to_img(mol, rand = True, filename = f"{cp_path}/aug_2/{smi[0]+1}.png")

        if args.dataset_type == "finetuning":
            mol_to_img(mol, rand = False, r_replace = args.r_replace, filename = f"{cp_path}/imgs/{smi[0]+1}.png")

    except:     

        errors_file = open(f"{cp_path}/errors_log.txt", "a") 
        errors_file.write(f"{smi[0]+1},{smi[1]}\n")
        errors_file.close()

def fix_numbering(data_length):
    
    count = 0
    cp_path = f"{args.datasets_path}/{args.dataset_name}"

    with tqdm(total=data_length) as pbar:
        for i in range(data_length):

            if not os.path.isfile(f"{cp_path}/aug_1/{i+1}.png"):
                count += 1
                pbar.update()
                continue

            if count > 0:
                shutil.move(f"{cp_path}/aug_1/{i+1}.png", f"{cp_path}/aug_1/{i+1-count}.png")
                shutil.move(f"{cp_path}/aug_2/{i+1}.png", f"{cp_path}/aug_2/{i+1-count}.png")

            pbar.update()


def unite(data, args):
    
    data = [data]
    
    for k in args.k_values:
        with open(f"{args.datasets_path}/{args.dataset_name}/classes/cluster_{args.fp_type}_{k}.csv", 'r') as file:
            data.append([i[1] for i in list(csv.reader(file))[1:]])

    with open(f"{args.datasets_path}/{args.dataset_name}/{args.dataset_name}.csv", 'w') as file:
        classes_str = ",".join([f"classes_{k}" for k in args.k_values])
        len_classes = len (args.k_values)

        file.write(f"indices,smiles,{classes_str}\n")

        with tqdm(total=len(data[0])) as pbar:
            for idx in range(len(data[0])):
                classes_str = ",".join([f"{data[i][idx]}" for i in range(len_classes+1)])
                file.write(f"{idx+1},{classes_str}\n")
                pbar.update()



def parse_args():

    # Argument parsing
    parser = argparse.ArgumentParser(description='K-means clustering for molecular datasets')
    parser.add_argument('-datasets_path', type=str, default="../../Datasets", help='Path to the directory containing the dataset.')
    parser.add_argument('-dataset_name', type=str, default="chembl_25", help='Name of the dataset to be processed.')
    parser.add_argument('-dataset_type', type=str, default="pretraining", choices=["pretraining", "finetuning"], help='Type of dataset to process: "pretraining" or "finetuning".')
    parser.add_argument('-fp_type', type=str, default="maccs", choices=["maccs", "rdkit", "morgan", "descriptors"], help='Type of molecular fingerprints to generate: "maccs", "rdkit", "morgan", or "descriptors".')
    parser.add_argument('-function', default='full_generation', choices=["images", "fingerprints", "full_generation", "fp_clustering"], help='Select the operation to perform: generate molecular images, fingerprints, both ("full_generation"), or perform fingerprint-based clustering.')
    parser.add_argument('-num_workers', default=2, type=int, help='Number of parallel workers to use during image generation.')

    parser.add_argument('-r_replace', action='store_true', help='Whether to replace functional groups in the generated molecular images with numbered "R" groups (only in finetuning datasets).')

    parser.add_argument('-radius', default=2, type=int, help='Radius to use for generating Morgan fingerprints (only applicable for "morgan" fingerprint type).')
    parser.add_argument('-bits', default=1024, type=int, help='Number of bits to use when generating Morgan fingerprints.')

    parser.add_argument('-k_values', default=[30, 300], nargs='+', type=int,help='List of k-values (number of clusters) to try during K-means clustering.')
    parser.add_argument('-log', default=True, type=bool, help='Whether to log the clustering process and results.')
    parser.add_argument('-batch_size', default=20000, type=int, help='Number of molecules to process in each batch during clustering.')

    parser.add_argument('-download', default=True, type=bool, help='Whether to automatically download chembl_25 data.')

    parser.set_defaults(r_replace = False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.dataset_type == "pretraining":
        
        args.datasets_path += "/Pretraining"
        full_path = f"{args.datasets_path}/{args.dataset_name}"
        
        file_path = f"{full_path}/{args.dataset_name}.txt"

        if not os.path.isfile(f"{full_path}/{args.dataset_name}.txt"):

            if args.download and args.dataset_name == "chembl_25":
                if not os.path.isdir(full_path):
                    os.mkdir(full_path)
                download_chembl25(f"{full_path}/{args.dataset_name}.txt")
            
            else:
                raise Exception("smiles data isn't available")             

        with open(file_path, "r") as csv_file:    
            data = [i[0] for i in list(csv.reader(csv_file))]

        if args.function == 'images' or args.function == 'full_generation':
            if not os.path.isdir(f"{full_path}/aug_1"):
                os.mkdir(f"{full_path}/aug_1")  
            if not os.path.isdir(f"{full_path}/aug_2"):
                os.mkdir(f"{full_path}/aug_2") 

            original_len_data = len (data)
            print ("Generating images...")

            with Pool(processes=args.num_workers) as p: 
                with tqdm(total=len(data)) as pbar:
                    for _ in p.imap_unordered(gen_imgs, enumerate(data)): #generate all images and save in the output folders
                        pbar.update()
            
            # Reorder numbering if some images were'nt generated 
            if os.path.isfile(f"{full_path}/errors_log.txt"): 
                fix_numbering (len(data))

                with open(f"{full_path}/errors_log.txt", "r") as errors_file:
                    error_smi = [i[1] for i in list(csv.reader(errors_file, delimiter =','))]
            
                data = [smi for smi in data if smi not in error_smi] #filter smiles with no image

        if args.function in ['fingerprints', 'fp_clustering', 'full_generation']:
            if not os.path.isdir(f"{full_path}/FP"):
                os.mkdir(f"{full_path}/FP")  

            print ("Generating fingerprints...")  

            with open(f"{full_path}/FP/fingerprints_{args.fp_type}.txt", "w") as fingerprints_file:

                len_data = len(data)

                with tqdm(total=len_data) as pbar:
                    for smi in enumerate(data):
                        gen_fingerprints(smi, radius = args.radius, bits = args.bits, fp_type = args.fp_type)

                        pbar.update()

        if args.function in ['fp_clustering', 'full_generation'] :

            with open(f"{full_path}/FP/fingerprints_{args.fp_type}.txt", "r") as fingerprints_file:
                fingerprints = list(fingerprints_file)
            
            # preforming k-means amnalysis
            print ("k-means training...")  
            train(fingerprints, args)

            print ("k-means inference...")  
            inference(fingerprints, args)
            
            #unites the smiles file and the classes files to one csv, ready for training
            unite(data, args)

            print (f"generated a dataset of {len_data} molecules")

            if args.function != 'fp_clustering':
                if original_len_data != len_data:
                    print (f"Warning: {original_len_data - len_data} molecules weren't generated")

    if args.dataset_type == "finetuning":

        args.datasets_path += "/Finetuning"
        full_path = f"{args.datasets_path}/{args.dataset_name}"
        file_path = f"{full_path}/{args.dataset_name}.csv"

        with open(file_path, "r") as csv_file:    
            data = [i[1] for i in list(csv.reader(csv_file))[1:]]

        if not os.path.isdir(f"{full_path}/imgs"):
            os.mkdir(f"{full_path}/imgs")  

        with Pool(processes=args.num_workers) as p: 
            with tqdm(total=len(data)) as pbar:
                for _ in p.imap_unordered(gen_imgs, enumerate(data)): #generate all images and save in the output folders
                    pbar.update()

        original_len_data = len (data)

        if os.path.isfile(f"{full_path}/errors_log.txt"): 

            with open(f"{full_path}/errors_log.txt", "r") as errors_file:
                errors_smi = [i[1] for i in list(csv.reader(errors_file, delimiter =','))]
        
            len_data = len([smi for smi in data if smi not in errors_smi])

        else:
            len_data = original_len_data

        print (f"generated a dataset of {len_data} molecules")

        if original_len_data != len_data:
            print (f"Warning: {original_len_data - len_data} molecules weren't generated")


        

