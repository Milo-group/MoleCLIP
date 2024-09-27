import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import MinMaxScaler
from kneed import DataGenerator, KneeLocator
import joblib
from tqdm import tqdm
import random
import time
import argparse 
import csv 

def find_knee(x, y, args):
    kl = KneeLocator(
        x=x,
        y=y,
        S=args.sensitivity,
        curve=args.curve,
        direction="decreasing",
        interp_method=args.interp_method,
        polynomial_degree=args.polynomial_degree,
    )
    return kl

def knee_point (args, xy = None):

    if xy == None:
        try:
            with open(f"{args.datasets_path}/{args.dataset_name}/k_means/inetias_{args.fp_type}.txt", 'r') as f:
                inertias = list(csv.reader(f, delimiter=","))
        except:
            raise(Exception("inertias file is not exist"))

        x = [int(i[0]) for i in inertias]
        y = [float(i[1][1:]) for i in inertias]

    kl = find_knee(x, y, args)

    print (f"Knee K is {kl.knee}")

    if args.interp_method == "polynomial":
        y = kl.Ds_y

    plot_figure (x, y, kl, args)



def plot_figure(x, y, kl, args):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, y, color="cornflowerblue", linewidth=6, label="Inertia curve")

    ax.scatter(
    [kl.knee], 
    [kl.knee_y], 
    color="orangered", 
    s=256,  
    marker="X", 
    label="Knee point",
    zorder=2
    )

    # Set labels and title
    ax.set_title("Knee point curve", fontsize=20)
    ax.set_xlabel("K classes", fontsize=18)
    ax.set_ylabel("Inertia", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=14)
    
    # Set background color
    ax.set_facecolor('white')

    fig.savefig(f"{args.datasets_path}/{args.dataset_name}/k_means/kneepoint_{args.fp_type}.png")
    


def calc_kmeans(k, data, args):
    """
    K-means clustering.
    Args:
        k (int): Number of clusters.
        batch_size (int): Batch size.
        fp_type (str): Type of fingerprints.
        data (list): Input data for clustering.
    Returns:
        model: Trained K-means model.
        inertia_: Sum of squared distances of samples to their closest cluster center.
    """
    if args.fp_type == "PCA" :
        # Process the data for PCA type
        processed_data = np.array([[float(x) for x in row.split(",")[:-1]] for row in data])
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(processed_data)
        return kmeans, kmeans.inertia_
    
    elif len(data) < args.batch_size:
                        
        if args.fp_type == "descriptors":
            # Process data for descriptors
            processed_data = np.array([[float(x) for x in row.split(" ")[:-1]] for row in data])
            processed_data = np.nan_to_num(batch_data, nan=-1.0)
            processed_data = scaler.fit_transform(batch_data)
        else:
            # Process data for other types
            processed_data = np.array([[int(x) for x in row[:-1]] for row in data])

        print(f"K is {k}:")
        print (processed_data.shape)    
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(processed_data)
        
        return kmeans, kmeans.inertia_
    
    else:
        # For other types, use MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=args.batch_size)
        print(f"K is {k}:")
        batch_list = list(range(0, len(data), args.batch_size))
        scaler = MinMaxScaler()

        with tqdm(total=len(batch_list)) as pbar:
            for batch in batch_list:
                if args.fp_type == "descriptors":
                    # Process data for descriptors
                    batch_data = np.array([[float(x) for x in row.split(" ")[:-1]] for row in data[batch:batch+args.batch_size]])
                    batch_data = np.nan_to_num(batch_data, nan=-1.0)
                    batch_data = scaler.fit_transform(batch_data)
                else:
                    # Process data for other types
                    batch_data = np.array([[int(x) for x in row[:-1]] for row in data[batch:batch+args.batch_size]])
                kmeans = kmeans.partial_fit(batch_data)
                pbar.update()

        return kmeans, kmeans.inertia_

def train(data, args):
    """
    Train K-means models for different values of k.

    Args:
        data (list): Input data for clustering.
        dataset_name (str): Name of the dataset.
        fp_type (str): Type of fingerprints.
        k_values (list): List of k values for training. If None, default values are used.
        log (bool): Flag to log results.
    """

    if not os.path.isdir(f"{args.datasets_path}/{args.dataset_name}/k_means"):
        os.mkdir(f"{args.datasets_path}/{args.dataset_name}/k_means") 
    
    if not os.path.isdir(f"{args.datasets_path}/{args.dataset_name}/k_means/{args.fp_type}"):
        os.mkdir(f"{args.datasets_path}/{args.dataset_name}/k_means/{args.fp_type}") 


    if args.k_values == [0]:
        if len (data) > 10000:
            args.k_values = [3, 5, 7, 10, 20, 30, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
        else:
            args.k_values = list(range(2,101,2))

    if args.log:
        file = open(f"{args.datasets_path}/{args.dataset_name}/k_means/inetias_{args.fp_type}.txt", 'a')

    for k in args.k_values:
        model, inertia = calc_kmeans(k, data, args)
        print(f'K={k}, Inertia={inertia}')
        if args.log:
            file.write(f"{k}, {inertia}\n")
            joblib.dump(model, f'{args.datasets_path}/{args.dataset_name}/k_means/{args.fp_type}/model_{k}.joblib')

    if args.log:
        file.close()

def inference(data, args):
    """
    Perform inference using trained K-means models.
    Args:
        data (list): Input data for clustering.
        dataset_name (str): Name of the dataset.
        fp_type (str): Type of fingerprints.
    """

    if args.k_values == [0]:
        args.k_values = [10]

    if not os.path.isdir(f"{args.datasets_path}/{args.dataset_name}/classes"):
        os.mkdir(f"{args.datasets_path}/{args.dataset_name}/classes") 
    
    if not os.path.isdir(f"{args.datasets_path}/{args.dataset_name}/k_means"):
        raise(Exception("kmeans directory is not exist"))

    if not os.path.isdir(f"{args.datasets_path}/{args.dataset_name}/k_means/{args.fp_type}"):
        raise(Exception(f"kmeans directory is not exist for {args.fp_type}"))

    for k in args.k_values:

        try:
            kmeans = joblib.load(f'{args.datasets_path}/{args.dataset_name}/k_means/{args.fp_type}/model_{k}.joblib')
            print (f"model-{k} inference")
        
        except:
            print (f"{k} clusters model isn't exist")
            continue
        
        batch_list = list(range(0, len(data), args.batch_size))

        if args.fp_type == "descriptors": 
            scaler = MinMaxScaler()

        with open(f"{args.datasets_path}/{args.dataset_name}/classes/cluster_{args.fp_type}_{k}.csv", 'a') as file:
            file.write(f"indices, classes\n")
            with tqdm(total=len(batch_list)) as pbar:
                for batch_idx, batch in enumerate(batch_list):
                    if args.fp_type == "descriptors":
                        batch_data = np.array([[float(x) for x in row.split(" ")[:-1]] for row in data[batch:batch+args.batch_size]])
                        batch_data = np.nan_to_num(batch_data, nan=-1.0)
                        batch_data = scaler.fit_transform(batch_data)
                    elif args.fp_type == "PCA":
                        batch_data = np.array([[float(x) for x in row.split(",")[:-1]] for row in data[batch:batch+args.batch_size]])
                    else:
                        batch_data = np.array([[int(x) for x in row[:-1]] for row in data[batch:batch+args.batch_size]])
                    classes = kmeans.predict(batch_data)
                    file.writelines([f"{batch + idx + 1}, {classes[idx]}\n" for idx in range(batch_data.shape[0])])
                    pbar.update()

def parse_args():

    # Argument parsing
    parser = argparse.ArgumentParser(description='K-means clustering')
    parser.add_argument('-datasets_path', type=str, default="../../Datasets/Pretraining", help='dataset path')
    parser.add_argument('-dataset_name', type=str, default="chembl_25", help='Name of the dataset')
    parser.add_argument('-fp_type', type=str, default="maccs", help='Type of fingerprints')
    parser.add_argument('-k_values', default = [0], nargs='+', type=int, help='k_values to use')
    parser.add_argument('-batch_size', type=int, default = 20000, help='batch_size')
    parser.add_argument('-function', default = 'train', choices=["train", "inference", "kneepoint", "unite"], help='choose function to execute')
    parser.add_argument('-random_seed', type=int, default = 42, help='random_seed')
    parser.add_argument('-save', action='store_false', help='weather to save training data')
    
    parser.add_argument('-knee_plot', action='store_true',  help='weather to infer instead of train')
    parser.add_argument('-sensitivity', type=float, default = 1.0, help='kneedle sensitivity')
    parser.add_argument('-curve',  default = 'convex', choices=["concave", "convex"], help='kneedle curve')
    parser.add_argument('-interp_method',  default = 'interp1d', choices=["interp1d", "polynomial"], help='kneedle interpetation method')
    parser.add_argument('-polynomial_degree',  default = 10, type=float, help='kneedle polynomial degree')

    parser.set_defaults(inference = False, save = True, knee_plot = False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    file_path = f"{args.datasets_path}/{args.dataset_name}/FP/fingerprints_{args.fp_type}.txt"

    try:
        with open(file_path, 'r') as file:
            data = list(file) 
    except:
        raise(Exception("fingerprints file is not exist"))


    if args.function == "inference":
        inference(data, args)
    elif args.function == "train":
        random.seed(args.random_seed)
        random.shuffle(data)
        train(data, args)
    elif args.function == "kneepoint":
        knee_point(args)

