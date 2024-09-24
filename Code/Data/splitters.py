####
#### partially taken from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/pahelix/utils/splitters.py
####

from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*') 

import random
from collections import defaultdict
from typing import List, Set, Union, Dict
import numpy as np

from rdkit.Chem.Scaffolds import MurckoScaffold



def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold



def random_scaffold_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 42):
        
    """
    Args:
        dataset(InMemoryDataset): the dataset to split. Make sure each element in
            the dataset has key "smiles" which will be used to calculate the 
            scaffold.
        frac_train(float): the fraction of data to be used for the train split.
        frac_valid(float): the fraction of data to be used for the valid split.
        frac_test(float): the fraction of data to be used for the test split.
        seed(int|None): the random seed.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)

    for ind in range(N):

        try:
            scaffold = generate_scaffold(dataset[ind], include_chirality=True)
            scaffolds[scaffold].append(ind)
        except:
            continue

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # all_sets = train_idx + valid_idx + test_idx

    return train_idx, valid_idx, test_idx



def scaffold_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1):

    """
    Args:
        dataset(InMemoryDataset): the dataset to split. Make sure each element in
            the dataset has key "smiles" which will be used to calculate the 
            scaffold.
        frac_train(float): the fraction of data to be used for the train split.
        frac_valid(float): the fraction of data to be used for the valid split.
        frac_test(float): the fraction of data to be used for the test split.
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for ind in range(N):
        scaffold = generate_scaffold(dataset[ind], include_chirality=True)

        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [ind]
        else:
            all_scaffolds[scaffold].append(ind)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0


    return train_idx, valid_idx, test_idx

def random_split(indices, val_size, test_size, seed):

    split_test = int(np.floor(val_size * len(indices))) 
    split_val = int(np.floor(test_size * len(indices))) 
    if seed > 0:
        np.random.seed(seed)
    np.random.shuffle(indices)

    return indices[(split_test+split_val):], indices[split_test:(split_test+split_val)], indices[:split_test]
