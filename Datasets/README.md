# Dataset Formats
## Pretraining
A dataset file should be a CSV containing the following four columns: indices, smiles, classes_1, and classes_2. The class columns refer to the pseudo-labels assigned by a fingerprint-based clustering procedure.
To generate the necessary CSV file and corresponding pretraining images:
1. Place a text file containing molecular SMILES strings (one per line) in the Datasets folder within a subfolder named after the dataset. For example, for the chembl_25_subset dataset, the file path should be: MoleCLIP/Datasets/Pretraining/chembl_25_subset/chembl_25_subset.txt
2. Generate the dataset CSV and pretraining images by running the following commands:

## Finetuning
For finetuning, the dataset should be a CSV file containing three columns: indices, smiles, and labels. The labels should be numeric, representing the target values. If a molecule has multiple labels, separate them with spaces. The CSV file should be placed in the Datasets folder within a subfolder named after the dataset. For example, for the BACE dataset, the file path should be: MoleCLIP/Datasets/Finetuning/bace/bace.csv

## Finetuning_DTA
To prepare the data for fine-tuning on the DTA task, follow these steps:
1. Clone or download the KIBA and Davis dataset folders from the [GraphDTA repository](https://github.com/thinng/GraphDTA/tree/master).
2. Copy the dataset folders into the current directory.
3. Run the following command to preprocess the data: python create_data.py