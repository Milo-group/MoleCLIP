# MoleCLIP
## Background
MoleCLIP is a molecular representation learning framework that accepts images of molecules as inputs and adopts a ViT(visual transformer) architecture initialized with weights from OpenAI’s CLIP (Contrastive Language–Image Pretraining).
The reliance on CLIP as a foundation model enables MoleCLIP to train in a few-shot manner on significantly less molecular pretraining data than training from scratch. Additionally, it contributes to MoleCLIP's robustness to distribution shifts, which is the ability to maintain performance on new tasks or domains that differ from those used in training.

## Dependencies
MoleCLIP requires Python 3.7 or newer.
To avoid conflicts with existing packages, it's recommended to create a new conda environment before installation.
To install the necessary dependencies, navigate to the MoleCLIP directory and run the following:

Create a new conda environment:
```
conda create -n MoleCLIP python=3.7
conda activate MoleCLIP
```

Install the necessary dependencies by navigating to the MoleCLIP directory and running:
```
pip install -r requirements.txt
```
## Pretraining

### Dataset format and example usage
Before starting the pretraining process, ensure your dataset files are correctly formatted and in the appropriate directory.
Dataset Format
A dataset file should be a CSV containing the following four columns: indices, smiles, classes_1, and classes_2. The class columns refer to the pseudo-labels assigned by a fingerprint-based clustering procedure.
#### Example Usage
To generate the necessary CSV file and corresponding pretraining images:
1. Place a text file containing molecular SMILES strings (one per line) in the Datasets folder within a subfolder named after the dataset. For example, for the chembl_25_subset dataset, the file path should be: MoleCLIP/Datasets/chembl_25_subset/chembl_25_subset.txt
2. Generate the dataset CSV and pretraining images by running the following commands:
```
cd Code/Preprocessing
python dataset_generation.py -dataset_name chembl_25_subset
```
**Note:** This command will generate both the dataset CSV and the corresponding training images. It is recommended to generate images in advance, especially for large datasets, as this process can be time-consuming.

3. To skip the image generation step during pretraining and generate images on the fly (which may slow down the pretraining process), use the **-function fp_clustering** argument. However, this approach will rely on CPU processing during training, which can significantly increase the pretraining time.

To run the pretraining on the chembl_25_subset dataset:
```
cd Code
python Pretrain.py -dataset_name chembl_25_subset -batch 16 -log_every 10
```
If you want to run pretraining without pre-generated images, add the argument **-dataset_type smiles**


### Training on the full chembl_25 dataset
To automatically download and preprocess the full ChEMBL_25 dataset, use the following commands:
```
cd Code/Preprocessing
python dataset_generation.py 
```
**Note:** Image generation can be resource-intensive. It is recommended to use a strong computer/machine with multiple CPUs. You can adjust the number of parallel processes by adding the -num_workers argument, and specify the number of available CPUs.


After preprocessing, initiate the pretraining with:
```
cd Code
python Pretrain.py
```
**Important:** Pretraining is optimized for GPUs. Training on CPUs is possible but significantly slower and not recommended.

## Finetuning
### Dataset Format
For finetuning, the dataset should be a CSV file containing three columns: indices, smiles, and labels. The labels should be numeric, representing the target values. If a molecule has multiple labels, separate them with spaces. The CSV file should be placed in the Datasets folder within a subfolder named after the dataset. For example, for the BACE dataset, the file path should be: MoleCLIP/Datasets/bace/bace.csv

### Data generation
To generate images before finetuning, use the following command:
```
cd Code/Preprocessing
python dataset_generation.py -dataset_name bace -dataset_type finetuning
```
**Note:** Add an **-r_replace** argument to replace functional groups in the generated molecular images with numbered "R" groups.

### Example finetuning run
Finetuning should be performed based on pretrained weights. By default, the finetuning process uses pretrained weights provided by the MoleCLIP framework. To download these weights and start finetuning on a given dataset, run:

```
cd Code
python Finetune.py -dataset_name bace -download
```

### Key Finetuning Arguments
Several arguments can be added to customize the finetuning process:

1. If you prefer to use custom pretrained weights, the pretraining checkpoints are saved in the MolCLIP/Checkpoints directory. Each checkpoint is automatically organized within a unique folder corresponding to the pretraining run. The checkpoints follow this naming convention:{run_name}/{run_name-version}.pth. To start a finetuning session with a specific checkpoint, add the argument **-cp_name '{run_name-version}.pth'**
2. **-dataset_type smiles:** Use this to run finetuning without pre-generated images, generating them on-the-fly.
3. **-lr_bb, -lr_ff, -wd, -batch, -augmentation, and more:** These are hyperparameters for the finetuning process. You can specify multiple values for each to perform hyperparameter optimization. For example: **-lr_ff 0.001 0.0004 0.0001**