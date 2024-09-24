# Checkpoints saving
Finetuning should be performed based on pretrained weights. By default, the finetuning process uses pretrained weights provided by the MoleCLIP framework. To download these weights and start finetuning on a given dataset, run:

```
cd Code
python Finetune.py -dataset_name bace -download
```

If you prefer to use custom pretrained weights, the pretraining checkpoints are saved in the MolCLIP/Checkpoints directory. Each checkpoint is automatically organized within a unique folder corresponding to the pretraining run. The checkpoints follow this naming convention:{run_name}/{run_name-version}.pth. To start a finetuning session with a specific checkpoint, add the argument -cp_name '{run_name-version}.pth'
