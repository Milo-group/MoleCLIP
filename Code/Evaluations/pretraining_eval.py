import torch
from torch import nn

def features_to_logits (model, image1_features, image2_features, temperature = 100, check = False):
        
    image1_features = image1_features / image1_features.norm(dim=1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=1, keepdim=True)

    logits_per_image1 = temperature * image1_features @ image2_features.t()       
    logits_per_image2 = logits_per_image1.t()

    return logits_per_image1, logits_per_image2

def pt_eval(model, val_dataset, classes, args, eval_part = 1 ,device = "cuda"):
    
    model.eval()

    data = iter(val_dataset)
    batches_num = int (len(data) * eval_part)

    CNE = nn.CrossEntropyLoss()
    
    total_samples = 0 
    total_acc = 0

    total_singles_loss = 0
    total_cls1_loss = 0
    total_cls2_loss = 0
    total_loss = 0
    
    it_dir = {}

    print ("accuracy evaluation")
    for it in range(batches_num):

        image1, image2, cls1_labels, cls2_labels = next(data)

        image1_features = model.model_image(image1).float()
        image2_features = model.model_image(image2).float()

        labels = torch.arange(image1.shape[0]).to(device)

        cls_1_preds = model.cls_1(image1_features)
        cls_2_preds = model.cls_2(image1_features)

        logits_per_image1, logits_per_image2 = features_to_logits (model, image1_features, image2_features, args.temperature)


        loss_singles = (CNE(logits_per_image1, labels) + CNE(logits_per_image2, labels))/2
        cls1_loss = CNE(cls_1_preds, cls1_labels) 
        cls2_loss = CNE(cls_2_preds, cls2_labels)

        loss = loss_singles + cls1_loss/args.cls_1_l + cls2_loss/args.cls_2_l 

        image1_class = torch.argmax(logits_per_image1, 1)
        image2_class = torch.argmax(logits_per_image2, 1)

        acc_1 = torch.sum(torch.eq(image1_class, labels))
        acc_2 = torch.sum(torch.eq(image2_class, labels))
        
        total_singles_loss += loss_singles.item()
        total_cls1_loss += cls1_loss
        total_cls2_loss += cls2_loss
        total_loss += loss.item()

        total_acc += acc_1 + acc_2
        total_samples += labels.shape[0] * 2

    it_dir = {'Val/eval_score': total_acc/total_samples, 'Val/eval_loss': total_loss/batches_num, f'Val/total_cls{classes[0]}_loss': total_cls1_loss/batches_num,
              f'Val/total_cls{classes[1]}_loss': total_cls2_loss/batches_num, 'Val/eval_single_loss': total_singles_loss/batches_num}

    return it_dir