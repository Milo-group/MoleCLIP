import sys
sys.path.append('..')

import torch
from torch import nn

import clip


def create_mlp(input_dim = 512, inner_layers = 2, inner_dim = 1024, output_dim = 1, dropout_rate = 0.2):

    layers = []

    if inner_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    
    else:
        layers.append(nn.Linear(input_dim, inner_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        for i in range(inner_layers-1):
            layers.append(nn.Linear(inner_dim, inner_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(inner_dim, output_dim))

    mlp = nn.Sequential(*layers)

    return mlp

class Image_encoding_CLIP(nn.Module):
    def __init__(self, model) :
        super(Image_encoding_CLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)
      
    
    
class Model(nn.Module):
    def __init__(self, vit = "ViT-B/32", parallel = True, device = "cuda", classes = []):
        super().__init__()

        clip_model, img_preprocess = clip.load(vit)

        if parallel and device == "cuda":
            self.model_image = torch.nn.DataParallel(Image_encoding_CLIP(clip_model))
        else:
            self.model_image = Image_encoding_CLIP(clip_model)

        if classes != []:
            self.cls_1 = create_mlp(inner_layers = 0, output_dim = classes[0], dropout_rate = 0).to(device)
            self.cls_2 = create_mlp(inner_layers = 0, output_dim = classes[1], dropout_rate = 0).to(device)


    

    


    

