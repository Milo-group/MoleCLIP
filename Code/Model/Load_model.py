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

class Conv1D(torch.nn.Module):
    def __init__(self,  num_features_xt=25, n_filters=32, embed_dim=128, output_dim=128):

        super(Conv1D, self).__init__()

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)


    def forward(self, target):

        # target is the protein sequence

        print (target.shape)
        embedded_xt = self.embedding_xt(target)
        print (embedded_xt.shape)
        conv_xt = self.conv_xt_1(embedded_xt)
        print (conv_xt.shape)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        return xt

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


    

    


    

