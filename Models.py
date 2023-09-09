import os
import torch
import torch.nn as nn
from collections import OrderedDict
from monai.networks.nets import ViTAutoEnc

class Classifier(nn.Module):    
    def __init__(self):
        super().__init__()
        self.model = ViTAutoEnc(
                    in_channels=4,
                    out_channels=4,
                    img_size=(96, 96, 96),
                    patch_size=(16, 16, 16),
                    pos_embed='conv',
                    num_heads=32,
                    num_layers=16,
                    hidden_size=2048,
                    mlp_dim=3072
        )
        '''Load the pre-trained model parameters'''
        logdir_path = os.path.normpath('./Pretrained_models')
        self.model_path = os.path.join(logdir_path, 'SSL_ViT_Block16.pth') # or SSL_ViT_Block4.pth
        vit_weights = torch.load(self.model_path, map_location=torch.device('cpu'))
        vit_weights_sp = OrderedDict()
        for k,v in vit_weights.items():
            name = k[7:]
            vit_weights_sp[name]=v
        self.model.load_state_dict(vit_weights_sp)
        
        '''Define the pooling layer and fully connected layers'''
        self.pool = nn.AvgPool2d(3,stride=3)
        self.FC = nn.Sequential(
            nn.Linear(72*682,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            )
    def forward(self, x):
        x = self.model(x)[1][15]
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.FC(x)
        return x
