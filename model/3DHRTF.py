import torch
import torch.nn as nn
import sys
sys.path.append('/autofs/thau04b/hghallab/comp/Huawei/TechArena20241016')
from models import PointNetFeatureExtractor, Generator
from learning3d.models import DGCNN

class CustomPointNet(nn.Module):
    def __init__(self, freeze_features=True, load_feat_weights=True, regressor='mlp'):
        super(CustomPointNet, self).__init__()
        self.num_targets = 793 * 2 * 129 * 2
        self.freeze_features = freeze_features
        self.load_feat_weights = load_feat_weights
        self.regressor_type = regressor
        # self.frequency_enc = nn.Embedding(1, 16)

        # Feature extractor
        self.feat = PointNetFeatureExtractor()
        self.mid_channel = 2048 

        if load_feat_weights:
            # Load pretrained weights
            pretrained_dict = torch.load('/autofs/thau04b/hghallab/comp/3D_to_HRTF/point_cloud/ptnetwonrormals.pth')['model_state_dict']
            selected_dict = {}
            layers_to_load = ['sa1', 'sa2', 'sa3']
            for name, param in pretrained_dict.items():
                if any(layer in name for layer in layers_to_load):
                    selected_dict[name] = param
            
            # Load selected weights
            model_dict = self.feat.state_dict()
            model_dict.update(selected_dict)
            self.feat.load_state_dict(model_dict, strict=False)
      
        print("Number of parameters in Features Extractor:", sum(p.numel() for p in self.feat.parameters()))
        
        # Freeze feature extractor
        if freeze_features:
            for param in self.feat.parameters():
                param.requires_grad = False

        # Regression Head
        if self.regressor_type == 'mlp':
            self.regressor = nn.Sequential(
                nn.Linear(self.mid_channel, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_targets//2)
            )
        elif self.regressor_type == 'accoustic':
            self.regressor = Generator(self.mid_channel, target=129) 
        
        print("Number of parameters in Regressor:", sum(p.numel() for p in self.regressor.parameters()))
        print("Number of Trainable Parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        left, right = x
        left_feat = self.feat(left)
        right_feat = self.feat(right)
        feat = torch.cat((left_feat, right_feat), dim=-1)
        feat = feat.view(feat.size(0), -1)
        x = self.regressor(feat)
        # x = x.view(x.size(0), 793, 2, 129)
        x = x.permute(0, 2, 1, 3)
        return x


 
 