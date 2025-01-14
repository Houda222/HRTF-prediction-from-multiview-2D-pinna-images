import torch
import torch.nn as nn
import torch.nn.functional as F
from pointNet_utils import *


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)), inplace=True)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
    
# Define the feature extractor as used in the original PointNet architecture
class PointNetFeatureExtractor(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointNetFeatureExtractor, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        return l3_points
    
class Generator(nn.Module):
    def __init__(self, in_channel, freq):
        super().__init__()
        self.in_channel = in_channel
        self.freq = freq
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 1024, kernel_size=(4, 4), stride=(3, 3)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(3, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv7 = nn.Sequential(
            nn.Upsample(size=(793, self.freq), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1, stride=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_channel, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.final_conv(x)
        return x