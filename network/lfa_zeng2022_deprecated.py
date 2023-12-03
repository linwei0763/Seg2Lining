import network.pytorch_utils as pt_utils
import torch
import torch.nn as nn


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.block1 = Block(d_in, d_out // 2, self.config)
        self.block2 = Block(d_out // 2, d_out // 2, self.config)
        self.block3 = Block(d_out, d_out, self.config)
        self.mlp1 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_out * 2, d_out * 2, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        f1 = self.block1(xyz, feature, neigh_idx)
        f2 = self.block2(xyz, f1, neigh_idx)
        f3 = self.block3(xyz, torch.cat([f1, f2], dim=1), neigh_idx)
        f = self.mlp1(torch.cat([f1, f2, f3], dim=1))
        
        shortcut = self.shortcut(feature)
        f = f + shortcut
        f = self.mlp2(f)
        
        return f


class Block(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.mlp1 = pt_utils.Conv2d(10, d_out, kernel_size=(1, 1), bn=True)

        
    def forward(self, xyz, feature, neigh_idx):
        
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        
        

        
        return 
        
    def relative_pos_encoding(self, xyz, neigh_idx):
        
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz

        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))

        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features