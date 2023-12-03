import network.pytorch_utils as pt_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class LFA(nn.Module):
    
    def __init__(self, d_in, d_out, config):
        super().__init__()
        
        self.config = config
        
        self.fc1 = pt_utils.Conv2d(d_in, 2 * d_out, kernel_size=(1, 1), activation=None)
        
        self.w_qs = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.w_ks = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.w_vs = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        
        self.fc_delta1 = pt_utils.Conv2d(3, 3, kernel_size=(1, 1), activation=None)
        self.ln_delta1 = nn.LayerNorm(3)
        self.fc_delta2 = pt_utils.Conv2d(3, 2 * d_out, kernel_size=(1, 1), activation=None)
        
        self.fc_gamma1 = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.ln_gamma1 = nn.LayerNorm(2 * d_out)
        self.fc_gamma2 = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), activation=None)
        self.ln_gamma2 = nn.LayerNorm(2 * d_out)
        
        self.mlp = pt_utils.Conv2d(2 * d_out, 2 * d_out, kernel_size=(1, 1), bn=True)
        self.shortcut = pt_utils.Conv2d(d_in, 2 * d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        
        pre = feature
        
        knn_xyz = self.gather_neighbour(xyz, neigh_idx).permute((0, 3, 1, 2))
        
        x = self.fc1(feature)
        
        q = self.w_qs(x)
        k = self.gather_neighbour(self.w_ks(x).squeeze(3).permute((0, 2, 1)), neigh_idx).permute((0, 3, 1, 2))
        v = self.gather_neighbour(self.w_vs(x).squeeze(3).permute((0, 2, 1)), neigh_idx).permute((0, 3, 1, 2))

        pos_enc = self.fc_delta1(xyz.permute(0, 2, 1).unsqueeze(3).repeat(1, 1, 1, neigh_idx.shape[-1]) - knn_xyz)
        pos_enc = pos_enc.permute((0, 2, 3, 1))
        pos_enc = F.relu(self.ln_delta1(pos_enc))
        pos_enc = pos_enc.permute((0, 3, 1, 2))
        pos_enc = self.fc_delta2(pos_enc)
        
        attn = q.repeat(1, 1, 1, neigh_idx.shape[-1]) - k + pos_enc
        attn = attn.permute((0, 2, 3, 1))
        attn = F.relu(self.ln_gamma1(attn))
        attn = attn.permute((0, 3, 1, 2))
        attn = self.fc_gamma1(attn)
        attn = attn.permute((0, 2, 3, 1))
        attn = F.relu(self.ln_gamma2(attn))
        attn = attn.permute((0, 3, 1, 2))
        attn = self.fc_gamma2(attn)

        attn = F.softmax(attn, dim=-1)
        
        res = torch.einsum('bdnk,bdnk->bdn', attn, v + pos_enc).unsqueeze(3)
        
        res = self.mlp(res)
        
        shortcut = self.shortcut(pre)

        return F.leaky_relu(res + shortcut, negative_slope=0.2)
        
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, d))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        
        return features
