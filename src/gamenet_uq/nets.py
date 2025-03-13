"""Module containing the Graph Neural Network classes."""
from typing import Optional
import math

import torch
from torch.nn import Linear, Softplus
from torch import Tensor
from torch.distributions import Normal
from torch_geometric.nn.conv import SAGEConv, TAGConv
from torch_geometric.utils import to_dense_batch

class MAB(torch.nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, bias: bool):
        """
        Multihead Attention Block (MAB).
        Readapted from PyG 2.0.3 without normalization and dropout, and bias option.
        """
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = Linear(dim_Q, dim_V, bias)
        self.layer_k = Linear(dim_K, dim_V, bias)
        self.layer_v = Linear(dim_K, dim_V, bias)
        self.fc_o = Linear(dim_V, dim_V, bias)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        Q = self.fc_q(Q)

        K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        out = out + self.fc_o(out).relu()        

        return out

class PMA(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int, num_seeds: int, bias: bool):
        """
        Pooling Multihead Attention (PMA).
        Readapted from PyG 2.0.3 without normalization and dropout, and bias option.
        """
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask)


class GameNetUQ(torch.nn.Module):
    def __init__(self, 
                 node_features: int,               
                 dim: int,                  
                 num_linear: int = 0,
                 num_conv: int = 3,
                 bias: bool = False,
                 conv = SAGEConv, 
                 pool_heads: int = 1, 
                 seed: int = None):
        """

        Args:
            num_in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            dim (int, optional): Layer width.
            num_linear (int, optional): Number of dense. Default to 3.
            num_conv (int, optional): Number of convolutional layers. Default to 3.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        if seed is not None and type(seed) == int:
            torch.manual_seed(seed)
        super(GameNetUQ, self).__init__()
        self.sigma = torch.nn.ReLU()     

        self.input_layer = Linear(node_features, dim, bias=bias)
        self.lin_block = torch.nn.ModuleList([Linear(dim, dim, bias=bias) for _ in range(num_linear)])
        self.conv_block = torch.nn.ModuleList([conv(dim, dim, bias=bias) for _ in range(num_conv)])
        self.ts_layer = TAGConv(dim, dim, bias=bias, normalize=False, K=3)
        self.lin_a = Linear(dim, dim, bias=bias) 
        self.lin_b = Linear(dim, 2, bias=bias)  
        self.pma = PMA(channels = dim, 
                       num_heads = pool_heads, 
                       num_seeds = 1, 
                       bias = bias)
        
    def forward(self, data):
        #---------------------------------#
        # NODE LEVEL (FFNN & CONVOLUTION) #
        #---------------------------------#        
        out = self.sigma(self.input_layer(data.x))   
        for layer in range(len(self.lin_block)):  
            out = self.sigma(self.lin_block[layer](out))
        for layer in range(len(self.conv_block)):  
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        out = self.ts_layer(out, data.edge_index, data.edge_attr)  
        out = self.lin_a(out)
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        batch_x, mask = to_dense_batch(x=out, 
                                       batch=data.batch, 
                                       fill_value=0.0, 
                                       max_num_nodes=500, # conservative val to avoid different predictions in inference (with big graphs)
                                       batch_size=None)  
        mask = (~mask).unsqueeze(1).to(dtype=out.dtype) * -1e9
        out = self.pma(x=batch_x, mask=mask)
        out = self.lin_b(out.squeeze(1))  
        return Normal(out[:, 0], Softplus()(out[:, 1]))
    

class GameNetUQ_ablation(torch.nn.Module):
    def __init__(self, 
                 node_features: int,               
                 dim: int,                  
                 num_linear: int = 0,
                 num_conv: int = 3,
                 bias: bool = False,
                 conv = SAGEConv, 
                 pool_heads: int = 1, 
                 ts_layer: bool = True, 
                 gcn:bool = True, 
                 uq: bool = True,
                 seed: int = None):
        """
        GAME-Net-UQ model class for ablation study.

        Args:
            num_in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            dim (int, optional): Layer width.
            num_linear (int, optional): Number of dense. Default to 3.
            num_conv (int, optional): Number of convolutional layers. Default to 3.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
            ts_layer (bool, optional): Include TAGConv layer. Default to True.
            gcn (bool, optional): If True, include this node feature in the model, else not.
            uq (bool, optional): If True, model outputs a Normal distribution, else a point estimate (no uq).
        """
        super(GameNetUQ_ablation, self).__init__()
        if seed is not None and type(seed) == int:
            torch.manual_seed(seed)
        self.sigma = torch.nn.ReLU()     

        self.input_layer = Linear(node_features, dim, bias=bias)
        self.lin_block = torch.nn.ModuleList([Linear(dim, dim, bias=bias) for _ in range(num_linear)])
        self.conv_block = torch.nn.ModuleList([conv(dim, dim, bias=bias) for _ in range(num_conv)])
        self.ts_layer = TAGConv(dim, dim, bias=bias, normalize=False, K=3)
        self.lin_a = Linear(dim, dim, bias=bias) 
        self.pma = PMA(channels = dim, 
                       num_heads = pool_heads, 
                       num_seeds = 1, 
                       bias = bias)
        self.gcn = gcn
        self.ts = ts_layer
        self.uq = uq
        self.out_dim = 2 if uq else 1
        self.lin_b = Linear(dim, self.out_dim, bias=bias)  
        
    def forward(self, data):
        #---------------------------------#
        # NODE LEVEL (FFNN & CONVOLUTION) #
        #---------------------------------# 
        if self.gcn:
            out = self.sigma(self.input_layer(data.x))
        else:       
            out = self.sigma(self.input_layer(data.x[:,:-1]))   
        for layer in range(len(self.lin_block)):  
            out = self.sigma(self.lin_block[layer](out))
        for layer in range(len(self.conv_block)):  
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        if self.ts:
            out = self.ts_layer(out, data.edge_index, data.edge_attr)  
        out = self.lin_a(out)
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        batch_x, mask = to_dense_batch(x=out, 
                                       batch=data.batch, 
                                       fill_value=0.0, 
                                       max_num_nodes=500, # conservative value to avoid different predictions during inference (mainly with big graphs)
                                       batch_size=None)  
        mask = (~mask).unsqueeze(1).to(dtype=out.dtype) * -1e9
        out = self.pma(x=batch_x, mask=mask)
        out = self.lin_b(out.squeeze(1))
        if self.uq:
            return Normal(out[:, 0], Softplus()(out[:, 1]))
        else:
            return Normal(out[:, 0], 1.0)  # constant uncertainty = no uncertainty -> min(negative log likelihood) = min(MSE)