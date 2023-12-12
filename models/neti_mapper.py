import random
from typing import Optional, List
import pdb
import torch
import torch.nn.functional as F
from torch import nn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import SAGEConv

from constants import UNET_LAYERS
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder
from utils.types import PESigmas


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GraphConvolutionalNetwork, self).__init__()
        # Define graph convolution layers
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=hidden_feats, aggregator_type='mean', activation=nn.ReLU(), norm=nn.BatchNorm1d(hidden_feats))
        # self.conv2 = SAGEConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, 160)
    
    def forward(self, g, features):
        # Apply graph convolutional layers
        x = self.conv1(g, features, edge_weight=g.edata['w'])
        # x = F.relu(self.conv2(g, x))
        
        # Global pooling to obtain a graph-level representation
        # g.ndata['h'] = x
        # global_mean = dgl.mean_nodes(g, 'h')
        
        # Classification layer
        output = self.fc(x)
        
        return output


class NeTIMapper(nn.Module):
    """ Main logic of our NeTI mapper. """

    def __init__(self, output_dim: int = 768,
                 unet_layers: List[str] = UNET_LAYERS,
                 use_nested_dropout: bool = True,
                 nested_dropout_prob: float = 0.5,
                 norm_scale: Optional[torch.Tensor] = None,
                 use_positional_encoding: bool = True,
                 num_pe_time_anchors: int = 10,
                 pe_sigmas: PESigmas = PESigmas(sigma_t=0.03, sigma_l=2.0),
                 output_bypass: bool = True,
                 token_embeds: bool = True, 
                 placeholder_token_id_list: bool = True,
                 g: bool = True):
        super().__init__()
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_prob = nested_dropout_prob
        self.norm_scale = norm_scale
        self.output_bypass = output_bypass
        if self.output_bypass:
            output_dim *= 2  # Output two vectors

        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.encoder = NeTIPositionalEncoding(sigma_t=pe_sigmas.sigma_t, sigma_l=pe_sigmas.sigma_l).cuda()
            self.input_dim = num_pe_time_anchors * len(unet_layers)
        else:
            self.encoder = BasicEncoder().cuda()
            self.input_dim = 2

        self.set_net(num_unet_layers=len(unet_layers),
                     num_time_anchors=num_pe_time_anchors,
                     output_dim=output_dim)
        # self.token_embeds = token_embeds
        # self.placeholder_token_id_list = placeholder_token_id_list
        self.g = g

    def set_net(self, num_unet_layers: int, num_time_anchors: int, output_dim: int = 768):
        self.input_layer = self.set_input_layer(num_unet_layers, num_time_anchors)
        # self.net = nn.Sequential(self.input_layer,
        #                          nn.Linear(self.input_dim, 128), nn.LayerNorm(128), nn.LeakyReLU(),
        #                          nn.Linear(128, 128), nn.LayerNorm(128), nn.LeakyReLU())
        self.net = nn.Sequential(self.input_layer,)
        # self.net_emb = nn.Sequential(nn.Linear(768, self.input_dim),)
        self.net_emb = GraphConvolutionalNetwork(768, 16)
        self.net2 = nn.Sequential(nn.Linear(self.input_dim*2, 128), nn.LayerNorm(128), nn.LeakyReLU(),
                         nn.Linear(128, 128), nn.LayerNorm(128), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(128, output_dim))

    def set_input_layer(self, num_unet_layers: int, num_time_anchors: int) -> nn.Module:
        # pdb.set_trace()
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w * 2, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(num_time_anchors, num_unet_layers)
        else:
            input_layer = nn.Identity()
        return input_layer

    def forward(self, token_embs: torch.Tensor, timestep: torch.Tensor, unet_layer: torch.Tensor, new_token_ids: torch.Tensor, new_embs: torch.Tensor, truncation_idx: int = None) -> torch.Tensor:
        # pdb.set_trace()
        embedding = self.extract_hidden_representation(new_embs, timestep, unet_layer, new_token_ids)
        if self.use_nested_dropout:
            embedding = self.apply_nested_dropout(embedding, truncation_idx=truncation_idx)
        embedding = self.get_output(embedding)
        return embedding

    def get_encoded_input(self, token_embs: torch.Tensor, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(timestep, unet_layer)

    def extract_hidden_representation(self, new_embs: torch.Tensor, timestep: torch.Tensor, unet_layer: torch.Tensor, new_token_ids: torch.Tensor) -> torch.Tensor:
        encoded_input = self.get_encoded_input(new_embs, timestep, unet_layer)
        embedding = self.net2(torch.cat((self.net(encoded_input), self.net_emb(self.g, new_embs)[new_token_ids-49408]), 1))

        return embedding

    def apply_nested_dropout(self, embedding: torch.Tensor, truncation_idx: int = None) -> torch.Tensor:
        if self.training:
            if random.random() < self.nested_dropout_prob:
                dropout_idxs = torch.randint(low=0, high=embedding.shape[1], size=(embedding.shape[0],))
                for idx in torch.arange(embedding.shape[0]):
                    embedding[idx][dropout_idxs[idx]:] = 0
        if not self.training and truncation_idx is not None:
            for idx in torch.arange(embedding.shape[0]):
                embedding[idx][truncation_idx:] = 0
        return embedding

    def get_output(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.output_layer(embedding)
        if self.norm_scale is not None:
            embedding = F.normalize(embedding, dim=-1) * self.norm_scale
        return embedding
