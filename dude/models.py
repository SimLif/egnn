'''
Author: haoqiang haoqiang@mindrank.ai
Date: 2022-08-05 02:28:42
LastEditors: haoqiang haoqiang@mindrank.ai
LastEditTime: 2022-08-05 03:59:38
FilePath: /work-home/egnn/dude/models.py
Description: 

Copyright (c) 2022 by haoqiang haoqiang@mindrank.ai, All Rights Reserved. 
'''
import os
import sys

from icecream import ic
base_dir = os.path.dirname(os.path.dirname(__file__))
ic(base_dir)
sys.path.append(base_dir)

import torch
from torch import nn
from models.gcl import E_GCL


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col    = edge_index
        trans       = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg         = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord      += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, edge_mask, edge_attr=None, node_attr=None):
        row, col           = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf  = hidden_nf
        self.device     = device
        self.n_layers   = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)
        self.similarity  = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.loss        = nn.CrossEntropyLoss()

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes, label=None):
        h = []
        for i in range(2):
            h[i] = self.embedding(h0[i])
            for j in range(0, self.n_layers):
                if self.node_attr:
                    h[i], _, _ = self._modules[f"gcl_{j}"](h[i], edges[i], x[i], edge_mask[i], edge_attr=edge_attr[i], node_attr=h0[i])
                else:
                    h[i], _, _ = self._modules[f"gcl_{j}"](h[i], edges[i], x[i], edge_mask[i], edge_attr=edge_attr[i], node_attr=None)

            h[i] = self.node_dec(h[i])
            h[i] = h[i] * node_mask[i]
            h[i] = h[i].view(-1, n_nodes[i], self.hidden_nf)
            h[i] = torch.sum(h[i], dim=1)


        sim = self.similarity(h[0], h[1])
        if label:
            loss = self.loss(sim, label)

        return (smi, loss) if label esle (sim, )
