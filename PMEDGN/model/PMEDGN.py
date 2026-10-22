import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, seq_len, num_nodes, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=num_nodes, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class SpatialAttention(nn.Module):

    def __init__(self, num_nodes):
        super().__init__()
        self.W_q = nn.Linear(num_nodes, num_nodes)
        self.W_k = nn.Linear(num_nodes, num_nodes)

    def forward(self, x):
        attn = torch.bmm(self.W_q(x), self.W_k(x).transpose(1, 2))
        attn = F.softmax(attn / (x.shape[-1] ** 0.5), dim=-1)
        return torch.bmm(attn, x)


class MultiScaleGraphLearner(nn.Module):
    def __init__(self, num_nodes, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_nodes, num_nodes * 2),
                nn.ReLU(),
                nn.Linear(num_nodes * 2, num_nodes)
            )
            for _ in range(num_scales)
        ])

    def forward(self, x):
        graphs = []
        for scale, learner in enumerate(self.learners):
            if scale == 0:
                agg = x.mean(dim=1)
            elif scale == 1:
                agg = x[:, ::2, :].mean(dim=1)
            else:
                agg = x[:, ::3, :].mean(dim=1)

            adj = learner(agg)
            adj = F.softmax(adj, dim=-1)
            graphs.append(adj)

        return graphs

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = x[:, -1, :]
        h_prop = h * adj
        return self.W(h_prop)


class PMEDGN(nn.Module):
    def __init__(self, num_nodes, embed_dim, seq_len, hidden_channels, out_channels=1,
                 num_scales=3, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_scales = num_scales
        self.channel_conv = nn.Conv1d(seq_len, num_scales * hidden_channels, kernel_size=1)
        self.temporal_attn = TemporalAttention(seq_len, num_nodes, num_heads=num_heads)
        self.spatial_attn = SpatialAttention(num_nodes)
        self.graph_learners = MultiScaleGraphLearner(num_nodes, num_scales)
        self.gcn_layers = nn.ModuleList([
            GraphConv(num_nodes, hidden_channels)
            for _ in range(num_scales)
        ])
        self.global_fusion = nn.Sequential(
            nn.Linear(num_scales * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_attn = self.temporal_attn(x)
        x_attn = self.spatial_attn(x_attn)
        graphs = self.graph_learners(x_attn)

        scale_outputs = []
        for i, (graph, gcn) in enumerate(zip(graphs, self.gcn_layers)):
            h = gcn(x_attn, graph)
            scale_outputs.append(h)

        combined = torch.cat(scale_outputs, dim=-1)
        output = self.global_fusion(combined)

        return output