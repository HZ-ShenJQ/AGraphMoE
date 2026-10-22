import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Embed import DataEmbedding
from masking import TriangularCausalMask
from dynamic_graph_learningv3 import GraphLearning

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        if adj.dim() == 3:
            adj = adj[0]
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

class self_attention(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(self_attention, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention(attention_dropout=0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out, attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class cross_attention(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(cross_attention, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        output, attn_weights = self.inner_attention(queries, keys, values, attn_mask)

        output = output.view(B, L, -1)
        output = self.out_projection(output)

        return output, attn_weights

class CrossAttention_Block(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, activation: str = "relu"):
        super(CrossAttention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.cross_attn = cross_attention(attention=FullAttention(attention_dropout=dropout, output_attention=True), d_model=d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_weights = self.cross_attn(query, key, value, attn_mask)
        query = query + self.dropout(attn_output)

        x = self.norm1(query)

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)

        output = self.norm2(x + y)

        return output, attn_weights

class Predict(nn.Module):
    def __init__(self, individual, c_out, seq_len, pred_len, dropout):
        super().__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len, pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len, pred_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:, i, :])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out, dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out

class GraphBlock(nn.Module):
    def __init__(self, d_out, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len):
        super().__init__()

        self.start_conv = nn.Conv2d(1, conv_channel, (1, seq_len - d_out + 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, d_out))
        self.linear = nn.Linear(seq_len, seq_len)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x, adj):
        out = x.unsqueeze(1)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out, adj))
        out = self.end_conv(out).squeeze().transpose(-1, -2)
        out = self.linear(out)

        return self.norm(x + out)

class Gating(nn.Module):
    def __init__(self, hidden_size, topk, expert_num):
        super().__init__()
        self.topk = topk
        self.expert_num = expert_num
        self.gate = nn.Linear(hidden_size, expert_num)

    def forward(self, x):
        logits = self.gate(x)
        logits_topk, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_log = zeros.scatter(dim=-1, index=indices, src=logits_topk)
        sparse_log = F.softmax(sparse_log, dim=-1)
        gate_log = logits.view(-1, self.expert_num)

        return sparse_log, indices, gate_log

class TimeGating(nn.Module):
    def __init__(self, hidden_size, topk, expert_num):
        super().__init__()
        self.topk = topk
        self.expert_num = expert_num
        self.gate = nn.Linear(hidden_size, expert_num)

    def FFT_for_Period(self, x, k=2):
        x = x.float()
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    def gate_logits(self, x):
        logits = self.gate(x)
        logits_topk, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_log = zeros.scatter(dim=-1, index=indices, src=logits_topk)
        sparse_log = F.softmax(sparse_log, dim=-1)
        gate_log = logits.view(-1, self.expert_num)
        return sparse_log, indices, gate_log

    def forward(self, x):
        B, N = x.shape[0], x.shape[2]
        scale_list, scale_weight = self.FFT_for_Period(x, self.topk)
        sparse_log, indices, gate_log = [], [], []
        for i in range(self.topk):
            scale = scale_list[i]
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

            sl, ind, gl = self.gate_logits(out)
            sl = sl.reshape(B, -1, self.expert_num)
            ind = ind.reshape(B, -1, self.expert_num)
            gl = gl.reshape(B, -1, self.expert_num)
            sparse_log.append(sl), indices.append(ind), gate_log.append(gl)

        return sparse_log, indices, gate_log

class EmbedGating(nn.Module):
    def __init__(self, hidden_size, topk, expert_num):
        super().__init__()
        self.topk = topk
        self.expert_num = expert_num
        self.gate = nn.Linear(hidden_size, expert_num)

    def forward(self, emb):
        logits = self.gate(emb)
        logits_topk, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_log = zeros.scatter(dim=-1, index=indices, src=logits_topk)
        sparse_log = F.softmax(sparse_log, dim=-1)
        gate_log = logits.view(-1, self.expert_num)

        return sparse_log, indices, gate_log

class StaGraphExpert(nn.Module):
    def __init__(self, d_out, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len, node_dim):
        super().__init__()

        self.nodevec1 = nn.Parameter(torch.randn(d_out, node_dim))
        self.nodevec2 = nn.Parameter(torch.randn(d_out, node_dim))
        self.expert = GraphBlock(d_out, conv_channel, skip_channel,
                                 gcn_depth, dropout, propalpha, seq_len)

    def init_weights(self):
        nn.init.xavier_uniform_(self.nodevec1)
        nn.init.xavier_uniform_(self.nodevec2)

    def forward(self, x):
        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2.T)), dim=1)
        return self.expert(x, adj) + x

class DyGraphExpert(nn.Module):
    def __init__(self, d_out, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len):
        super().__init__()

        self.expert = GraphBlock(d_out, conv_channel, skip_channel,
                                 gcn_depth, dropout, propalpha, seq_len)

    def forward(self, x, dy_adj):
        return self.expert(x, dy_adj) + x

class MoELayer(nn.Module):
    def __init__(self, d_out, d_ff, d_k, conv_channel, skip_channel, num_heads, 
                 num_dexperts, num_sexperts, topk,
                 gcn_depth, dropout, propalpha, sigma, seq_len, num_nodes, node_dim,
                 graph_type='adaptive', static_adj=None):
        super().__init__()

        self.num_sexperts = num_sexperts
        self.num_dexperts = num_dexperts
        self.topk = topk
        self.graph_type = graph_type
        self.static_adj = static_adj

        self.static_experts = nn.ModuleList([
            GraphBlock(d_out, conv_channel, skip_channel,
                       gcn_depth, dropout, propalpha, seq_len) 
            for _ in range(num_sexperts)
        ])
        self.dynamic_experts = nn.ModuleList([
            GraphBlock(d_out, conv_channel, skip_channel,
                       gcn_depth, dropout, propalpha, seq_len) 
            for _ in range(num_dexperts)
        ])

        if graph_type == 'adaptive':
            self.graph = GraphLearning(d_k, num_dexperts, num_nodes, node_dim, 
                                       seq_len, dropout, propalpha, sigma)
        else:
            self.graph = None

        self.static_gate = Gating(seq_len, 1, num_sexperts)
        self.dynamic_gate = EmbedGating(d_k * num_dexperts, topk, num_dexperts)
        self.cross_att = CrossAttention_Block(seq_len, d_ff, num_heads, dropout, activation="gelu")

        self.streams_static = [torch.cuda.Stream() for _ in range(num_sexperts)]
        self.streams_dynamic = [torch.cuda.Stream() for _ in range(num_dexperts)]

    def forward(self, x):
        batch, nodes, seq = x.size()

        if self.graph_type == 'adaptive':
            st_adj, dy_adj, emb = self.graph(x)
            dy_sparse_log, dy_indices, dy_gate_log = self.dynamic_gate(emb)
        else:
            st_adj = self.static_adj.unsqueeze(0).expand(batch, -1, -1)  # (batch, nodes, nodes)
            dy_adj = st_adj
            emb = None
            dy_sparse_log, dy_indices, dy_gate_log = None, None, None

        sparse_log, indices, gate_log = self.static_gate(x)

        default_stream = torch.cuda.current_stream()

        static_outputs_list = [None] * self.num_sexperts
        for i, expert in enumerate(self.static_experts):
            stream = self.streams_static[i]
            stream.wait_stream(default_stream)
            with torch.cuda.stream(stream):
                if self.graph_type == 'adaptive':
                    adj_for_static = st_adj
                else:
                    adj_for_static = self.static_adj  # 2D
                static_outputs_list[i] = expert(x, adj_for_static)
            default_stream.wait_stream(stream)

        dynamic_outputs_list = [None] * self.num_dexperts
        if self.graph_type == 'adaptive':
            adj_for_experts = dy_adj
        else:
            adj_for_experts = self.static_adj

        for i, expert in enumerate(self.dynamic_experts):
            stream = self.streams_dynamic[i]
            stream.wait_stream(default_stream)
            with torch.cuda.stream(stream):
                if self.graph_type == 'adaptive':
                    adj = adj_for_experts[i] if isinstance(adj_for_experts, list) else adj_for_experts
                else:
                    adj = adj_for_experts
                dynamic_outputs_list[i] = expert(x, adj)
            default_stream.wait_stream(stream)

        static_output_stack = torch.stack(static_outputs_list, dim=2)
        dynamic_output_stack = torch.stack(dynamic_outputs_list, dim=2)

        mask_static = torch.zeros((batch, nodes, self.num_sexperts), dtype=torch.bool, device=x.device)
        mask_static.scatter_(2, indices, True)
        gate_scores_static = sparse_log * mask_static.float()

        if self.graph_type == 'adaptive':
            mask_dynamic = torch.zeros((batch, nodes, self.num_dexperts), dtype=torch.bool, device=x.device)
            mask_dynamic.scatter_(2, dy_indices, True)
            gate_scores_dynamic = dy_sparse_log * mask_dynamic.float()
        else:
            gate_scores_dynamic = torch.ones((batch, nodes, self.num_dexperts), device=x.device) / self.num_dexperts

        static_final = torch.einsum('bnes, bne -> bns', static_output_stack, gate_scores_static)
        dynamic_final = torch.einsum('bnes, bne -> bns', dynamic_output_stack, gate_scores_dynamic)

        output, attn_weights = self.cross_att(dynamic_final, static_final, static_final)

        # 保存 debug 信息（静态图模式下 emb 为 None）
        self.debug_data = (st_adj, dy_adj, emb, dy_indices, indices)
        self.layer_out = x + output

        return x + output

class GraphMoE(nn.Module):
    def __init__(self, d_ff, d_k, conv_channel, skip_channel, num_heads, 
                 num_dexperts, num_sexperts, topk,
                 gcn_depth, dropout, propalpha, sigma, seq_len, pred_len, 
                 num_nodes, node_dim, moe_layers, individual,
                 graph_type='adaptive', static_adj=None):
        super().__init__()

        self.pred_len = pred_len

        self.embedding = DataEmbedding(num_nodes, num_nodes, dropout)

        self.moe_layer = nn.ModuleList([
            MoELayer(num_nodes, d_ff, d_k, conv_channel, skip_channel, num_heads,
                     num_dexperts, num_sexperts, topk,
                     gcn_depth, dropout, propalpha, sigma, seq_len, num_nodes, node_dim,
                     graph_type=graph_type, static_adj=static_adj)
            for _ in range(moe_layers)
        ])

        self.predictor = Predict(individual, num_nodes, seq_len, pred_len, dropout)
        self.key_fc = nn.Linear(num_nodes, 1)

    def forward(self, x_enc):
        enc_out = self.embedding(x_enc, None).transpose(1, 2)
        for layer in self.moe_layer:
            enc_out = layer(enc_out)

        dec_out = self.predictor(enc_out).transpose(1, 2)

        key_preds = self.key_fc(dec_out.squeeze())

        return key_preds.squeeze()

class GraphMoEAutoencoder(nn.Module):
    def __init__(self, d_ff, d_k, conv_channel, skip_channel, num_heads, 
                 num_dexperts, num_sexperts, topk,
                 gcn_depth, dropout, propalpha, sigma, seq_len, pred_len, 
                 num_nodes, node_dim, moe_layers, individual, pretrain=False,
                 graph_type='adaptive', static_adj=None):
        super().__init__()

        self.pred_len = pred_len
        self.pretrain = pretrain

        self.embedding = DataEmbedding(num_nodes, num_nodes, dropout)

        self.encoder = nn.ModuleList([
            MoELayer(num_nodes, d_ff, d_k, conv_channel, skip_channel, num_heads,
                     num_dexperts, num_sexperts, topk,
                     gcn_depth, dropout, propalpha, sigma, seq_len, num_nodes, node_dim,
                     graph_type=graph_type, static_adj=static_adj)
            for _ in range(moe_layers)
        ])

        self.decoder1 = nn.ModuleList([
            MoELayer(num_nodes, d_ff, d_k, conv_channel, skip_channel, num_heads,
                     num_dexperts, num_sexperts, topk,
                     gcn_depth, dropout, propalpha, sigma, seq_len, num_nodes, node_dim,
                     graph_type=graph_type, static_adj=static_adj)
            for _ in range(moe_layers)
        ])

        self.decoder2 = nn.ModuleList([
            MoELayer(num_nodes, d_ff, d_k, conv_channel, skip_channel, num_heads,
                     num_dexperts, num_sexperts, topk,
                     gcn_depth, dropout, propalpha, sigma, seq_len, num_nodes, node_dim,
                     graph_type=graph_type, static_adj=static_adj)
            for _ in range(moe_layers)
        ])

        self.reconstructor = Predict(individual, num_nodes, seq_len, seq_len, dropout)
        self.predictor = Predict(individual, num_nodes, seq_len, pred_len, dropout)

        self.key_fc = nn.Linear(num_nodes, 1)

    def forward(self, x_enc):
        enc_out = self.embedding(x_enc, None).transpose(-1, -2)
        for layer in self.encoder:
            enc_out = layer(enc_out)

        dec_out = enc_out

        if self.pretrain:
            for layer in self.decoder1:
                dec_out = layer(dec_out)
            return self.reconstructor(dec_out).transpose(1, 2)

        for layer in self.decoder2:
            dec_out = layer(dec_out)

        dec_out = self.predictor(enc_out).transpose(1, 2)

        key_preds = self.key_fc(dec_out)

        return key_preds.squeeze()

if __name__ == '__main__':
    params = {
        "d_ff": 128,
        "d_k": 32,
        "conv_channel": 32,
        "skip_channel": 32,
        "num_heads": 8,
        "num_dexperts": 12,
        "num_sexperts": 8,
        "topk": 6,
        "gcn_depth": 2,
        "dropout": 0.2,
        "propalpha": 0.1,
        "sigma": 0.1,
        "seq_len": 96,
        "pred_len": 1,
        "num_nodes": 30,
        "node_dim": 40,
        "moe_layers": 1,
        "individual": False,
        "pretrain": True
    }

    model = GraphMoEAutoencoder(**params, graph_type='adaptive', static_adj=None)

    batch_size = 32
    x = torch.randn(batch_size, params["seq_len"], params["num_nodes"])
    pm = sum([p.nelement() for p in model.parameters()])

    with torch.no_grad():
        model.pretrain = True
        output = model(x)

    print('parameters: ')
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")