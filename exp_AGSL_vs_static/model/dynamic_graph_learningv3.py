import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class StaticGraph(nn.Module):

    def __init__(self, num_nodes, embedding_dim, subgraph_size, alpha, dropout, ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.emb1 = nn.Parameter(torch.Tensor(num_nodes, embedding_dim))
        self.emb2 = nn.Parameter(torch.Tensor(num_nodes, embedding_dim))
        self.node_idx = torch.arange(self.num_nodes)
        self.subgraph_size = subgraph_size

        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        product_matrix = torch.softmax(torch.matmul(self.emb1, self.emb2.T), dim=-1)

        return torch.softmax(product_matrix, dim=1)


class GraphLearning(nn.Module):

    def __init__(self, d_k, head_experts, num_nodes, static_emb_dim, seq_len, dropout, alpha, sigma):
        super().__init__()

        self.n = num_nodes
        self.h = head_experts

        self.d_k = d_k
        self.d_model = head_experts * d_k

        self.p = dropout
        self.alpha = alpha
        self.sigma = sigma

        self.fc_xh = nn.Linear(seq_len, self.d_model)

        self.Wr = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Ur = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Br = nn.Parameter(torch.Tensor(d_k))

        self.Wz = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Uz = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Bz = nn.Parameter(torch.Tensor(d_k))

        self.Wh = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Uh = nn.Parameter(torch.Tensor(d_k, d_k))
        self.Bh = nn.Parameter(torch.Tensor(d_k))

        self.fc_q = nn.Linear(self.d_model, self.d_model)
        self.fc_k = nn.Linear(self.d_model, self.d_model)
        self.head_weights = nn.Parameter(torch.ones(self.h) / self.h)

        self.fc = nn.Linear(self.d_model, self.d_model)

        self.norm = nn.LayerNorm(self.d_model)

        self.dynamic_embedding = nn.Parameter(torch.Tensor(head_experts, num_nodes, d_k))
        self.static_vec1 = nn.Parameter(torch.Tensor(num_nodes, static_emb_dim))
        self.static_vec2 = nn.Parameter(torch.Tensor(num_nodes, static_emb_dim))
        self.weighted_init()

    def static_graph(self):
        temp = torch.matmul(self.static_vec1, self.static_vec2.T) - torch.matmul(self.static_vec2, self.static_vec1.T)
        product_matrix = F.gelu(self.alpha * temp)

        return torch.softmax(product_matrix, dim=1)

    def weighted_init(self):
        xavier_uniform_(self.fc_xh.weight)
        xavier_uniform_(self.dynamic_embedding.data)
        xavier_uniform_(self.static_vec1.data)
        xavier_uniform_(self.static_vec2.data)

        for param in [self.Wr, self.Ur, self.Wz, self.Uz, self.Wh, self.Uh]:
            xavier_uniform_(param)

        for bias in [self.Br, self.Bz, self.Bh]:
            constant_(bias, 0.0)

    def recurrent_embedding(self, h):  # inspired by GRU
        # (b,h,n,d_k) → (b, n, d_k*h)
        bs = h.shape[0]
        M = self.dynamic_embedding  # (h,n,d_k)
        # r ^ T = σ(Wr*M + Ur*X^T + br)
        r = torch.sigmoid(torch.matmul(M, self.Wr) + torch.matmul(h, self.Ur) + self.Br)  # (b, h, n, d_k)
        # z^T = σ(Wz*M + Uz*X^T + bz)
        z = torch.sigmoid(torch.matmul(M, self.Wz) + torch.matmul(h, self.Uz) + self.Bz)  # (b, h, n, d_k)
        # ĥ^T = tanh(Wh*X^T + Uh*(r^T⊙M) + bh)
        h_ = torch.tanh(torch.matmul(h, self.Wh) + torch.matmul(r * M, self.Uh) + self.Bh)  # (b, h, n, d_k)
        # h^T：h^T = (1-z^T)⊙M + z^T⊙ĥ^T
        h = (1 - z) * h + z * h_ + self.sigma * M

        return h.transpose(1, 2).reshape(bs, self.n, -1)  # (b,experts, n, d_k)

    def dynamic_expert_graph(self, x):  # inspired by multi-head attention

        st_adj = self.static_graph()

        h = self.fc_xh(x)
        bs = h.shape[0]
        h_norm = self.norm(h)  # (b,n,d_k*h)
        rec_emb = self.recurrent_embedding(h_norm.reshape(bs, self.n, self.h, self.d_k).transpose(1, 2))

        # Q,K (b,n,d_k*h) d_model = d_k*h
        Q = self.fc_q(rec_emb)
        K = self.fc_k(rec_emb)
        # (b,n,d_k*h)->(b,n,heads,d_k)->(b,heads,n,d_k)
        Q = Q.view(bs, self.n, self.h, self.d_k).transpose(1, 2)
        K = K.view(bs, self.n, self.h, self.d_k).transpose(1, 2)
        # Ã^(h) = <Q, K^T>/sqrt(d_k) → (b, h, n, n)
        score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        score_dp = F.dropout(score, self.p)

        att_s = (score_dp * self.head_weights.view(1, self.h, 1, 1) + score)  # (b,e,n,n)

        s_h = self.fc(h).view(bs, self.n, self.h, self.d_k).transpose(1, 2)
        skip = F.layer_norm((s_h @ s_h.transpose(-1, -2)), [self.n])

        mix = att_s + skip
        dy_adj = F.softmax(F.gelu(mix.mean(0)), dim=-1)

        return st_adj, dy_adj, rec_emb

    def forward(self, x):
        return self.dynamic_expert_graph(x)  # (heads,n,n)


if __name__ == '__main__':
    GraphLearning = GraphLearning(d_k=32, num_nodes=30, head_experts=6, static_emb_dim=40, seq_len=96, dropout=0.5, alpha=3, sigma=0.1)
    x = torch.rand([32, 30, 96])
    adj, emb = GraphLearning(x)
    # print(adj)
    print(adj.size())
