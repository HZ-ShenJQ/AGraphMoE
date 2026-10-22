import torch
import torch.nn as nn
import torch.nn.functional as F


class TMoE_P(nn.Module):
    def __init__(self, num_nodes, embed_dim, seq_len, hidden_channels, out_channels=1,
                 num_experts=4, top_k=2, expert_hidden_dim=None):
        super(TMoE_P, self).__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        input_dim = num_nodes * seq_len

        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_channels

        if embed_dim is not None and embed_dim < input_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
            expert_input_dim = embed_dim
        else:
            self.input_proj = nn.Identity()
            expert_input_dim = input_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(expert_input_dim, num_experts)

    def forward(self, x):
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)
        x_proj = self.input_proj(x_flat)
        gate_logits = self.gate(x_proj)
        gate_weights = F.softmax(gate_logits, dim=-1)

        if self.top_k < self.num_experts:
            top_k_vals, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            mask = torch.zeros_like(gate_weights).scatter_(-1, top_k_indices, 1.0)
            gate_weights = gate_weights * mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-9)

        expert_outputs = torch.stack([expert(x_proj) for expert in self.experts], dim=1)
        output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs)
        output = output.squeeze(1)

        return output