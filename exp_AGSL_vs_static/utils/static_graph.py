import torch
import torch.nn.functional as F

def compute_pearson_correlation(train_data: torch.Tensor) -> torch.Tensor:
    num_nodes = train_data.shape[2]
    flat = train_data.view(train_data.shape[0], -1)
    node_means = flat.mean(dim=0) + 1e-8
    node_std = flat.std(dim=0) + 1e-8
    norm_flat = (flat - node_means) / node_std
    node_features = norm_flat.view(train_data.shape[0], num_nodes, -1).mean(dim=-1)
    corr = torch.corrcoef(node_features.T)
    return F.softmax(torch.abs(corr), dim=-1)


def compute_knn_graph(train_data: torch.Tensor, k: int = 10) -> torch.Tensor:
    num_nodes = train_data.shape[2]
    node_features = train_data.mean(dim=(0, 1))
    dist = torch.cdist(node_features.unsqueeze(1), node_features.unsqueeze(1))

    effective_k = min(k, num_nodes - 1)
    if effective_k < 1:
        return torch.eye(num_nodes)

    knn_indices = dist.topk(k=effective_k + 1, largest=False).indices[:, 1:]
    adj = torch.zeros(num_nodes, num_nodes)
    adj.scatter_(1, knn_indices, 1.0)
    adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
    return adj