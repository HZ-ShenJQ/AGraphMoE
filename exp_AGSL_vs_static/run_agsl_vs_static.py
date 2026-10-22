import sys
import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
from thop import profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SsAGraphMoE'))

from PD_GraphMoEv3_for_q1 import params, create_logger_and_model_save_path
from model.SsGraphMoE_static import GraphMoE
from utils.ZSDataset import ZSDataset
from utils.train import train, val
from utils.static_graph import compute_pearson_correlation, compute_knn_graph


def run_experiment(graph_type, device, base_args, labeled_ratio, seed=0):
    args = copy.deepcopy(base_args)
    args.random_seed = seed
    args.labeled_ratio = labeled_ratio
    args.unlabeled_ratio = 0.3

    train_data = ZSDataset(
        args.data, 'train_labeled', args.zs_train, args.zs_test,
        args.seq_len, args.stride, args.scale, args.random_seed,
        train_ratio=args.train_ratio, test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio, labeled_ratio=args.labeled_ratio,
        unlabeled_ratio=args.unlabeled_ratio
    )

    test_data = ZSDataset(
        args.data, 'test', args.zs_train, args.zs_test,
        args.seq_len, args.stride, args.scale, args.random_seed,
        train_ratio=args.train_ratio, test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio, labeled_ratio=args.labeled_ratio,
        unlabeled_ratio=args.unlabeled_ratio
    )

    train_loader = DataLoader(train_data, batch_size=args.pretrain_batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.fine_batch, shuffle=False)

    static_adj = None
    if graph_type in ['corr', 'knn']:
        raw_data = train_data.x
        if graph_type == 'corr':
            static_adj = compute_pearson_correlation(raw_data)
        else:  # knn
            static_adj = compute_knn_graph(raw_data, k=args.knn_k)
        static_adj = static_adj.to(device)
        print(f"{graph_type} static_adj shape: {static_adj.shape}")

    model = GraphMoE(
        args.d_ff, args.d_k, args.conv_channel, args.skip_channel,
        args.num_heads, args.num_dexperts, args.num_sexperts, args.topk,
        args.gcn_depth, args.dropout, args.propalpha, args.sigma, args.seq_len,
        args.pred_len, args.num_nodes, args.node_dim, args.moe_layers,
        args.individual,
        graph_type=graph_type, static_adj=static_adj
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(1, args.seq_len, args.num_nodes).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    save_dir = f'ModelSave/AGSL_vs_Static/{graph_type}/ratio_{int(labeled_ratio*100)}/seed_{seed}'
    args.save_main_path = save_dir
    logger, model_path, main_path = create_logger_and_model_save_path(
        args.save_main_path, args.random_seed
    )
    args.model_save_path = model_path
    args.main_path = main_path

    start_time = time.time()
    train(model, args, train_loader, test_loader, output_print=True)
    train_time = time.time() - start_time

    model.eval()
    latency_list = []
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            latency_list.append((time.time() - start) * 1000)
    avg_latency = sum(latency_list) / len(latency_list)

    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    _, _, _, mse, mae, rmse, r2 = val(model, test_loader, device)

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'Params': total_params,
        'FLOPs': flops,
        'Latency': avg_latency,
        'TrainTime': train_time
    }


def main(target_ratio):
    original_argv = sys.argv
    sys.argv = [original_argv[0]]
    base_args = params()
    sys.argv = original_argv

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    base_args.data = ''
    base_args.knn_k = 10

    graph_types = ['corr', 'knn']
    results = {g: [] for g in graph_types}

    for graph_type in graph_types:
        print(f"\n{'=' * 40}")
        print(f"Running: {graph_type} for ratio={target_ratio * 100:.0f}%")
        print(f"{'=' * 40}")
        for seed in range(3):
            print(f"  Seed {seed}...")
            res = run_experiment(graph_type, device, base_args, labeled_ratio=target_ratio, seed=seed)
            results[graph_type].append(res)
            print(f"    MSE: {res['MSE']:.4f}, MAE: {res['MAE']:.4f}, R2: {res['R2']:.4f}")

    print(f"\n--- Summary for ratio {target_ratio * 100:.0f}% ---")
    for g in graph_types:
        mse_list = [r['MSE'] for r in results[g]]
        mae_list = [r['MAE'] for r in results[g]]
        r2_list = [r['R2'] for r in results[g]]
        print(f"{g.upper():6} MSE: {np.mean(mse_list):.4f}±{np.std(mse_list):.4f}  "
              f"MAE: {np.mean(mae_list):.4f}±{np.std(mae_list):.4f}  "
              f"R2:  {np.mean(r2_list):.4f}±{np.std(r2_list):.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ratio = float(sys.argv[1])
        main(ratio)
    else:
        print("Usage: python run_agsl_vs_static.py <ratio>  e.g., python run_agsl_vs_static.py 0.1")
        sys.exit(1)