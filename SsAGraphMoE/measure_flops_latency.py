import argparse
import os
import sys
import time

import numpy as np
import torch
from thop import profile, clever_format

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.SsGraphMoEv3 import GraphMoE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--conv_channel', type=int, default=32)
    parser.add_argument('--skip_channel', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_dexperts', type=int, default=6)
    parser.add_argument('--num_sexperts', type=int, default=12)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--gcn_depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1493)
    parser.add_argument('--propalpha', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=0.4664)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=15,
                        help='HLT=26, PD=15')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--moe_layers', type=int, default=3)
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pt)')
    return parser.parse_args()


def measure_latency_batch1(model, dummy_input, device, warmup=50, repeats=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(repeats):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return float(np.mean(timings)), float(np.sum(timings))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(repeats):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)
        return float(np.mean(timings)), float(np.sum(timings))


if __name__ == "__main__":
    args = get_args()

    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = GraphMoE(
        d_ff=args.d_ff,
        d_k=args.d_k,
        conv_channel=args.conv_channel,
        skip_channel=args.skip_channel,
        num_heads=args.num_heads,
        num_dexperts=args.num_dexperts,
        num_sexperts=args.num_sexperts,
        topk=args.topk,
        gcn_depth=args.gcn_depth,
        dropout=args.dropout,
        propalpha=args.propalpha,
        sigma=args.sigma,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_nodes=args.num_nodes,
        node_dim=args.node_dim,
        moe_layers=args.moe_layers,
        individual=args.individual,
    ).to(device)

    if not os.path.exists(args.model_path):
        print(f"[Error] Checkpoint not found: {args.model_path}")
        exit(1)

    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"[OK] Loaded checkpoint: {args.model_path}")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Trainable parameters: {n_params / 1e6:.3f} M ({n_params:,})")

    dummy_input = torch.randn(1, args.seq_len, args.num_nodes).to(device)

    try:
        flops, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_thop_str = clever_format([flops, params_thop], "%.3f")
        print(f"[OK] FLOPs (batch=1, per-sample): {flops_str}")
        print(f"[OK] THOP parameter count: {params_thop_str}")
    except Exception as e:
        print(f"[Warning] FLOPs computation failed: {e}")

    avg_latency, total_time_ms = measure_latency_batch1(
        model, dummy_input, device, warmup=50, repeats=100
    )
    print(f"[OK] Average inference latency (batch=1, per-sample): "
          f"{avg_latency:.4f} ms")
    print(f"[OK] Total inference time over 100 runs: "
          f"{total_time_ms / 1000.0:.2f} s = {total_time_ms / 60000.0:.2f} min")