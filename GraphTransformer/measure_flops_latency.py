import argparse
import os
import time

import numpy as np
import torch
from thop import profile, clever_format

from model.model_gtn import GTN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', default=26, type=int,
                        help='HLT=26, PD=15')
    parser.add_argument('--seq_len', default=96, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pt)')
    return parser.parse_args()


def measure_latency_batch1(model, dummy_input, target_dummy, device,
                           warmup=50, repeats=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input, target_dummy, eval=True)

    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(repeats):
                starter.record()
                _ = model(dummy_input, target_dummy, eval=True)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return float(np.mean(timings)), float(np.sum(timings))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(repeats):
                start = time.perf_counter()
                _ = model(dummy_input, target_dummy, eval=True)
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

    model = GTN(
        num_channels=args.num_channels,
        w_in=args.seq_len,
        w_out=args.hidden_dim,
        num_nodes=args.num_nodes,
        num_layers=args.num_layers,
    ).to(device)

    if not os.path.exists(args.model_path):
        print(f"[Error] Checkpoint not found: {args.model_path}")
        exit(1)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[OK] Loaded checkpoint: {args.model_path}")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Trainable parameters: {n_params / 1e6:.3f} M ({n_params:,})")

    dummy_input = torch.randn(1, args.num_nodes, args.seq_len).to(device)
    target_dummy = torch.randn(1, 1).to(device)

    try:
        flops, params_thop = profile(
            model,
            inputs=(dummy_input, target_dummy, True),
            verbose=False,
        )
        flops_str, params_thop_str = clever_format([flops, params_thop], "%.3f")
        print(f"[OK] FLOPs (batch=1, per-sample): {flops_str}")
        print(f"[OK] THOP parameter count: {params_thop_str}")
    except Exception as e:
        print(f"[Warning] FLOPs computation failed: {e}")

    avg_latency, total_time_ms = measure_latency_batch1(
        model, dummy_input, target_dummy, device, warmup=50, repeats=100
    )
    print(f"[OK] Average inference latency (batch=1, per-sample): "
          f"{avg_latency:.4f} ms")
    print(f"[OK] Total inference time over 100 runs: "
          f"{total_time_ms / 1000.0:.2f} s = {total_time_ms / 60000.0:.2f} min")