import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='.')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    params = [
        ('num_dexperts', 'Number of Dynamic Experts', [2, 4, 6, 8, 10, 12]),
        ('node_dim', 'Node Embedding Dimension', [32, 64, 96, 128]),
        ('d_k', 'Mapping Dimension', [16, 32, 64, 128]),
        ('num_heads', 'Cross-Attention Heads', [4, 8, 12, 16]),
    ]

    chosen_values = {
        'num_dexperts': 6,
        'node_dim': 96,
        'd_k': 64,
        'num_heads': 12,
    }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    for col, label, values in params:
        data = []
        for v in values:
            subset = df[df['params_' + col] == v]['value'].dropna()
            data.append(subset)

        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(
            data,
            labels=values,
            patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black'),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='gray',
                            markersize=4, alpha=0.5),
        )

        chosen = chosen_values[col]
        if chosen in values:
            chosen_idx = values.index(chosen)
            bp['boxes'][chosen_idx].set_facecolor('#E64B35')
            bp['boxes'][chosen_idx].set_edgecolor('black')
            bp['boxes'][chosen_idx].set_linewidth(2.5)

            group_values = data[chosen_idx]
            if not group_values.empty:
                y_max = group_values.max()
                if np.isnan(y_max) or y_max == 0:
                    y_max = ax.get_ylim()[1] * 0.98
                else:
                    y_max = y_max * 1.02
                ax.text(
                    chosen_idx + 1,
                    y_max,
                    'Selected',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    color='red',
                    fontweight='bold',
                )

        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel('R²', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_title(f'Optuna Search Results: {label}', fontsize=16)

        plt.tight_layout()

        save_name = os.path.join(
            args.output_dir, f'optuna_results_boxplot_{col}.pdf'
        )
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")


if __name__ == '__main__':
    main()