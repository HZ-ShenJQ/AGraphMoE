import argparse, datetime, os, logging, optuna
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.ZSDataset import ZSDataset
from utils.train import train, val
from model.TMoE_P import TMoE_P


def create_logger_and_model_save_path(save_file_path, random_seed, trial_id=None):
    datestr = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if trial_id is not None:
        main_path = os.path.join(save_file_path, f'trial_{trial_id}_seed{random_seed}_{datestr}')
    else:
        main_path = os.path.join(save_file_path, f'best_{datestr}_seed{random_seed}')

    os.makedirs(main_path, exist_ok=True)

    model_save_path = os.path.join(main_path, 'model.pt')
    log_file_path = f"{main_path}/logs.log"

    logger_name = f'Trial{trial_id}_Seed{random_seed}_Time{datestr}' if trial_id else f'Seed{random_seed}_Time{datestr}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y-%H:%M:%S"
    ))
    logger.addHandler(file_handler)

    return logger, model_save_path, main_path


def run(args, logger, trial=None):
    if len(args.gpu_ids) > 0:
        device = f'cuda:{args.gpu_ids[0]}'
    else:
        device = 'cpu'

    logger.info(f"Train beginning random seed: {args.random_seed}")

    trainData = ZSDataset(
        args.data, 'train_labeled', args.zs_train, args.zs_test,
        args.seq_len, args.stride, args.scale, args.random_seed,
        train_ratio=args.train_ratio, test_ratio=args.test_ratio, valid_ratio=args.valid_ratio, labeled_ratio=args.labeled_ratio, unlabeled_ratio=args.unlabeled_ratio, )
    testData = ZSDataset(
        args.data, 'test', args.zs_train, args.zs_test,
        args.seq_len, args.stride, args.scale, args.random_seed,
        train_ratio=args.train_ratio, test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio, labeled_ratio=args.labeled_ratio,
        unlabeled_ratio=args.unlabeled_ratio,
        scaler=trainData.scaler,
        label_scaler=trainData.label_scaler,
    )

    train_loader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=False)

    model = TMoE_P(
        num_nodes=args.num_nodes,
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        num_experts=4,
        top_k=2
    )

    logger.info(f'Model training parameters setting: {vars(args)}')
    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info(f'Number of model parameters: {nParams}')

    train(model, args, train_loader, test_loader, output_print=False)

    model.load_state_dict(torch.load(args.model_save_path))

    test_loss, pred, truth, mse, mae, rmse, r2 = val(model, test_loader, device)

    try:

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(truth, torch.Tensor):
            truth = truth.detach().cpu().numpy()

        np.save(os.path.join(args.main_path, 'pred.npy'), pred)
        np.save(os.path.join(args.main_path, 'truth.npy'), truth)
        logger.info(f"Predictions and Truth saved to {args.main_path}")

        plt.figure(figsize=(15, 6))

        pred_flat = pred.reshape(-1)
        truth_flat = truth.reshape(-1)

        plot_len = min(len(truth_flat), 500)

        plt.plot(truth_flat[:plot_len], label='Ground Truth', alpha=0.7, color='blue')
        plt.plot(pred_flat[:plot_len], label='Prediction', alpha=0.7, color='red', linestyle='--')

        plt.xlabel('Sample point')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        plot_path = os.path.join(args.main_path, 'pred_vs_truth.png')
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Comparison plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error saving plots or data: {e}")
        print(f"Error saving plots or data: {e}")

    logger.info(f'Best model save path: {args.model_save_path}')
    logger.info(f'Random Seed: {args.random_seed}')
    logger.info(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2: {r2}')
    logger.info('')
    print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2: {r2}')

    return mse, mae, rmse, r2


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='location of the data file', default='../Data/PD')
    parser.add_argument('--zs_train', type=str, help='path to save the final model', default=['k1', 'k2', 'k3','k4'])
    parser.add_argument('--zs_test', type=str, help='path to save the final model', default='u1')
    parser.add_argument('--save_main_path', type=str, help='path to save the final model', default='ModelSave/PD/TMoE_P')
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--test_ratio', type=float, default=1)
    parser.add_argument('--valid_ratio', type=float, default=0)
    parser.add_argument('--labeled_ratio', type=float, default=0.3)
    parser.add_argument('--unlabeled_ratio', type=float, default=0.2)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)

    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--epoch', help='train epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--decay', type=float, help='weight decay rate', default=1e-5)

    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--embed_dim', default=10, type=int)
    parser.add_argument('--num_nodes', type=int, help='number of nodes/variables', default=15)

    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seq_len', type=int, help='input sequence length', default=96)

    parser.add_argument('--default_graph', default=True)
    parser.add_argument('--random_seed', type=int, help='random_seed', default=0)

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[1])
    parser.add_argument('--use_optuna', default=False, help='whether to use optuna for hyperparameter search')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    ratios = [0.1, 0.15, 0.2, 0.3]

    for ratio in ratios:

        args = params()
        args.labeled_ratio = ratio
        print(f"\n{'=' * 40}")
        print(f"Starting Experiment: labeled_ratio = {int(ratio * 100)}%")
        print(f"{'=' * 40}")

        mse_list, mae_list, rmse_list, r2_list = [], [], [], []

        for seed in range(5):
            args.random_seed = seed

            logger, model_save_path, main_path = create_logger_and_model_save_path(
                args.save_main_path, args.random_seed
            )
            # main_path = os.path.join(args.save_main_path, f'Ratio{int(ratio * 100)}_Seed{seed}')

            args.model_save_path = model_save_path
            args.main_path = main_path

            macro_mse, macro_mae, macro_rmse, macro_r2 = run(args, logger)

            mse_list.append(macro_mse)
            mae_list.append(macro_mae)
            rmse_list.append(macro_rmse)
            r2_list.append(macro_r2)

        print(f"\n--- Summary for labeled_ratio: {int(ratio * 100)}% ---")
        print(f"MSE:  {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
        print(f"MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
        print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
        print(f"R2:   {np.mean(r2_list):.4f}  ± {np.std(r2_list):.4f}")

'''
    运行命令示例（Optuna模式）：
    python HLT-ZS.py --gpu_ids 1

'''
