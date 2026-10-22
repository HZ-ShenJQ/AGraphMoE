import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


class ZSDataset(Dataset):
    def __init__(self, data_path, mode, zs_train, zs_test, slide_win, slide_stride,
                 scale, random_seed=42, horizon=0,
                 train_ratio=None, test_ratio=None, valid_ratio=None,
                 labeled_ratio=None, unlabeled_ratio=None,
                 scaler=None, label_scaler=None):

        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.mode = mode
        self.zs_test = zs_test
        self.w = slide_win
        self.s = slide_stride
        self.scale = scale
        self.horizon = horizon

        self.x_ = None
        self.y_ = None

        # Load and process data
        if mode in ['train', 'train_labeled', 'train_unlabeled']:
            self.load_train_data(data_path, zs_train, train_ratio, labeled_ratio, unlabeled_ratio)
        else:
            self.load_test_or_valid_data(data_path, zs_test, test_ratio, valid_ratio)

        self.scaler = None
        self.label_scaler = None
        if self.scale != 0:
            if scaler is not None and label_scaler is not None:
                self.scaler = scaler
                self.label_scaler = label_scaler
            else:
                self.init_scaler()
            self.normalize_subset()
        else:
            self.x = self.x_
            self.y = self.y_

    def load_train_data(self, data_path, zs_train, train_ratio, labeled_ratio, unlabeled_ratio):

        x_all, y_all, c_all = [], [], []

        for file in os.listdir(data_path):
            if not file.endswith('.csv') or file == "dA.csv":
                continue

            fname = os.path.splitext(file)[0]
            if fname not in zs_train:
                continue

            df = pd.read_csv(os.path.join(data_path, file)).values
            data = torch.tensor(df, dtype=torch.float32)
            features, target = data[:, :-1], data[:, -1]

            x_proc, y_proc = self.process(features, target)

            match = re.findall(r'\d+', fname)

            cond_label = int(match[0]) - 1
            cond_tensor = torch.full((len(y_proc),), cond_label, dtype=torch.long)

            total = len(y_proc)
            perm = torch.randperm(total)
            keep_num = int(total * train_ratio)
            keep_idx = perm[:keep_num]

            x_proc, y_proc = x_proc[keep_idx], y_proc[keep_idx]
            cond_tensor = cond_tensor[keep_idx]

            x_all.append(x_proc), y_all.append(y_proc), c_all.append(cond_tensor)

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        c_all = torch.cat(c_all, dim=0)

        # labeled / unlabeled split
        total_num = len(y_all)
        perm = torch.randperm(total_num)

        labeled_num = int(total_num * labeled_ratio)
        unlabeled_num = int((total_num - labeled_num) * unlabeled_ratio)

        labeled_idx = perm[:labeled_num]
        unlabeled_idx = perm[labeled_num:labeled_num + unlabeled_num]

        if self.mode == 'train_labeled':
            self.x_, self.y_, self.c = x_all[labeled_idx], y_all[labeled_idx], c_all[labeled_idx]
        else:
            # train or train_unlabeled
            self.x_, self.y_, self.c = x_all[unlabeled_idx], y_all[unlabeled_idx], c_all[unlabeled_idx]

    # def load_train_data(self, data_path, zs_train, train_ratio, labeled_ratio, unlabeled_ratio):
    #
    #     x_list, y_list = [], []
    #
    #     for file in os.listdir(data_path):
    #
    #         if not file.endswith('.csv') or file == "dA.csv":
    #             continue
    #
    #         fname = os.path.splitext(file)[0]
    #         if fname not in zs_train:
    #             continue
    #
    #         df = pd.read_csv(os.path.join(data_path, file)).values
    #         data = torch.tensor(df, dtype=torch.float32)
    #
    #         features, target = data[:, :-1], data[:, -1]
    #         x_proc, y_proc = self.process(features, target)
    #
    #         total = len(y_proc)
    #         perm = torch.randperm(total)
    #         keep_num = int(total * train_ratio)
    #         keep_idx = perm[:keep_num]
    #
    #         x_proc = x_proc[keep_idx]
    #         y_proc = y_proc[keep_idx]
    #
    #         sub_total = len(y_proc)
    #         sub_perm = torch.randperm(sub_total)
    #
    #         labeled_num = int(sub_total * labeled_ratio)
    #         unlabeled_num = int(sub_total * unlabeled_ratio)
    #
    #         labeled_idx = sub_perm[:labeled_num]
    #         unlabeled_idx = sub_perm[labeled_num:labeled_num + unlabeled_num]
    #
    #         if self.mode == 'train_labeled':
    #             x_list.append(x_proc[labeled_idx])
    #             y_list.append(y_proc[labeled_idx])
    #
    #         else:  # train_unlabeled or train
    #             x_list.append(x_proc[unlabeled_idx])
    #             y_list.append(y_proc[unlabeled_idx])
    #
    #     self.x_ = torch.cat(x_list, dim=0)
    #     self.y_ = torch.cat(y_list, dim=0)

    def load_test_or_valid_data(self, data_path, zs_test, test_ratio, valid_ratio):

        for file in os.listdir(data_path):
            if os.path.splitext(file)[0] == zs_test:
                df = pd.read_csv(os.path.join(data_path, file)).values
                data = torch.tensor(df, dtype=torch.float32)
                features, target = data[:, :-1], data[:, -1]
                match = re.findall(r'\d+', file)
                x_proc, y_proc = self.process(features, target)
                break
        else:
            raise ValueError(f"未找到测试集文件: {zs_test}")

        total = len(y_proc)
        perm = torch.randperm(total) if self.mode == 'valid' else torch.arange(total)
        test_num = int(total * test_ratio)
        valid_num = int(total * valid_ratio) if valid_ratio is not None else 0

        cond_label = int(match[0]) - 1
        cond_tensor = torch.full((len(y_proc),), cond_label, dtype=torch.long)

        if self.mode == 'test':
            idx = perm[:test_num]
        else:  # valid
            idx = perm[test_num:test_num + valid_num]

        self.x_, self.y_, self.c = x_proc[idx], y_proc[idx], cond_tensor[idx]

    def process(self, features, target):

        x_list, y_list = [], []
        max_i = len(features) - self.horizon

        for i in range(self.w, max_i, self.s):
            x_list.append(features[i - self.w:i])
            y_list.append(target[i + self.horizon])

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x.contiguous(), y.contiguous()

    def init_scaler(self):

        if self.scale == 1:
            self.scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        elif self.scale == 2:
            self.scaler = MaxAbsScaler()
            self.label_scaler = MaxAbsScaler()
        elif self.scale == 3:
            self.scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()
        else:
            raise ValueError("scale 必须为 0/1/2/3")

        # Flatten features for fitting
        flat_x = self.x_.reshape(-1, self.x_.shape[-1]).numpy()
        self.scaler.fit(flat_x)

        flat_y = self.y_.reshape(-1, 1).numpy()
        self.label_scaler.fit(flat_y)

    def normalize_subset(self):

        x_shape = self.x_.shape
        flat_x = self.x_.reshape(-1, x_shape[-1]).numpy()
        x_norm = self.scaler.transform(flat_x)
        self.x = torch.tensor(x_norm, dtype=torch.float32).reshape(x_shape)

        flat_y = self.y_.reshape(-1, 1).numpy()
        y_norm = self.label_scaler.transform(flat_y).flatten()
        self.y = torch.tensor(y_norm, dtype=torch.float32)

    def inverse_scale(self, y_norm):
        if self.scale == 0:
            return y_norm

        is_tensor = isinstance(y_norm, torch.Tensor)
        y_np = y_norm.cpu().numpy() if is_tensor else y_norm

        flat = y_np.reshape(-1, 1)
        y_inv = self.label_scaler.inverse_transform(flat).reshape(y_np.shape)

        if is_tensor:
            return torch.tensor(y_inv, dtype=torch.float32, device=y_norm.device)
        return y_inv

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]
