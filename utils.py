import random
import os

import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def mysplit(X, y, fold, n_split, seed):
    skf = StratifiedKFold(n_splits=n_split, random_state=seed, shuffle=True)
    for idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        if idx == fold:
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
    return X_train, X_valid, y_train, y_valid