import random
import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_directory(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)

def showCurve(args, points, figname):
    plt.figure(figsize = (8,6))
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')
    epochs_num = np.arange(points.shape[0])
    plt.yscale("log")
    plt.plot(epochs_num, points, linestyle='-', color='b', linewidth=2)
    cf = plt.gcf()
    cf.savefig(f'{args.log_path}/{figname}.png', format='png', bbox_inches='tight', dpi=600)
