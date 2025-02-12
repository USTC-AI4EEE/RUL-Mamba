import random
import torch
import shutil
import numpy as np
import time
import copy
import os
import argparse
import subprocess
import torch.backends.cudnn as cudnn
import glob
import logging
import sys
import yaml
import json


# 寻找现存最大的显卡编号
def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory', stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id = int(np.argmax(memory_available))
    return id, memory_available


# 设置random, numpy， torch（cpu和cuda）的seed
def set_seed(seed):
    """
    set seed of numpy and torch
    :param seed:
    :return:
    """
    if seed is None:
        seed = np.random.randint(1e6)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed
