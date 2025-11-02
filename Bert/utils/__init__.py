import logging
from .logger import *
from .eval import *
from .argparser import parse_args
from .misc import *
from .earlystopper import EarlyStopping
import torch
import torch.nn.functional as F
import os
import shutil

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    if is_main_process:
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, f"model_best.pth"))

def create_save_path(args):

    ans_str = f"_{args.model}_{args.loss}_{args.batch_size}"

    if args.loss == "FLSD":
        ans_str += f"_gamma={args.gamma}"
        return ans_str
    
    if "focal_loss" in args.loss or "FL" in args.loss:
        ans_str += f"_gamma={args.gamma}"
    
    if "LS" in args.loss:
        ans_str += f"_alpha={args.alpha}"

    if "MDCA" in args.loss:
        ans_str += f"_beta={args.beta}"
        return ans_str

    if "DCA" in args.loss or "MMCE" in args.loss:
        ans_str += f"_beta={args.beta}"

    return ans_str





''' ************************************* For distributed training ****************************************'''
import io
import os
import time
from collections import defaultdict, deque
import datetime
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

# from torch import inf




def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
#     """
#     Compute per-sample entropy for a batch of probability distributions.
#     probs: Tensor of shape [N, C] with softmax probabilities.
#     Returns: Tensor of shape [N] with entropy values.
#     """
#     return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


# def compute_kl_divergence(probs: torch.Tensor, batch_avg: torch.Tensor) -> torch.Tensor:
#     """
#     Compute per-sample KL divergence KL(batch_avg || probs).
#     probs: Tensor of shape [N, C], model predictions.
#     batch_avg: Tensor of shape [N, C], batch-level average for each sample's class.
#     Returns: Tensor of shape [N] with KL divergence values.
#     """
#     # F.kl_div input: log_probs, target_probs; reduction='none' gives [N, C] elementwise
#     kl_elementwise = F.kl_div(probs.log(), batch_avg, reduction='none')
#     return kl_elementwise.sum(dim=1)


# def compute_cosine_similarity(probs: torch.Tensor, batch_avg: torch.Tensor) -> torch.Tensor:
#     """
#     Compute per-sample cosine similarity between prediction and batch_avg.
#     Returns: Tensor of shape [N] with similarity in [-1, 1].
#     """
#     return F.cosine_similarity(probs, batch_avg, dim=1)


EPS = 1e-8  # small constant to avoid log(0)

def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample entropy for a batch of probability distributions, with clamping.
    probs: Tensor of shape [N, C] with softmax probabilities.
    Returns: Tensor of shape [N] with entropy values.
    """
    p = probs.clamp(min=EPS, max=1.0)
    return -(p * torch.log(p)).sum(dim=1)

def compute_kl_divergence(probs: torch.Tensor, batch_avg: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample KL divergence KL(batch_avg || probs), with clamping.
    probs: Tensor of shape [N, C], model predictions.
    batch_avg: Tensor of shape [N, C], batch-level average for each sample's class.
    Returns: Tensor of shape [N] with KL divergence values.
    """
    p = probs.clamp(min=EPS, max=1.0)
    q = batch_avg.clamp(min=EPS, max=1.0)
    return (q * (torch.log(q) - torch.log(p))).sum(dim=1)

def compute_cosine_similarity(probs: torch.Tensor, batch_avg: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample cosine similarity between prediction and batch_avg.
    Returns: Tensor of shape [N] with similarity in [-1, 1], with clamping to avoid zero-denom.
    """
    p = probs.clamp(min=EPS, max=1.0)
    q = batch_avg.clamp(min=EPS, max=1.0)
    return F.cosine_similarity(p, q, dim=1)
    
