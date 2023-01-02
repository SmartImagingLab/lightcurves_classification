# -*- coding: utf-8 -*-
# @Time : 2022/12/21 0:29
# @Author : lwb
# @File : train_variable.py

import sys
sys.path.append("./models")

import numpy as np
import argparse
import os
# 加载预先相关参数
from argparse import Namespace
from datasets.variable_Dataset import Variable_Dataset
from datasets.ConcatDataset import ConcatDataset
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from models.TransformerEncoder import TransformerEncoder
from utils.scheduled_optimizer import ScheduledOptim
from utils.logger import Logger
import torch.optim as optim
from utils.trainer import Trainer


name_list = ["APER", "CEP", "HADS", "MIRA", "PER", "QPER", "RRC"]

TUM_dataset_random_split = Namespace(
    dataset = "variable_data",
    trainregions = name_list,
    testregions = name_list,
    scheme="random",
    mode="traintest",
    test_on = "test",
    train_on = "train",
    samplet = 70     # 定义序列最大长度
)

# from hyperparameter import select_hyperparameter
# def get_hyperparameter_args():
#     return select_hyperparameter(args.experiment, args.hparamset, args.hyperparameterfolder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataroot', type=str, default='../data/variable_interp', help='root to dataset. default ../data')
#     parser.add_argument(
#         '--dataroot', type=str, default='../data', help='root to dataset. default ../data')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=256, help='batch size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=150, help='number of training epochs')    # 初始:150
    parser.add_argument(
        '-l', '--learning_rate', type=float, default=0.255410, help='learning rate')    # 初始:0.255410
    parser.add_argument(
        '--weight_decay', type=float, default=0.000413, help='weight_decay')    # 初始:0.000413
    parser.add_argument(
        '--warmup', type=int, default=1000, help='warmup')    # 初始:1000
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite automatic snapshots if they exist")
    parser.add_argument(
        '--classmapping', type=str, default=None, help='classmapping')
    parser.add_argument(
        '--hyperparameterfolder', type=str, default=None, help='hyperparameter folder')
    parser.add_argument(
        '-x', '--experiment', type=str, default="train", help='experiment prefix')  # 这个参数干嘛用的？
    parser.add_argument(
        '--store', type=str, default="./tmp", help='store run logger results')
    parser.add_argument(
        '--test_every_n_epochs', type=int, default=1, help='skip some test epochs for faster overall training')
    parser.add_argument(
        '--checkpoint_every_n_epochs', type=int, default=5, help='save checkpoints during training')
    parser.add_argument(
        '--seed', type=int, default=0, help='seed for batching and weight initialization')
    parser.add_argument(
        '--hparamset', type=int, default=0, help='rank of hyperparameter set 0: best hyperparameter')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=1, help='show n samples in visdom')
    args, _ = parser.parse_known_args()

    return args

def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)


def train(args):

    args.experiment = "variable"
    args.mode = None
    args = merge([args, TUM_dataset_random_split])
    print(args.mode)
    # args.trainregions = ["holl"]
    # args.testregions = ["holl"]
    args.classmapping = os.path.join(args.dataroot, "classmapping7.csv")
    print(args)

    root = args.dataroot

    # ImbalancedDatasetSampler
    test_dataset_list = list()
    for region in args.testregions:
        test_dataset_list.append(
            Variable_Dataset(root=root, region=region, partition=args.test_on,
                                 classmapping=args.classmapping, samplet=args.samplet,
                                 scheme=args.scheme, mode=args.mode, seed=args.seed)
        )

    print(test_dataset_list)

    train_dataset_list = list()
    for region in args.trainregions:
        train_dataset_list.append(
            Variable_Dataset(root=root, region=region, partition=args.train_on,
                                 classmapping=args.classmapping, samplet=args.samplet,
                                 scheme=args.scheme, mode=args.mode, seed=args.seed)
        )

    print("setting random seed to " + str(args.seed))
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.random.manual_seed(args.seed)

    traindataset = ConcatDataset(train_dataset_list)
    traindataloader = torch.utils.data.DataLoader(dataset=traindataset, sampler=RandomSampler(traindataset),
                                                  batch_size=args.batchsize, num_workers=args.workers)

    testdataset = ConcatDataset(test_dataset_list)

    testdataloader = torch.utils.data.DataLoader(dataset=testdataset, sampler=SequentialSampler(testdataset),
                                                 batch_size=args.batchsize, num_workers=args.workers)

    args.nclasses = traindataloader.dataset.nclasses
    classname = traindataloader.dataset.classname
    klassenname = traindataloader.dataset.klassenname
    args.seqlength = traindataloader.dataset.sequencelength
    # args.seqlength = args.samplet
    args.input_dims = traindataloader.dataset.ndims
    print(args)
    # print(classname)
    # print(klassenname)

    args.model = "transformer"
    if args.model == "transformer":
        #     hidden_dims = args.hidden_dims # 256
        #     n_heads = args.n_heads # 8
        #     n_layers = args.n_layers # 6
        hidden_dims = 128
        n_heads = 3
        n_layers = 3
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        dropout = 0.262039
        warmup = args.warmup

        len_max_seq = args.samplet  # 定义序列最大长度
        d_inner = hidden_dims * 4

        #     # 补充！
        #     fields = ["hidden_dims",
        #     "n_heads",
        #     "n_layers",
        #     "weight_decay",
        #     "learning_rate",
        #     "warmup",
        #     "dropout"]
        #     dtypes = [int, int, int, float, float, int, float]

        model = TransformerEncoder(in_channels=args.input_dims, len_max_seq=len_max_seq,
                                   d_word_vec=hidden_dims, d_model=hidden_dims, d_inner=d_inner,
                                   n_layers=n_layers, n_head=n_heads, d_k=hidden_dims // n_heads,
                                   d_v=hidden_dims // n_heads,
                                   dropout=dropout, nclasses=args.nclasses)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("initialized {} model ({} parameters)".format(args.model, pytorch_total_params))

    # from utils.visdomLogger import VisdomLogger  # 是否增加可视化部分
    # visdomenv = "{}_{}".format(args.experiment, args.dataset)
    # visdomlogger = VisdomLogger(env=visdomenv)
    visdomlogger = None
    store = os.path.join(args.store, args.experiment)
    logger = Logger(columns=["accuracy"], modes=["train", "test"], rootpath=store)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=weight_decay),
        model.d_model, warmup)

    config = dict(
        epochs=args.epochs,
        learning_rate=learning_rate,
        show_n_samples=args.show_n_samples,
        store=store,
        visdomlogger=visdomlogger,
        overwrite=args.overwrite,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        test_every_n_epochs=args.test_every_n_epochs,
        logger=logger,
        optimizer=optimizer
    )

    trainer = Trainer(model, traindataloader, testdataloader, **config)
    logger = trainer.fit()
    logger.save()


if __name__=="__main__":

    args = parse_args()
    train(args)
