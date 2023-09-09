#!/bin/bash
python -W ignore -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 ssl_brain_train_DDP.py --traindir=/train/train --testdir=/train/test --logdir=/train/log --cachedir=/cache --epochs=300
