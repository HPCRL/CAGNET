#!/bin/bash

rank="$SLURM_PROCID"
echo rank=$rank
graph=$1
root="127.0.0.1"
echo root=$root
hidden=$2
np=2
e=2000
act="--activations=True"
norm="--normalization=True"
#act=""
norm=""
cmd="python -m torch.distributed.run --nproc_per_node=$np --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=12394 src/gcn_distr_graphsaint.py  --accperrank=$np --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm --accuracy=True "
#cmd="python -m torch.distributed.run --nproc_per_node=$np --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=12394 gcn_distr_acc_transpose.py  --accperrank=$np --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm --accuracy=True"
echo $cmd
$cmd

