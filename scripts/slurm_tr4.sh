#!/bin/bash

pwd=$(pwd)
cd ~
source ~/anaconda3/etc/profile.d/conda.sh
cd $pwd
ls
source $pwd/env.sh
ngpu=$(echo $CUDA_VISIBLE_DEVICES)
rank="$SLURM_PROCID"
echo rank=$rank
graph=$1
root=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo root=$root
hidden=$2
rep=$3
mm=$4
e=10
act="--activations=True"
norm="--normalization=True"
#act=""
norm=""
cmd="python -m torch.distributed.run --nproc_per_node=1 --nnodes=$SLURM_NTASKS  --node_rank=$rank --master_addr=$root --master_port=1234 src/gnn_rdm_nccl.py --accperrank=1 --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $norm $act --mmorder=$mm --replication=$rep"
echo $cmd
$cmd

