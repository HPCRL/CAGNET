import os
import sys
import torch

# Simple wrapper to run scripts on 1 node
dataset = sys.argv[1]
midlayer = sys.argv[2]
repliacation = sys.argv[3]

num_gpus = torch.cuda.device_count()

# launch scripts according to number of available gpus
if num_gpus>1:
    # run distribute script
    cmd = f'python -m torch.distributed.run --nproc_per_node={num_gpus} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 src/gcn_distr_transpose_15d_auto.py --accperrank={num_gpus} --epochs=100 --graphname={dataset} --timing=True --midlayer={midlayer} --runcount=1 --replication={repliacation}'
else:
    # run standard gcn script
    cmd = f'python src/gcn.py --graphname {dataset} --midlayer {midlayer}' 

print(cmd)
os.system(cmd)