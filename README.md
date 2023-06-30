# GNN-RDM
Repo for paper Communication Optimization for Distributed Execution of Graph Neural Networks.
This repo is built based on CAGNET: Communication-Avoiding Graph Neural nETworks

## Dependencies 
- Python 3.7.11
- torch                   1.9.1+cu111
- torch-cluster           1.5.9
- torch-geometric         2.0.1
- torch-scatter           2.0.8
- torch-sparse            0.6.12
- CUDA 11.0
- GCC 9.2.0
- ogb 
- sparse-extension 

The installation of torch-geometric can be a bit tricky. We recommend to install torch-scatter and torch-sparse using pre-built wheels at https://data.pyg.org/whl/torch-1.9.1%2Bcu111.html 

This code uses C++ extensions from CAGNET. To compile these, run

```bash
cd sparse-extension
python setup.py install
```

## Documentation: 

We reuse the flags from CAGNET: 

- `--accperrank <int>` : Number of GPUs on each node
- `--epochs <int>`  : Number of epochs to run training
- `--graphname <str> ` : Graph dataset to run training on
- `--timing <True/False>` : Enable timing barriers to time phases in training
- `--midlayer <int>` : Number of activations in the hidden layer
- `--runcount <int>` : Number of times to run training
- `--normalization <True/False>` : Normalize adjacency matrix in preprocessing
- `--activations <True/False>` : Enable activation functions between layers
- `--accuracy <True/False>` : Compute and print accuracy metrics 
- `--replication <int>` : Replication factor  
- `--download <True/False>` : Download datasets

## Running with slurm on RI2 

Our implementation of redistribution of dense matrices is at `src/gcn_distr_transpose_15d.py`

Run the following command to download the ogbn-products dataset:
`python src/gcn_distr_transpose_15d.py --graphname='ogbn-products' --download=True`

This will download ogbn-products into `../data`. After downloading the ogbn-products dataset, run the following command to run 1.5D and transoposing benchmarks

`bash run_slurm.sh`

This script outamatically runs benchmarks for 1.5D and transpose. However it is set for 1 GPU per node, which might not be the case in other systems. Also this script tests for Reddit, ogbn-products, ogbn-mag and ogbn-arxiv. If some of these are not downloaded before it will cause runtime errors. Accelerator per gpu parameters can be changed in slurm_tr.sh and slurm_15d.sh scripts for other systems.

## Running with torch.distributed.launch on CHPC (example)

Run the following command to download the Reddit dataset:

`python src/gcn_distr_transpose_15d.py --graphname=Reddit --download=True`

This will download Reddit into `../data`. After downloading the Reddit dataset, run the following command to run training

To run with torch.distributed.launch, MASTER_PORT, MASTER_ADDR, WORLD_SIZE, RANK are required. The training script is setting them and this may cause some issues. I disabled the lines setting these environment variables and only passed them through the command below in an interactive job:

`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=10.242.66.106 --master_port=61234 gcn_distr_transpose_15d.py --accperrank=1 --epochs=100 --graphname=Reddit --timing=True --midlayer=128 --runcount=1 --replication=1`

In a non-interactive job, the required environment variables can be obtained by 
`master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=12345
rank=$SLURM_PROCID
world_size=$SLURM_NTASKS`

Then they can be passed through a python command:

`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$world_size --node_rank=$rank --master_addr=$master_addr --master_port=$master_port gcn_distr_transpose_15d.py --accperrank=1 --epochs=100 --graphname=Reddit --timing=True --midlayer=128 --runcount=1 --replication=1`

## Citation

> Süreyya Emre Kurt, Jinghua Yan, Aravind Sukumaran-Rajam, Prashant Pandey, P. Sadayappan. Communication Optimization for Distributed Execution of Graph Neural Networks. Proceedings of the 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2023
