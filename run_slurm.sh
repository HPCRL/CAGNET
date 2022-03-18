
for graph in Reddit "ogbn-products" "ogbn-mag" "ogbn-arxiv"
do
    for hidden in 128 256 512 #1024
    do
        srun slurm_tr.sh $graph $hidden
        for replication in 1 2 4
        do
            srun slurm_15d.sh $graph $hidden $replication
        done
    done
done
