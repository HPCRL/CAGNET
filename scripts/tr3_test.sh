

input=$1
hidden=$2
for mmorder in ssss sssd ssds ssdd sdss sdsd sdds sddd dsss dssd dsds dsdd ddss ddsd ddds dddd
do
    for rep in 1 2 4
    do
        cmd="srun scripts/slurm_tr3.sh $input $hidden $rep $mmorder"
        echo $cmd
        $cmd
    done
done
