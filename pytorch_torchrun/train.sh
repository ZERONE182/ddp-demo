[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${MASTER_IP}" ] && MASTER_IP=172.20.115.3
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=$1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=$2

torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$2  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
    demo.py