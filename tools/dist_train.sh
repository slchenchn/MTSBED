set -x

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# echo ${PWD}
# echo $0
CUR_PATH=$(dirname "$0")
# echo $PYTHONPATH
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# automatically find a available port
LOWERPORT=23456
UPPERPORT=65535
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
