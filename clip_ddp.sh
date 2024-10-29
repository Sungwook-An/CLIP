# 민석
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     clip.py --batch_size 1024 --lr 1e-4 --train_type lp --warmup_epochs 0

# 성욱
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=1234 \
    clip.py --batch_size 128 --lr 1e-4 --train_type ft --warmup_epochs 10