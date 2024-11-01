# # # 민석 train
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     clip.py --base_path='/home/mango/LGD-CLIP' --batch_size 512 --lr 1e-6 --train_type lp --epochs 100 --resume "/home/mango/LGD-CLIP/clip_ckpt/last.pth.tar"
    
# # 민석 validate
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     clip.py --base_path='/home/mango/LGD2024/examples_old/imagenet' --batch_size 1024 --validate --train_type lp --resume "/home/mango/LGD-CLIP/clip_ckpt_LP/model_best.pth.tar"
    
# 민석 classification
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=1234 \
    clip.py --base_path='/home/mango/LGD-CLIP' --batch_size 1024 --classification --train_type lp --resume "/home/mango/LGD-CLIP/clip_ckpt_LP/last.pth.tar"

# 성욱
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     clip.py --base_path='/root/LGD2024/examples_old/imagenet' --batch_size 128 --lr 1e-4 --train_type ft --warmup_epochs 10