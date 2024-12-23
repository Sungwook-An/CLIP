# # # 민석 train
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     lgd_clip.py --base_path='/home/mango/LGD-CLIP' --batch_size 512 --lr 1e-5 --train_type teft --epochs 30
    
# # 민석 validate
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     lgd_clip.py --base_path='/home/mango/LGD-CLIP' --batch_size 512 --validate --train_type lp --resume "/home/mango/LGD-CLIP/[60.192]clip_ckpt_LP/last.pth.tar"
    
# 민석 classification
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     lgd_clip.py --base_path='/home/mango/LGD-CLIP' --batch_size 1024 --classification --train_type lp --resume "/home/mango/LGD-CLIP/clip_ckpt_LP/last.pth.tar"

# 성욱 train
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=1234 \
    lgd_clip.py --base_path='/root/LGD2024/examples_old/imagenet' --batch_size 512 --lr 1e-5 --eta_min 1e-7 --train_type teft --epochs 10 --warmup_epochs 0 --class_wise

# # 성욱 validation
# CUDA_VISIBLE_DEVICES=2,3 torchrun \
#     --nproc_per_node=2 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=1234 \
#     lgd_clip.py --base_path='/root/LGD2024/examples_old/imagenet' --batch_size 512 --lr 1e-5 --eta_min 1e-7 --train_type teft --warmup_epochs 10 --validate --resume "/root/LGD2024/examples_old/imagenet/clip_ckpt/model_best.pth.tar"