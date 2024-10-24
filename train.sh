# train.sh
export CUDA_VISIBLE_DEVICES=2,3

# nohup python main.py /database/Data/CLS-LOC --arch 'efficientnet-b1' --workers 8 --batch-size 256 --epochs 120 --seed 1 --lr 0.05 > ./nohup/b1_0.05_0704_1730.out &

# with resume
# nohup python main.py /database/Data/CLS-LOC --arch 'efficientnet-b1' --workers 8 --batch-size 256 --epochs 150 --seed 1 --lr 0.05 --resume /root/LGD2024/examples/imagenet/checkpoint_b1_1719369595_0.05.pth.tar > ./nohup/b1_0.05_resume_0701_0830.out &

# without nohup
python main.py /database/Data/CLS-LOC --arch 'efficientnet-b1' --workers 8 --batch-size 256 --epochs 300 --seed 42 --lr 0.03
