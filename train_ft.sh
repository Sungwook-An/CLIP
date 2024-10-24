# train_ft.sh
export CUDA_VISIBLE_DEVICES=0

# without nohup
python finetune_main.py /database/PCB_DATASET/crop_images
