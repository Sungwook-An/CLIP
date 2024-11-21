python lgd_server_clip.py \
--imagenet_pretrained 'ImageNet_clip_ckpt/model_best_512.pth.tar' \
--batch_size 128 \
--lr 1e-5 \
--eta_min 1e-7 \
--epochs 100 \
--save_folder 'save_models/first_test'

python main_ce_middle.py
