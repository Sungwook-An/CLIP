python second_train.py \
--imagenet_pretrained 'ImageNet_clip_ckpt/model_best_512.pth.tar' \
--batch_size 256 \
--lr 1e-6 \
--eta_min 1e-7 \
--warmup_epochs 0 \
--epochs 10 \
--projection_dim 512 \
--save_folder 'save_models/bsz256_lr1e-6_etaMin1e-8_epochs10_projDim512'
