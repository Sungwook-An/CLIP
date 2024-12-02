python second_train.py \
--imagenet_pretrained 'ImageNet_clip_ckpt/model_best_class_wise.pth.tar' \
--batch_size 128 \
--lr 1e-8 \
--eta_min 0.0 \
--warmup_epochs 0 \
--epochs 20 \
--projection_dim 512 \
--save_folder 'save_models/bsz256_lr1e-6_etaMin1e-8_epochs10_projDim512'
