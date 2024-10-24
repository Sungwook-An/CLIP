# train.sh
export CUDA_VISIBLE_DEVICES=2
# nohup python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' -n /root/LGD2024/examples/imagenet/noisy_student_efficientnet-b1.pth --pretrained --workers 16 --batch-size 128 > eval_noisy_student.out &
python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --workers 4 --batch-size 512 --pretrained


#################### tf to py weights eval EfficientNet-B1 ####################
# Vanilla
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained --workers 8 --batch-size 512
# Noisy Student
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -n /root/LGD2024/examples/imagenet/tf_to_pt_weights/advprop_aa_efficientnet-b1.pth --workers 8 --batch-size 512
# AdvProp + AA
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --advprop --pretrained -n /root/LGD2024/examples/imagenet/tf_to_pt_weights/advprop_aa_efficientnet-b1.pth --workers 8 --batch-size 512
# AA
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -n /root/LGD2024/examples/imagenet/tf_to_pt_weights/aa_efficientnet-b1.pth --workers 8 --batch-size 512
