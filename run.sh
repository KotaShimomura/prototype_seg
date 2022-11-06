#/bin/bash

python -u main.py --drop_last y \
                --gathered n \
                --loss_balance y \
                --log_to_file n \
                --distributed \
                --max_iters 80000 \
                --train_batch_size 8 \
                --base_lr 0.01 \
                --data_dir /workspace/workspace/ProtoSeg_local/data/Cityscapes \
                --checkpoints_root /workspace/workspace/ProtoSeg_local/output/Cityscapes \
                --pretrained /workspace/workspace/ProtoSeg_local/checkpoints/cityscapes/hrnetv2_w48_imagenet_pretrained.pth
